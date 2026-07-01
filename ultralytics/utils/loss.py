# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

import global_variables
from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, bbox_ps, probiou
from .tal import bbox2dist

class MatchabilityAwareLoss(nn.Module):
    """
    Matchability-Aware Loss (MAL) based on:
    https://arxiv.org/abs/2008.13367 (Varifocal Loss) but modified as in Section 3.3 of MAL paper.
    """

    def __init__(self, gamma=1.5, mal_alpha=None):
        """
        Args:
            gamma (float): exponent for q and p.
            mal_alpha (float or None): optional scaling factor for negative samples (like α in VFL).
        """
        super().__init__()
        self.gamma = gamma
        self.mal_alpha = mal_alpha

    def forward(self, pred_score, gt_iou, label):
        """
        Args:
            pred_score: [B, N] raw logits from classifier
            gt_iou: [B, N] IoU between matched pred boxes and GT boxes (q)
            label: [B, N] binary labels (1 for matched, 0 for unmatched)
        """
        # Step 1: target = q^gamma for positives, 0 for negatives
        target = (gt_iou ** self.gamma) * label  # 正样本 q^γ，负样本 0

        # Step 2: weight for negatives = p^γ, positives weight=1
        p_sigmoid = pred_score.sigmoid().detach()
        if self.mal_alpha is not None:
            weight = self.mal_alpha * (p_sigmoid ** self.gamma) * (1 - label) + label
        else:
            weight = (p_sigmoid ** self.gamma) * (1 - label) + label

        # Step 3: BCE with target and weight
        loss = (
            F.binary_cross_entropy_with_logits(pred_score.float(), target.float(), reduction="none") * weight
        )
        loss = loss.mean(1).sum()
        return loss


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)

class IKLDLoss(nn.Module):
    """
    Improved Kullback-Leibler Divergence (IKLD) Loss for tiny object detection.
    
    This loss function models bounding boxes as 2D Gaussian distributions
    and computes KL divergence normalized by area.
    """
    
    def __init__(self, lambda_factor: float = 1.0, eps: float = 1e-7):
        """
        Args:
            lambda_factor: Control parameter for loss sensitivity
            eps: Small value for numerical stability
        """
        super(IKLDLoss, self).__init__()
        self.lambda_factor = lambda_factor
        self.eps = eps
    
    def forward(
        self, 
        pred_boxes: torch.Tensor,  # Predicted boxes [N, 4] or [..., 4]
        target_boxes: torch.Tensor,  # Target boxes [N, 4] or [..., 4]
        xywh: bool = True  # If True, boxes are in (x, y, w, h) format
    ) -> torch.Tensor:
        """
        Compute IKLD loss between predicted and target boxes.
        
        Args:
            pred_boxes: Predicted bounding boxes
            target_boxes: Target bounding boxes  
            xywh: Box format flag
            
        Returns:
            IKLD loss values
        """
        
        # Convert to center-size format if needed
        if xywh:
            pred_x, pred_y, pred_w, pred_h = pred_boxes.chunk(4, dim=-1)
            target_x, target_y, target_w, target_h = target_boxes.chunk(4, dim=-1)
        else:
            # Convert from (x1, y1, x2, y2) to (cx, cy, w, h)
            pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.chunk(4, dim=-1)
            target_x1, target_y1, target_x2, target_y2 = target_boxes.chunk(4, dim=-1)
            
            pred_x = (pred_x1 + pred_x2) / 2
            pred_y = (pred_y1 + pred_y2) / 2
            pred_w = pred_x2 - pred_x1
            pred_h = pred_y2 - pred_y1
            
            target_x = (target_x1 + target_x2) / 2
            target_y = (target_y1 + target_y2) / 2
            target_w = target_x2 - target_x1
            target_h = target_y2 - target_y1
        
        # Ensure positive dimensions
        pred_w = torch.clamp(pred_w, min=self.eps)
        pred_h = torch.clamp(pred_h, min=self.eps)
        target_w = torch.clamp(target_w, min=self.eps)
        target_h = torch.clamp(target_h, min=self.eps)
        
        # Compute KL divergence components
        kl_divergence = self._compute_kl_divergence(
            pred_x, pred_y, pred_w, pred_h,
            target_x, target_y, target_w, target_h
        )
        
        # Compute area normalization
        area_sum = self._compute_area_sum(pred_w, pred_h, target_w, target_h)
        
        # Basic IKLD loss
        ikld_basic = kl_divergence / (area_sum + self.eps)
        
        # Normalized IKLD loss
        ikld_normalized = torch.exp(-self.lambda_factor * torch.sqrt(ikld_basic + self.eps))
        
        return ikld_normalized
    
    def _compute_kl_divergence(
        self,
        pred_x: torch.Tensor, pred_y: torch.Tensor, 
        pred_w: torch.Tensor, pred_h: torch.Tensor,
        target_x: torch.Tensor, target_y: torch.Tensor,
        target_w: torch.Tensor, target_h: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between two 2D Gaussian distributions.
        
        For diagonal covariance matrices:
        Σ_p = diag([w_p²/4, h_p²/4])
        Σ_t = diag([w_t²/4, h_t²/4])
        """
        
        # Center differences
        delta_x = pred_x - target_x
        delta_y = pred_y - target_y
        
        # Variance terms (σ² = (w/2)² = w²/4)
        pred_var_x = pred_w.pow(2) / 4
        pred_var_y = pred_h.pow(2) / 4
        target_var_x = target_w.pow(2) / 4
        target_var_y = target_h.pow(2) / 4
        
        # KL divergence formula for diagonal Gaussians:
        # KL(p||t) = 1/2 * [Σ_i ((μ_p_i - μ_t_i)²/σ_t_i² + σ_p_i²/σ_t_i² + ln(σ_t_i²/σ_p_i²)) - d]
        # where d is the dimensionality (d=2 for 2D)
        
        # Mean difference term: (μ_p - μ_t)^T Σ_t^(-1) (μ_p - μ_t)
        mean_diff_term = (delta_x.pow(2) / target_var_x + 
                         delta_y.pow(2) / target_var_y)
        
        # Trace term: Tr(Σ_t^(-1) Σ_p)
        trace_term = (pred_var_x / target_var_x + 
                     pred_var_y / target_var_y)
        
        # Log determinant term: ln(|Σ_t|/|Σ_p|)
        log_det_term = (torch.log(target_var_x / (pred_var_x + self.eps)) + 
                       torch.log(target_var_y / (pred_var_y + self.eps)))
        
        # Complete KL divergence
        kl_div = 0.5 * (mean_diff_term + trace_term + log_det_term - 2)
        
        return kl_div
    
    def _compute_area_sum(
        self,
        pred_w: torch.Tensor, pred_h: torch.Tensor,
        target_w: torch.Tensor, target_h: torch.Tensor
    ) -> torch.Tensor:
        """Compute sum of areas of two bounding boxes."""
        pred_area = pred_w * pred_h
        target_area = target_w * target_h
        return pred_area + target_area

class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.ratio = 1.2
        self.count = 1

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False,CIoU=True)
        # iou_ps = bbox_ps(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        self.count += 1

        loss_iou = ((1.0 - iou ) * weight).sum() / target_scores_sum 
        # ikld_values = self.ikld_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
        # Convert IKLD to loss (since IKLD returns similarity, we need 1-IKLD for loss)
        # loss_ikld = ((1.0-ikld_values )* weight.squeeze(-1)).sum() / target_scores_sum
        # out_iou = 0.3*loss_iou + 0.7*loss_ikld

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.vfl = MatchabilityAwareLoss()

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=4, num_classes=self.nc, alpha=0.7, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.contrastive_head = nn.Sequential(
            nn.Linear(64, 32,bias=False,device="cuda"),
        )

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0].long()  # 确保是int64类型  # image index
            _, counts = i.unique(return_counts=True) #counts表示batch每个图像的gtbox数量
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        # 预处理preds
        distill_loss = None
        if isinstance(preds, tuple) is False:
            distill_loss = preds.pop(-1)
            
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds # [16,144,80,80] [16,144,40,40] [16,144,20,20]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        ) # 拼接 使得20*20+40*40+80*80 = 8400

        pred_scores = pred_scores.permute(0, 2, 1).contiguous() # [16,80,8400]->[16,8400,80]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous() # [16,64,8400]->[16,8400,64]

        # 生成 anchor
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # 处理Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        # neg_labels, neg_bboxes, neg_src_idx = global_variables.sample_negative_samples(
        #     gt_labels, gt_bboxes, 10, n=100, box_noise_scale=0.1
        #     )
        # nos_mask_gt = neg_bboxes.sum(2, keepdim=True).gt_(0.0)
        
        # nos_labels, nos_bboxes, nos_scores, nos_fg_mask, nos_gt_idx = self.assigner(
        #     # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
        #     pred_scores.detach().sigmoid(),
        #     (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        #     anchor_points * stride_tensor,
        #     neg_labels,
        #     neg_bboxes,
        #     nos_mask_gt,
        # )
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx,mask_nor = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        features = global_variables.prepare_features_from_head_features(global_variables.head_features)
        class_avg_feature = global_variables.get_class_prototype_vectors(features, gt_labels, target_gt_idx, fg_mask)
        class_best_feature = global_variables.get_topk_feature_per_class(features, target_scores, gt_labels, target_gt_idx, fg_mask)
        class_avg_feature_norm = self.contrastive_head(class_avg_feature[0])
        class_best_feature_norm = self.contrastive_head(class_best_feature[0])
        class_avg_feature_norm = F.normalize(class_avg_feature_norm, p=2, dim=-1)
        class_best_feature_norm = F.normalize(class_best_feature_norm, p=2, dim=-1)
        logits = class_avg_feature_norm @ class_best_feature_norm.t() 
        targets = torch.arange(class_avg_feature_norm.shape[0], dtype=torch.long,device="cuda")
        loss_contrast = F.cross_entropy(logits, targets) * 0.05
        
        # ##### mask loss
        # pred_mask = self.create_mask(distill_loss)[1]
        # tar_obj_mask, tar_bg_mask = self.build_tarmask(pred_mask, batch)
        # tar_obj_mask = F.max_pool2d(tar_obj_mask,kernel_size=3, stride=1, padding=1)

        # pred_obj_mask = pred_mask*tar_obj_mask
        # pred_bg_mask = pred_mask-pred_obj_mask
        # n_pix = tar_bg_mask.shape[2]*tar_bg_mask.shape[3]*tar_bg_mask.shape[0]
        # l_obj_mask = F.mse_loss(tar_obj_mask, pred_obj_mask)*n_pix / (float(sum(sum(sum(sum(tar_obj_mask)))))+1.0)
        # l_bg_mask = F.mse_loss(tar_bg_mask, pred_bg_mask)*n_pix / (n_pix-float(sum(sum(sum(sum(tar_obj_mask)))))+1.0)
        # lmask += 5 * l_obj_mask + l_bg_mask*0.5

        
        # # pred_scores: [B, N, C]
        # pred_cls = pred_scores.argmax(dim=-1)  # [B, N], 每个位置的预测类别
        # gt_labels_flat = gt_labels.squeeze(-1)  # [B, M]

        # # 获取每个预测框 assign 到的 GT 的真实类别
        # B, N = target_gt_idx.shape
        # batch_idx = torch.arange(B).unsqueeze(1).expand(B, N).to(target_gt_idx.device)
        # assigned_gt_labels = gt_labels_flat[batch_idx, target_gt_idx]  # [B, N]

        # # 找到预测类别与真实 GT 类别一致的位置
        # correct_cls_mask = (pred_cls == assigned_gt_labels)  # [B, N]

        # # 同时满足：是前景 & 预测正确
        # fg_mask_clean = fg_mask & correct_cls_mask

        # noisy_feats, clean_feats = global_variables.sample_noise_clean_feature_pairs_fast(
        #     nos_fg_mask, nos_gt_idx,
        #     fg_mask_clean, target_gt_idx,
        #     global_variables.head_features,
        #     strides=self.stride    
        # )
        
        
        # # target_gt_idx_scale, fg_mask_scale = self.reshape_to_multiscale(target_gt_idx, fg_mask, self.stride, imgsz)
        # if global_variables.head_features is not None or len(global_variables.head_features) > 0:
        #     # pos_features, neg_features, pos_gt_idx, valid_pairs = global_variables.sample_negative_around_positive_fast(target_gt_idx_scale, fg_mask_scale, global_variables.head_features)
        #     if any(pos_feat.shape[0] > 0 for pos_feat in clean_feats):
        #         similarity_loss, scale_losses = global_variables.compute_similarity_loss_fast(
        #             [clean_feats], [noisy_feats], [None], 
        #             temperature=0.2, loss_type='cosine'
        #         )
                
        #         # print(f"\n相似度损失:")
        #         # print(f"Total loss: {similarity_loss.item():.4f}")
        #         # for i, loss_ in enumerate(scale_losses):
        #         #     print(f"Scale {i} loss: {loss_.item():.4f}")
        #     else:
        #         print("没有找到有效的正样本进行相似度计算")
            
        target_scores_sum = max(target_scores.sum(), 1)
        
        

        # Cls loss
        # target_labels = target_labels.unsqueeze(-1).expand(-1, -1, self.nc)  # self.nc: class num
        # one_hot = torch.zeros(target_labels.size(), device=self.device)
        # one_hot.scatter_(-1, target_labels, 1)
        # target_labels = one_hot
        # loss[1] = self.vfl(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # mask_nor = (1-mask_nor)
        # mask_nor = (mask_nor > 0).any(dim=1).float()
        # mask_nor = (1 - mask_nor * (1.0 - fg_mask.float())).unsqueeze(-1)
        # loss[1] = (mask_nor * self.bce(pred_scores, target_scores.to(dtype))).sum() / target_scores_sum  # BCE
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        if distill_loss is not None:
            loss[3] = distill_loss
        loss[3] = loss_contrast
        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    
    def reshape_to_multiscale(self, target_gt_idx, fg_mask, strides, input_size):
        """
        将单一尺度的target_gt_idx和fg_mask重新整理为多尺度格式
        
        Args:
            target_gt_idx (Tensor): shape [B, A] - 单一尺度的GT索引
            fg_mask (Tensor): shape [B, A] - 单一尺度的前景掩码
            strides (list): 每个尺度的步长，如 [8, 16, 32]
            input_size (int or tuple): 输入图像尺寸，如 640 或 (640, 640)
        
        Returns:
            target_gt_idx_multiscale (list): [Tensor([B, H, W]), ...] 每个尺度的GT索引
            fg_mask_multiscale (list): [Tensor([B, H, W]), ...] 每个尺度的前景掩码
        """
        if isinstance(input_size, int):
            input_h, input_w = input_size, input_size
        else:
            input_h, input_w = input_size
        
        batch_size = target_gt_idx.shape[0]
        device = target_gt_idx.device
        
        target_gt_idx_multiscale = []
        fg_mask_multiscale = []
        
        anchor_start = 0
        
        for stride in strides:
            # 计算当前尺度的特征图尺寸
            feat_h = int(input_h // stride)
            feat_w = int(input_w // stride)
            num_anchors_current_scale = int(feat_h * feat_w)
            
            # 提取当前尺度的数据
            anchor_end = anchor_start + num_anchors_current_scale
            
            current_target_gt_idx = target_gt_idx[:, anchor_start:anchor_end]  # [B, H*W]
            current_fg_mask = fg_mask[:, anchor_start:anchor_end]  # [B, H*W]
            
            # 重塑为 [B, H, W] 格式
            current_target_gt_idx = current_target_gt_idx.reshape(batch_size, feat_h, feat_w)
            current_fg_mask = current_fg_mask.reshape(batch_size, feat_h, feat_w)
            
            target_gt_idx_multiscale.append(current_target_gt_idx)
            fg_mask_multiscale.append(current_fg_mask)
            
            anchor_start = anchor_end
        
        return target_gt_idx_multiscale, fg_mask_multiscale
    
    def build_tarmask(self, mask, targets):
        tar_obj_mask = torch.zeros_like(mask)
        tar_bg_mask = torch.zeros_like(mask)
        for i in range(targets.shape[0]):
            mask_w = int(mask.shape[-2])
            mask_h = int(mask.shape[-1])

            batch = int(targets[i][0])
            cx = float(targets[i][2]) * mask_w
            cy = float(targets[i][3]) * mask_h
            w = float(targets[i][4]) * mask_w
            h = float(targets[i][5]) * mask_h

            xmin = int(cx - w/2 if cx-w/2 > 0 else 0)
            ymin = int(cy - h/2 if cy-h/2 > 0 else 0)
            xmax = int(cx + w/2 if cx+w/2 < mask_w-1 else mask_w-1)
            ymax = int(cy + h/2 if cy+h/2 < mask_h-1 else mask_h-1)

            tar_obj_mask[batch][0][xmin][ymin]=1
            if xmin != xmax:
                tar_obj_mask[batch][0][xmax][ymin]=1
            if ymin != ymax:
                tar_obj_mask[batch][0][xmin][ymax]=1
            if xmin != xmax and ymin != ymax:
                tar_obj_mask[batch][0][xmax][ymax]=1
        return tar_obj_mask, tar_bg_mask


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        distill_loss = None
        if isinstance(preds[0], tuple) is False:
            if isinstance(preds[0], torch.Tensor) is False:
                distill_loss = preds[0].pop(-1)


        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        
        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        if distill_loss is not None:
            loss[3] = distill_loss

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]

class MaskDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaskDecoder, self).__init__()
        
        # 上采样 + 卷积层
        self.upconv1 = nn.ConvTranspose2d(in_channels, 512, kernel_size=4, stride=2, padding=1)  # 上采样至原图大小
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # 卷积
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 上采样
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 卷积
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 上采样
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 卷积
        self.conv4 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)  # 输出掩码

    def forward(self, x):
        # 上采样和卷积
        x = F.relu(self.upconv1(x))  # 上采样
        x = F.relu(self.conv1(x))    # 卷积

        x = F.relu(self.upconv2(x))  # 上采样
        x = F.relu(self.conv2(x))    # 卷积

        x = F.relu(self.upconv3(x))  # 上采样
        x = F.relu(self.conv3(x))    # 卷积

        # 输出最终的掩码
        x = self.conv4(x)
        return torch.sigmoid(x)  # 使用Sigmoid生成二值掩码
    
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=True)  # 全局平均池化
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y  # 按通道加权
    
class CreatMask(nn.Module):
    def __init__(self, c1):#
        super(CreatMask, self).__init__()
        # 注意力模块
        self.se_block = SEBlock(c1)
        # Mask Decoder
        self.mask_decoder = MaskDecoder(c1, 1)  # 输出1通道的掩码
        self.conv = nn.Conv2d(c1, 1, kernel_size=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
 
    def forward(self, x):
        # 加入注意力机制
        x = self.se_block(x)
        # 使用mask decoder生成掩码
        out_put = self.mask_decoder(x)
        mask = torch.where(out_put > 0.5, 1.0, 0.0)
        if mask.dtype == torch.float32 and out_put.dtype == torch.float16:
            mask = mask.half()
        
        return [out_put, mask]
