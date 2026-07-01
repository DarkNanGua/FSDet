from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import global_variables as g

# class MultiScaleDistillLoss(nn.Module):
#     def __init__(self, channel=[], is_use=False):
#         super().__init__()
        
#         self.is_use = is_use
#         self.mse = nn.MSELoss()


#     def forward(self, features):
        
#         if self.is_use is False or self.training is False:
#             return .0
        
#         gas = g.GAS

#         return 


class MultiScaleDistillLoss(nn.Module):
    def __init__(self, channel=[], is_use=False, warmup=0, end_epoch=90, 
                 min_weight=0.2, max_weight=1, weights=[1,0.7,0.5, 0.1],
                 threshold=0.6, distance_type='l2', epoch_interval=1):
        super().__init__()
        
        self.is_use = is_use
        self.mse = nn.MSELoss(reduction='none')  # 改为不自动求平均 
        self.weights = weights  # 各尺度损失权重 
        self.warmup = warmup  # 预热步数 
        self.cur_epoch = 0
        self.cur_step = 0
        self.end_epoch = end_epoch 
        self.threshold = threshold  # p5_scale的阈值 
        self.distance_type = distance_type  # 距离类型：'l1', 'l2', 'smooth_l1' 
        self.epoch_interval = epoch_interval  # epoch间隔 
        
        total_epochs = self.end_epoch - self.warmup
        self.scheduler = CosineAnnealingScheduler(
            max_epochs=total_epochs,
            min_weight=min_weight,
            max_weight=max_weight,
            restart_cycle=20,  # 每20 epoch重启一次 
            epoch_interval=epoch_interval  # 传递epoch间隔参数 
        )

        self.p5_channel = channel[-1]
        
        self.loss = DensityRecallFocalLoss(beta=0.05, reduction='mean')
        self.cv1 = nn.Conv2d(self.p5_channel, 1, kernel_size=1, stride=1, padding=0)
        
        
    def compute_focal_distance_loss(self, bb, pan, p5_scale):
        """
        使用focal loss思想的距离损失
        Args:
            bb: backbone特征
            pan: PAN特征
            p5_scale: p5尺度特征(已sigmoid)
        """
        if self.distance_type == 'l1':
            base_loss = F.l1_loss(bb, pan, reduction='none')
        elif self.distance_type == 'l2':
            base_loss = self.mse(bb, pan)
        elif self.distance_type == 'smooth_l1':
            base_loss = F.smooth_l1_loss(bb, pan, reduction='none')
        
        # 使用p5_scale作为难度权重，值越高表示越重要
        alpha = 1.5  # focal loss的调节参数
        focal_weight = torch.pow(p5_scale, alpha)
        
        # 扩展权重到特征维度
        if len(base_loss.shape) == 4:  # [B, C, H, W]
            focal_weight = focal_weight.expand_as(base_loss)
            
        weighted_loss = base_loss * focal_weight
        return weighted_loss.mean()
    
    def forward(self, features):
        
        # 计算epoch
        self.cur_step = (self.cur_step + 1 ) % 810
        if self.cur_step == 0:
            self.cur_epoch += 1

        if self.is_use is False or self.training is False or self.cur_epoch < self.warmup or self.cur_epoch >= self.end_epoch:
            return .0
        
        # 检查是否在epoch间隔内
        if (self.cur_epoch - self.warmup) % self.epoch_interval != 0:
            return .0
        
        gas = g.GAS
        if gas is None:
            return .0
        # list 转换成batch
        if isinstance(gas, list):
            gas = torch.stack(gas, dim=0).to(features[0].device)
            gas = gas.unsqueeze(1)/0.0711

        p5 = features[-1]
        features = features[:-1]
        # current_weight = self.scheduler.get_weight()
        current_weight = 0.05
        len_ = len(features)
        bb_features = features[:len_//2]  # backbone特征
        pan_features = features[len_//2:]  # PAN特征
        scales = [bb_features[i].shape[-1] for i in range(len_//2)]
        total_loss = 0

        p5_proj = self.cv1(p5)  # [B, 1, H, W]
        p5_sigmoid = p5_proj.sigmoid() 
        # p5_softmax = p5_proj.softmax(-1) 
        # p5_loss = F.binary_cross_entropy_with_logits(p5_proj, gas)  # [B, 1, H, W]
        p5_loss = self.loss(p5_sigmoid, gas)  # [B, 1, H, W]

        for i, (bb, pan) in enumerate(zip(bb_features, pan_features)):
            # 将p5_sigmoid插值到当前尺度
            p5_scale = F.interpolate(gas.detach(), size=bb.shape[-2:],mode="nearest")
               
            # loss
            scale_loss = self.compute_focal_distance_loss(bb, pan, p5_scale) * self.weights[i]
            # scale_loss = self.loss(bb, pan.detach(), p5_scale) * self.weights[i]
            total_loss += scale_loss

        return total_loss * current_weight


class CosineAnnealingScheduler:
    def __init__(self, 
                 max_epochs, 
                 min_weight=0.1, 
                 max_weight=0.5,
                 restart_cycle=0,
                 epoch_interval=10):
        """
        restart_cycle: 重启周期（0表示不重启）
        epoch_interval: epoch间隔，只有在指定间隔的epoch才会更新权重
        """
        self.max_epochs = max_epochs
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.restart_cycle = restart_cycle
        self.epoch_interval = epoch_interval
        self.current_step = 0
        
    def get_weight(self):
        # 计算当前有效步数（考虑周期重启）
        if self.restart_cycle > 0:
            effective_step = self.current_step % self.restart_cycle
            cycle_ratio = effective_step / self.restart_cycle
        else:
            cycle_ratio = self.current_step / self.max_epochs
        
        # 余弦计算：从弱到强（原来是从强到弱）
        # 使用 (1 - cosine) 实现从弱到强的变化
        cosine = np.cos(np.pi * cycle_ratio)
        weight = self.min_weight + 0.5*(self.max_weight - self.min_weight)*(1 - cosine)
        
        # 只有在epoch间隔时才更新step
        self.current_step += 1
        return weight
    
    
class DensityRecallFocalLoss(nn.Module):
    def __init__(self, beta=0.5, reduction='mean'):
        super(DensityRecallFocalLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred_density, gt_density):
        """
        pred_density: [B, 1, H, W] -> 模型预测的密度图，经过 sigmoid 之后
        gt_density:   [B, 1, H, W] -> 高斯生成的密度图 Ground Truth
        """

        # 保证形状一致
        assert pred_density.shape == gt_density.shape, "Shape mismatch between prediction and ground truth."

        # α_{i,j} = d_{i,j}^{gt}
        alpha = gt_density

        # MSE 误差部分：α_{i,j} * (d_pred - d_gt)^2
        mse_loss = alpha * (pred_density - gt_density) ** 2

        # underestimation mask: 𝟙(pred < gt)
        under_mask = (pred_density < gt_density).float()

        # Focal penalty term：β * 𝟙(pred < gt) * d_{i,j}^{gt}
        focal_penalty = self.beta * under_mask * gt_density

        # DRFL loss 总和
        loss = mse_loss + focal_penalty

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()//8
        elif self.reduction == 'sum':
            return loss.sum()//8
        else:
            return loss  # 不做 reduction