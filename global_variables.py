from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

GAS = None
head_features = None
proj = torch.nn.Conv2d(128,128,1)
def convert_image_to_npz_path(image_path):
    """将图片路径转换为npz文件路径"""
    p = Path(image_path)
    # 替换 'images' 为 'output'，并改变文件扩展名
    return str(p.parent.parent / 'output' / (p.stem + '.npz'))

def convert_paths_batch(image_paths):
    """批量转换路径"""
    return [convert_image_to_npz_path(path) for path in image_paths]

def load_npz_to_tensor(npz_path):
    """读取单个npz文件并转换为tensor"""
    data = np.load(npz_path)
    # 假设npz文件中有一个主要的数组，通常是第一个key
    key = list(data.keys())[0]  # 或者指定具体的key
    array = data[key]
    return torch.from_numpy(array)

def load_npz_batch(npz_paths):
    """批量读取npz文件"""
    tensors = []
    for path in npz_paths:
        try:
            tensor = load_npz_to_tensor(path)
            tensors.append(tensor)
        except Exception as e:
            print(f"读取 {path} 时出错: {e}")
    return tensors
def sample_negative_around_positive(target_gt_idx_list, fg_mask_list, feature_maps, 
                                   kernel_size=3, sample_num=1, seed=None):
    """
    在正样本周围的邻域中采样负样本用于相似度损失
    不使用padding，边缘位置只在有效范围内采样
    
    Args:
        target_gt_idx_list (list): 多尺度GT索引列表 [Tensor([B, H, W]), ...]
        fg_mask_list (list): 多尺度前景掩码列表 [Tensor([B, H, W]), ...]
        feature_maps (list): 多尺度特征图列表 [Tensor([B, C, H, W]), ...]
        kernel_size (int): 邻域大小，默认3x3
        sample_num (int): 每个正样本周围采样的负样本数量
        seed (int): 随机种子
        
    Returns:
        pos_features (list): 每个尺度的正样本特征 [Tensor([N_pos, C]), ...]
        neg_features (list): 每个尺度的负样本特征 [Tensor([N_pos*sample_num, C]), ...]
        pos_gt_idx (list): 每个尺度的正样本GT索引 [Tensor([N_pos]), ...]
        valid_pairs (list): 每个尺度的有效配对数量
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    pos_features_all = []
    neg_features_all = []
    pos_gt_idx_all = []
    valid_pairs_all = []
    
    radius = kernel_size // 2  # 邻域半径
    
    for scale_idx, (target_gt_idx, fg_mask, feat_map) in enumerate(
        zip(target_gt_idx_list, fg_mask_list, feature_maps)):
        
        batch_size, channels, feat_h, feat_w = feat_map.shape
        
        # 存储当前尺度的结果
        scale_pos_features = []
        scale_neg_features = []
        scale_pos_gt_idx = []
        scale_valid_pairs = 0
        
        # 对每个batch单独处理
        for b in range(batch_size):
            batch_fg_mask = fg_mask[b]  # [H, W]
            batch_target_gt_idx = target_gt_idx[b]  # [H, W]
            batch_feat = feat_map[b]  # [C, H, W]
            
            # 找到所有正样本位置
            pos_coords = torch.nonzero(batch_fg_mask, as_tuple=False)  # [N_pos, 2]
            
            if pos_coords.shape[0] == 0:
                continue  # 当前batch没有正样本
            
            # 为每个正样本采样负样本
            for pos_y, pos_x in pos_coords:
                pos_y, pos_x = pos_y.item(), pos_x.item()
                
                # 计算有效的邻域范围
                y_start = max(0, pos_y - radius)
                y_end = min(feat_h, pos_y + radius + 1)
                x_start = max(0, pos_x - radius)
                x_end = min(feat_w, pos_x + radius + 1)
                
                # 如果邻域太小（只有中心点），跳过
                if (y_end - y_start) <= 1 and (x_end - x_start) <= 1:
                    continue
                
                # 提取邻域区域
                neighbor_mask = batch_fg_mask[y_start:y_end, x_start:x_end]  # [h_neighbor, w_neighbor]
                neighbor_features = batch_feat[:, y_start:y_end, x_start:x_end]  # [C, h_neighbor, w_neighbor]
                
                # 找到邻域中的负样本位置（不包括中心点）
                neg_mask = ~neighbor_mask.clone()  # 负样本掩码
                
                # 排除中心点（如果中心点在邻域内）
                center_y_in_neighbor = pos_y - y_start
                center_x_in_neighbor = pos_x - x_start
                if (0 <= center_y_in_neighbor < neg_mask.shape[0] and 
                    0 <= center_x_in_neighbor < neg_mask.shape[1]):
                    neg_mask[center_y_in_neighbor, center_x_in_neighbor] = False
                
                neg_coords = torch.nonzero(neg_mask, as_tuple=False)  # [N_neg, 2]
                
                if neg_coords.shape[0] == 0:
                    continue  # 邻域中没有负样本，跳过此正样本
                
                # 随机采样指定数量的负样本
                actual_sample_num = min(sample_num, neg_coords.shape[0])
                if neg_coords.shape[0] > actual_sample_num:
                    sampled_indices = torch.randperm(neg_coords.shape[0])[:actual_sample_num]
                    neg_coords = neg_coords[sampled_indices]
                
                # 提取正样本特征和GT索引
                pos_feature = batch_feat[:, pos_y, pos_x]  # [C]
                pos_gt_id = batch_target_gt_idx[pos_y, pos_x]  # scalar
                
                # 提取负样本特征
                neg_features_current = []
                for neg_y, neg_x in neg_coords:
                    neg_feature = neighbor_features[:, neg_y, neg_x]  # [C]
                    neg_features_current.append(neg_feature)
                
                if len(neg_features_current) > 0:
                    neg_features_current = torch.stack(neg_features_current)  # [actual_sample_num, C]
                    
                    # 存储结果（只有成功采样到负样本才存储）
                    scale_pos_features.append(pos_feature)
                    scale_neg_features.append(neg_features_current)
                    scale_pos_gt_idx.append(pos_gt_id)
                    scale_valid_pairs += actual_sample_num
        
        # 整合当前尺度的结果
        if len(scale_pos_features) > 0:
            scale_pos_features = torch.stack(scale_pos_features)  # [N_pos, C]
            scale_neg_features = torch.cat(scale_neg_features, dim=0)  # [N_neg_total, C]
            scale_pos_gt_idx = torch.stack(scale_pos_gt_idx)  # [N_pos]
        else:
            # 如果没有有效的正样本，创建空张量
            channels = feature_maps[scale_idx].shape[1]
            scale_pos_features = torch.empty(0, channels, dtype=feature_maps[scale_idx].dtype,
                                           device=feature_maps[scale_idx].device)
            scale_neg_features = torch.empty(0, channels, dtype=feature_maps[scale_idx].dtype,
                                           device=feature_maps[scale_idx].device)
            scale_pos_gt_idx = torch.empty(0, dtype=target_gt_idx_list[scale_idx].dtype,
                                         device=target_gt_idx_list[scale_idx].device)
        
        pos_features_all.append(scale_pos_features)
        neg_features_all.append(scale_neg_features)
        pos_gt_idx_all.append(scale_pos_gt_idx)
        valid_pairs_all.append(scale_valid_pairs)
    
    return pos_features_all, neg_features_all, pos_gt_idx_all, valid_pairs_all


def sample_negative_around_positive_fast(
    target_gt_idx_list, fg_mask_list, feature_maps,
    kernel_size=3, sample_num=1, seed=None
):
    """
    完全向量化的负样本采样：在 batch 维度上展开并并行处理所有正样本。
    使用 unfold + gather + batched multinomial，无显式 Python 循环。
    """
    if seed is not None:
        torch.manual_seed(seed)

    pos_feats_all, neg_feats_all, pos_gt_idxs_all, valid_pairs_all = [], [], [], []
    K = kernel_size
    center_idx = (K * K) // 2

    # 对每个尺度并行处理
    for target_gt_idx, fg_mask, feat_map in zip(target_gt_idx_list, fg_mask_list, feature_maps):
        B, C, H, W = feat_map.shape
        
        # 展开特征 [B, C, K*K, N]
        patches = F.unfold(feat_map, K).view(B, C, K*K, -1)
        # 展开掩码 [B, 1, K*K, N]
        mask_patches = F.unfold(fg_mask.unsqueeze(1).float(), K).view(B, 1, K*K, -1).bool()
        # 中心点正样本掩码 [B, N]
        center_mask = mask_patches[:, 0, center_idx, :]
        # 获取所有 (b, n) 正样本坐标
        pos_bn = torch.nonzero(center_mask, as_tuple=False)  # [P, 2]
        if pos_bn.numel() == 0:
            # 无正样本
            pos_feats_all.append(torch.empty(0, C, device=feat_map.device))
            neg_feats_all.append(torch.empty(0, C, device=feat_map.device))
            pos_gt_idxs_all.append(torch.empty(0, dtype=target_gt_idx.dtype, device=feat_map.device))
            valid_pairs_all.append(0)
            continue

        b_idx, col_idx = pos_bn[:,0], pos_bn[:,1]
        P = col_idx.size(0)
        # 提取 P 个补丁 [P, C, K*K]
        patches_bn = patches.permute(0,3,1,2)[b_idx, col_idx]  # [P, C, K*K]
        # 正样本特征 [P, C]
        pos_feats = patches_bn[:,:,center_idx]
        # 对应 GT id [P]
        Wout = W - K + 1
        ys = (col_idx // Wout) + (K//2)
        xs = (col_idx % Wout) + (K//2)
        gt_ids = target_gt_idx[b_idx, ys, xs]

        # 构建所有正样本背景权重 [P, K*K]
        masks_bn = mask_patches.permute(0,3,1,2)[b_idx, col_idx]  # [P, 1, K*K]
        masks_bn = masks_bn.squeeze(1)
        masks_bn[:, center_idx] = False
        weights = masks_bn.float()

        # 过滤无背景的
        row_sums = weights.sum(dim=1)
        valid = row_sums > 0
        if valid.sum().item() == 0:
            # 只有正样本无可选负样，等同于无输出
            pos_feats_all.append(torch.empty(0, C, device=feat_map.device))
            neg_feats_all.append(torch.empty(0, C, device=feat_map.device))
            pos_gt_idxs_all.append(torch.empty(0, dtype=target_gt_idx.dtype, device=feat_map.device))
            valid_pairs_all.append(0)
            continue

        pos_feats_valid = pos_feats[valid]
        gt_ids_valid = gt_ids[valid]
        weights_valid = weights[valid]
        Pv = weights_valid.size(0)

        # 并行采样 [Pv, sample_num]
        sampled = torch.multinomial(weights_valid, sample_num, replacement=True)
        # 收集负样本特征 [Pv * sample_num, C]
        # patches_bn [P, C, K*K]
        patches_valid = patches_bn[valid]  # [Pv, C, K*K]
        idx = sampled.unsqueeze(1).expand(-1, C, -1)  # [Pv, C, sample_num]
        feats = patches_valid.unsqueeze(2).expand(-1, -1, sample_num, -1)  # [Pv, C, sample_num, K*K]
        # gather in last dim
        neg = torch.gather(feats, 3, idx.unsqueeze(-1)).squeeze(-1)  # [Pv, C, sample_num]
        neg = neg.permute(0,2,1).reshape(-1, C)  # [Pv*sample_num, C]

        # 保存
        pos_feats_all.append(pos_feats_valid)
        neg_feats_all.append(neg)
        pos_gt_idxs_all.append(gt_ids_valid)
        valid_pairs_all.append(int(Pv * sample_num))

    return pos_feats_all, neg_feats_all, pos_gt_idxs_all, valid_pairs_all

def compute_similarity_loss(pos_features_list, neg_features_list, pos_gt_idx_list, 
                           temperature=0.1, loss_type='cosine'):
    """
    计算相似度损失
    
    Args:
        pos_features_list (list): 正样本特征列表
        neg_features_list (list): 负样本特征列表  
        pos_gt_idx_list (list): 正样本GT索引列表
        temperature (float): 温度参数
        loss_type (str): 损失类型 'cosine' 或 'euclidean'
        
    Returns:
        total_loss (Tensor): 总的相似度损失
        scale_losses (list): 每个尺度的损失
    """
    scale_losses = []
    total_samples = 0
    
    for scale_idx, (pos_feats, neg_feats, pos_gt_ids) in enumerate(
        zip(pos_features_list, neg_features_list, pos_gt_idx_list)):
        
        if pos_feats.shape[0] == 0:
            scale_losses.append(torch.tensor(0.0, device=pos_feats.device))
            continue
        
        # 计算相似度
        if loss_type == 'cosine':
            # 余弦相似度
            pos_feats_norm = F.normalize(pos_feats, p=2, dim=1)
            neg_feats_norm = F.normalize(neg_feats, p=2, dim=1)
            
            # 计算每个正样本与其对应负样本的相似度
            similarities = []
            neg_start_idx = 0
            
            for i, pos_feat in enumerate(pos_feats_norm):
                # 找到对应的负样本数量（假设每个正样本对应相同数量的负样本）
                samples_per_pos = neg_feats_norm.shape[0] // pos_feats_norm.shape[0]
                neg_end_idx = neg_start_idx + samples_per_pos
                
                corresponding_neg_feats = neg_feats_norm[neg_start_idx:neg_end_idx]
                sim = torch.mm(pos_feat.unsqueeze(0), corresponding_neg_feats.t()) / temperature
                similarities.append(sim.squeeze(0))
                neg_start_idx = neg_end_idx
            
            if similarities:
                similarities = torch.cat(similarities)
                # 使用对比损失：最小化正负样本的相似度
                scale_loss = torch.mean(torch.exp(similarities))
            else:
                scale_loss = torch.tensor(0.0, device=pos_feats.device)
                
        elif loss_type == 'euclidean':
            # 欧几里得距离
            distances = []
            neg_start_idx = 0
            
            for i, pos_feat in enumerate(pos_feats):
                samples_per_pos = neg_feats.shape[0] // pos_feats.shape[0]
                neg_end_idx = neg_start_idx + samples_per_pos
                
                corresponding_neg_feats = neg_feats[neg_start_idx:neg_end_idx]
                dist = torch.norm(pos_feat.unsqueeze(0) - corresponding_neg_feats, p=2, dim=1)
                distances.append(dist)
                neg_start_idx = neg_end_idx
            
            if distances:
                distances = torch.cat(distances)
                # 距离损失：最大化正负样本的距离
                scale_loss = torch.mean(1.0 / (distances + 1e-6))
            else:
                scale_loss = torch.tensor(0.0, device=pos_feats.device)
        
        scale_losses.append(scale_loss)
        total_samples += pos_feats.shape[0]
    
    # 计算总损失
    if total_samples > 0:
        total_loss = sum(scale_losses) / len([l for l in scale_losses if l > 0])
    else:
        total_loss = torch.tensor(0.0, device=scale_losses[0].device)
    
    return total_loss, scale_losses

def compute_similarity_loss_fast(pos_features_list, neg_features_list, pos_gt_idx_list, 
                           temperature=0.1, loss_type='cosine'):
    scale_losses = []
    
    for pos_feats, neg_feats in zip(pos_features_list, neg_features_list):
        if pos_feats.shape[0] == 0:
            scale_losses.append(torch.tensor(0.0, device=pos_feats.device))
            continue

        N, C = pos_feats.shape
        K = neg_feats.shape[0] // N  # 每个正样本对应K个负样本
        neg_feats = neg_feats.view(N, K, C)  # [N, K, C]

        pos_feats_norm = F.normalize(pos_feats, dim=1)        # [N, C]
        neg_feats_norm = F.normalize(neg_feats, dim=2)        # [N, K, C]

        # 扩展正样本维度并点乘负样本 [N, 1, C] x [N, K, C] → [N, K]
        sim = torch.sum(pos_feats_norm.unsqueeze(1) * neg_feats_norm, dim=2) / temperature

        # 对比学习中通常希望负样本相似度小，可直接对其 exp 后取 mean
        scale_loss = torch.mean(torch.exp(sim))

        
        scale_losses.append(scale_loss)

    # 求平均（忽略0）
    nonzero_losses = [l for l in scale_losses if l > 0]
    if len(nonzero_losses) > 0:
        total_loss = sum(nonzero_losses) / len(nonzero_losses)
    else:
        total_loss = torch.tensor(0.0, device=scale_losses[0].device)

    return total_loss, scale_losses

def sample_negative_samples(
    gt_labels: torch.Tensor,    # [B, N, 1] 整数标签，-1 表示该位置无有效 GT
    gt_bboxes: torch.Tensor,    # [B, N, 4] xyxy 格式
    num_classes: int,           # 类别总数
    n: int = 5,                 # 每个样本最多抽取多少个负样本
    box_noise_scale: float = 0.1,
    seed: int = 42,
):
    """
    为每个 batch 样本随机选取最多 n 个 GT 实例，生成负样本：
      – 类别随机取除原类别之外的一个
      – 对框的中心位置和长宽分别加入比例噪声
      – 同时返回原 GT 的索引，方便追踪

    Args:
        gt_labels: Tensor[B, N, 1]
        gt_bboxes: Tensor[B, N, 4]
        num_classes: int
        n: int
        box_noise_scale: float, 噪声强度
        seed: int | None
    Returns:
        neg_labels:  Tensor[B, n, 1]
        neg_bboxes:  Tensor[B, n, 4]
        neg_src_idx: Tensor[B, n]   原 GT 在 N 维度的索引
    """
    if seed is not None:
        torch.manual_seed(seed)

    B, N, _ = gt_labels.shape
    device = gt_labels.device

    # 判断哪些位置是真实 GT
    is_valid = (gt_labels.squeeze(-1) >= 0)  # [B, N]

    # 准备输出张量
    neg_labels  = gt_labels.new_full((B, n, 1), -1)
    neg_bboxes  = gt_bboxes.new_zeros((B, n, 4))
    neg_src_idx = torch.full((B, n), -1, dtype=torch.long, device=device)

    all_cls = torch.arange(num_classes, device=device)  # [C]

    for b in range(B):
        valid_idx = torch.nonzero(is_valid[b], as_tuple=False).squeeze(1)  # [M]
        M = valid_idx.numel()
        if M == 0:
            continue

        k = min(n, M)
        # 随机选择 k 个有效 GT
        perm = valid_idx[torch.randperm(M, device=device)[:k]]  # [k]

        # 原类别 [k]
        orig_cls = gt_labels[b, perm, 0]
        # 为每个 orig_cls 随机选一个不同的类别
        neg_cls = []
        for c in orig_cls:
            choices = all_cls[all_cls != c]
            neg_cls.append(choices[torch.randint(len(choices), (), device=device)])
        neg_cls = torch.stack(neg_cls).unsqueeze(-1)  # [k,1]

        # 原框 [k,4]
        orig_box = gt_bboxes[b, perm]  # (x1,y1,x2,y2)
        x1, y1, x2, y2 = orig_box.unbind(-1)
        w = (x2 - x1).clamp(min=1e-4)
        h = (y2 - y1).clamp(min=1e-4)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h

        # 在中心位置上加入噪声
        dx = torch.randn_like(w) * box_noise_scale * w
        dy = torch.randn_like(h) * box_noise_scale * h
        new_cx = cx + dx
        new_cy = cy + dy

        # 在长宽上加入噪声
        dw = torch.randn_like(w) * box_noise_scale * w
        dh = torch.randn_like(h) * box_noise_scale * h
        new_w = (w + dw).clamp(min=1e-4)
        new_h = (h + dh).clamp(min=1e-4)

        # 还原为 xyxy
        new_x1 = new_cx - 0.5 * new_w
        new_y1 = new_cy - 0.5 * new_h
        new_x2 = new_cx + 0.5 * new_w
        new_y2 = new_cy + 0.5 * new_h
        neg_box = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)  # [k,4]

        # 填充输出
        neg_labels[b, :k, 0]  = neg_cls.squeeze(-1)
        neg_bboxes[b, :k]     = neg_box
        neg_src_idx[b, :k]    = perm

    return neg_labels, neg_bboxes, neg_src_idx

def sample_noise_clean_feature_pairs(
    nos_fg_mask: torch.Tensor,
    nos_gt_idx: torch.Tensor,
    clean_fg_mask: torch.Tensor,
    clean_gt_idx: torch.Tensor,
    feature_maps: List[torch.Tensor],
    strides: List[int],
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    可选 seed 控制随机采样的一致性。
    """
    if seed is not None:
        torch.manual_seed(seed)

    B, P = nos_fg_mask.shape
    lvl_sizes = [fm.shape[2] * fm.shape[3] for fm in feature_maps]
    lvl_offsets = torch.tensor([0] + torch.cumsum(torch.tensor(lvl_sizes), dim=0).tolist(), device=nos_fg_mask.device)
    
    noisy_feats = []
    clean_feats = []
    
    for b in range(B):
        idxs_n = torch.nonzero(nos_fg_mask[b], as_tuple=False).squeeze(1)
        gts_n  = nos_gt_idx[b, idxs_n]
        for gt in torch.unique(gts_n):
            if gt < 0:
                continue
            pos_n = idxs_n[gts_n == gt]
            if pos_n.numel() == 0:
                continue
            n_idx = pos_n[torch.randint(pos_n.numel(), (1,), device=pos_n.device)].item()
            idxs_c = torch.nonzero(clean_fg_mask[b] & (clean_gt_idx[b] == gt), as_tuple=False).squeeze(1)
            if idxs_c.numel() == 0:
                continue
            c_idx = idxs_c[torch.randint(idxs_c.numel(), (1,), device=idxs_c.device)].item()
            
            def idx2lxy(idx: int):
                lvl = torch.searchsorted(lvl_offsets, torch.tensor(idx+1, device=lvl_offsets.device)) - 1
                lvl = int(lvl.clamp(0, len(feature_maps)-1))
                local = idx - lvl_offsets[lvl]
                H, W = feature_maps[lvl].shape[2:]
                y = int(local // W)
                x = int(local %  W)
                return lvl, y, x
            
            lvl_n, yn, xn = idx2lxy(n_idx)
            lvl_c, yc, xc = idx2lxy(c_idx)
            fn = feature_maps[lvl_n][b, :, yn, xn]
            fc = feature_maps[lvl_c][b, :, yc, xc]
            
            noisy_feats.append(fn)
            clean_feats.append(fc)
    
    if len(noisy_feats) == 0:
        return torch.empty(0, feature_maps[0].shape[1]), torch.empty(0, feature_maps[0].shape[1])
    
    return torch.stack(noisy_feats), torch.stack(clean_feats)

def sample_noise_clean_feature_pairs_fast(
    nos_fg_mask: torch.Tensor,     # [B, P]
    nos_gt_idx: torch.Tensor,      # [B, P]
    clean_fg_mask: torch.Tensor,   # [B, P]
    clean_gt_idx: torch.Tensor,    # [B, P]
    feature_maps: List[torch.Tensor],  # List[B, C, H, W]
    seed: int = 42,
    max_samples_per_gt: int = 1,
    strides: List[int] = [8, 16, 32, 64, 128]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    极致加速版本：支持批量构造 index 并一次性采样，无任何 for-loop。
    """

    if seed is not None:
        torch.manual_seed(seed)

    B, P = nos_fg_mask.shape
    C = feature_maps[0].shape[1]

    # Step 1: 拼接全部特征图，转为 [B, C, P]
    feats_all = torch.cat([
        fm.reshape(B, C, -1) for fm in feature_maps
    ], dim=2)  # [B, C, P]

    device = nos_fg_mask.device

    # Step 2: 找到所有正样本的索引
    nos_mask = nos_fg_mask & (nos_gt_idx >= 0)
    clean_mask = clean_fg_mask & (clean_gt_idx >= 0)

    # 为每个 batch 找出 (gt, index) 的映射
    noisy_pairs = []
    clean_pairs = []

    for b in range(B):
        n_idx = torch.nonzero(nos_mask[b], as_tuple=False).squeeze(1)
        c_idx = torch.nonzero(clean_mask[b], as_tuple=False).squeeze(1)

        if n_idx.numel() == 0 or c_idx.numel() == 0:
            continue

        n_gts = nos_gt_idx[b, n_idx]
        c_gts = clean_gt_idx[b, c_idx]

        unique_gts = torch.unique(n_gts)
        for gt in unique_gts:
            n_pos = n_idx[n_gts == gt]
            c_pos = c_idx[c_gts == gt]

            if n_pos.numel() == 0 or c_pos.numel() == 0:
                continue

            # 随机采样一个或多个
            n_sel = n_pos[torch.randint(n_pos.numel(), (max_samples_per_gt,), device=device)]
            c_sel = c_pos[torch.randint(c_pos.numel(), (max_samples_per_gt,), device=device)]

            for ni, ci in zip(n_sel, c_sel):
                noisy_pairs.append((b, ni.item()))
                clean_pairs.append((b, ci.item()))

    if len(noisy_pairs) == 0:
        return torch.empty(0, C, device=device), torch.empty(0, C, device=device)

    # Step 3: 构建 index，使用 gather 抽取特征
    noisy_pairs = torch.tensor(noisy_pairs, dtype=torch.long, device=device)  # [N, 2]
    clean_pairs = torch.tensor(clean_pairs, dtype=torch.long, device=device)  # [N, 2]

    b_n, i_n = noisy_pairs[:, 0], noisy_pairs[:, 1]
    b_c, i_c = clean_pairs[:, 0], clean_pairs[:, 1]

    # gather feature: feats_all [B, C, P]
    # expand to [N, C]
    noisy_feats = feats_all[b_n, :, i_n]  # [N, C]
    clean_feats = feats_all[b_c, :, i_c]  # [N, C]

    return noisy_feats, clean_feats



def prepare_features_from_head_features(head_features):
    """
    将多尺度的head_features转换为anchor-level的特征
    
    Args:
        head_features: list of tensors
            - head_features[0]: (bs, 64, 80, 80)
            - head_features[1]: (bs, 64, 40, 40)
            - head_features[2]: (bs, 64, 20, 20) # 如果有第三个尺度
    
    Returns:
        features: (bs, num_total_anchors, feature_dim)
    """
    bs = head_features[0].shape[0]
    feat_dim = head_features[0].shape[1]  # 64
    
    features_list = []
    
    for feat in head_features:
        # feat.shape = (bs, feat_dim, H, W)
        # 转换为 (bs, H*W, feat_dim)
        bs, c, h, w = feat.shape
        feat_reshaped = feat.view(bs, c, h * w)  # (bs, feat_dim, H*W)
        feat_reshaped = feat_reshaped.permute(0, 2, 1)  # (bs, H*W, feat_dim)
        features_list.append(feat_reshaped)
    
    # 拼接所有尺度的特征
    features = torch.cat(features_list, dim=1)  # (bs, num_total_anchors, feat_dim)
    
    return features

def get_class_prototype_vectors(features, gt_labels, target_gt_idx, fg_mask, num_classes=None):
    device = features.device
    bs, num_anchors, feat_dim = features.shape

    fg_mask = fg_mask.bool()
    if not fg_mask.any():
        if num_classes is None:
            return torch.zeros((0, feat_dim), device=device), torch.zeros((0,), dtype=torch.long, device=device), torch.zeros((0,), dtype=torch.bool, device=device)
        return (torch.zeros((num_classes, feat_dim), device=device, dtype=features.dtype),
                torch.zeros((num_classes,), device=device, dtype=torch.long),
                torch.zeros((num_classes,), device=device, dtype=torch.bool))

    batch_idx = torch.arange(bs, device=device).unsqueeze(1).expand(-1, num_anchors)
    batch_idx_fg = batch_idx[fg_mask]
    target_gt_idx_fg = target_gt_idx[fg_mask].long()

    fg_labels = gt_labels[batch_idx_fg, target_gt_idx_fg].squeeze(-1).long()  # (N_fg,)
    fg_features = features[fg_mask]  # (N_fg, feat_dim)

    if num_classes is None:
        num_classes = int(torch.max(fg_labels).item()) + 1

    # sums and counts
    class_sums = torch.zeros((num_classes, feat_dim), device=device, dtype=fg_features.dtype)
    # avoid index_add_ error if fg_labels contains label >= num_classes
    if fg_labels.max().item() >= num_classes:
        raise ValueError(f"fg_labels contains label >= num_classes ({fg_labels.max().item()} >= {num_classes})")

    class_sums.index_add_(0, fg_labels, fg_features)
    class_counts = torch.bincount(fg_labels, minlength=num_classes).to(device=device)
    has_sample = class_counts > 0

    class_prototypes = torch.zeros_like(class_sums)
    if has_sample.any():
        class_prototypes[has_sample] = class_sums[has_sample] / class_counts[has_sample].unsqueeze(1).to(fg_features.dtype)

    return class_prototypes, class_counts, has_sample


def get_topk_feature_per_class(features, target_scores, gt_labels, target_gt_idx, fg_mask, num_classes=None, return_locations=False):
    device = features.device
    bs, num_anchors, feat_dim = features.shape
    if num_classes is None:
        num_classes = target_scores.shape[-1]

    fg_mask = fg_mask.bool()
    target_gt_idx = target_gt_idx.long()
    gt_labels = gt_labels.long()

    if not fg_mask.any():
        best_features = torch.zeros((num_classes, feat_dim), device=device, dtype=features.dtype)
        best_scores = torch.full((num_classes,), float("-inf"), device=device, dtype=target_scores.dtype)
        has_sample = torch.zeros((num_classes,), dtype=torch.bool, device=device)
        if return_locations:
            best_locs = torch.full((num_classes, 2), -1, device=device, dtype=torch.long)
            return best_features, best_scores, has_sample, best_locs
        return best_features, best_scores, has_sample

    fg_features = features[fg_mask]                       # (N_fg, feat_dim)
    fg_scores_all = target_scores[fg_mask]                # (N_fg, num_classes)
    batch_idx = torch.arange(bs, device=device).unsqueeze(1).expand(-1, num_anchors)
    batch_idx_fg = batch_idx[fg_mask]                     # (N_fg,)
    target_gt_idx_fg = target_gt_idx[fg_mask].long()     # (N_fg,)
    fg_labels = gt_labels[batch_idx_fg, target_gt_idx_fg].squeeze(-1).long()  # (N_fg,)

    # Ensure no label exceeds num_classes-1
    if fg_labels.max().item() >= num_classes:
        raise ValueError(f"fg_labels contains label >= num_classes ({fg_labels.max().item()} >= {num_classes})")

    per_sample_scores = fg_scores_all.gather(1, fg_labels.unsqueeze(1)).squeeze(1)  # (N_fg,)

    anchors_idx = torch.arange(num_anchors, device=device).unsqueeze(0).expand(bs, -1)
    anchor_idx_fg = anchors_idx[fg_mask].long()  # (N_fg,)

    best_features = torch.zeros((num_classes, feat_dim), device=device, dtype=features.dtype)
    best_scores = torch.full((num_classes,), float("-inf"), device=device, dtype=per_sample_scores.dtype)
    has_sample = torch.zeros((num_classes,), dtype=torch.bool, device=device)
    if return_locations:
        best_locs = torch.full((num_classes, 2), -1, device=device, dtype=torch.long)

    unique_classes = torch.unique(fg_labels)
    for cls_t in unique_classes:
        cls = int(cls_t.item())
        cls_mask = (fg_labels == cls)
        cls_scores = per_sample_scores[cls_mask]
        max_idx_in_cls = torch.argmax(cls_scores)
        global_indices = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
        chosen_global_idx = global_indices[max_idx_in_cls]
        best_scores[cls] = per_sample_scores[chosen_global_idx]
        best_features[cls] = fg_features[chosen_global_idx]
        has_sample[cls] = True
        if return_locations:
            best_locs[cls, 0] = int(batch_idx_fg[chosen_global_idx].item())
            best_locs[cls, 1] = int(anchor_idx_fg[chosen_global_idx].item())

    if return_locations:
        return best_features, best_scores, has_sample, best_locs
    return best_features, best_scores, has_sample