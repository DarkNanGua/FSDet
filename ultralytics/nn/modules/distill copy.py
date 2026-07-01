from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# class GaussianMaskGenerator(nn.Module):
#     """生成目标区域的高斯掩码"""
#     def __init__(self, img_size=640, sigma_scale=0.5):
#         super().__init__()
#         self.sigma_scale = sigma_scale
#         self.img_size = img_size
        
#     def forward(self, targets):
#         """
#         targets: [batch, (img_id, x_center, y_center, w, h, cls)]
#         返回: 高斯掩码矩阵 [batch, 1, H, W]
#         """
#         batch_size = targets[:, 0].max().int() + 1
#         masks = torch.zeros(batch_size, 1, self.img_size, self.img_size)
        
#         for target in targets:
#             img_id = target[0].int()
#             x_c, y_c = target[1] * self.img_size, target[2] * self.img_size
#             w, h = target[3] * self.img_size, target[4] * self.img_size
            
#             # 生成高斯分布
#             x = torch.arange(self.img_size).view(1, -1)
#             y = torch.arange(self.img_size).view(-1, 1)
#             gauss = torch.exp(-(
#                 ((x - x_c)/ (w/2 * self.sigma_scale))**2 + 
#                 ((y - y_c)/ (h/2 * self.sigma_scale))**2
#             ))
            
#             masks[img_id] = torch.maximum(masks[img_id], gauss)
            
#         return masks
class CosineAnnealingScheduler:
    def __init__(self, 
                 max_steps, 
                 min_weight=0.1, 
                 max_weight=0.5,
                 restart_cycle=0):
        """
        restart_cycle: 重启周期（0表示不重启）
        """
        self.max_steps = max_steps
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.restart_cycle = restart_cycle
        self.current_step = 0
    def get_weight(self):
        # 计算当前有效步数（考虑周期重启）
        if self.restart_cycle > 0:
            effective_step = self.current_step % self.restart_cycle
            cycle_ratio = effective_step / self.restart_cycle
        else:
            cycle_ratio = self.current_step / self.max_steps
        # 余弦计算
        cosine = np.cos(np.pi * cycle_ratio)
        weight = self.min_weight + 0.5*(self.max_weight - self.min_weight)*(1 + cosine)
        self.current_step += 1
        return weight

class MultiStageScheduler:
    def __init__(self, stages):
        """
        stages: [(start_step, end_step, min_w, max_w), ...]
        """
        self.stages = sorted(stages, key=lambda x:x[0])
        self.current_step = 0
    def get_weight(self):
        for stage in self.stages:
            if stage[0] <= self.current_step < stage[1]:
                ratio = (self.current_step - stage[0]) / (stage[1]-stage[0])
                cosine = np.cos(np.pi * ratio)
                return stage[2] + 0.5*(stage[3]-stage[2])*(1+cosine)
        return 0.0  # 默认不启用
# # 示例配置：前期强蒸馏，中期弱化，后期增强
# scheduler = MultiStageScheduler([
#     (0, 50*809, 0.5, 0.8),    # 阶段1：强蒸馏
#     (50*809, 150*809, 0.1, 0.3), # 阶段2：弱蒸馏 
#     (150*809, 200*809, 0.4, 0.6) # 阶段3：恢复蒸馏
# ])


class MultiScaleDistillLoss(nn.Module):
    def __init__(self, is_use = False,warmup = 10,end_epoch = 120,min_weight=0.4,max_weight=1, weights=[1,0.7, 0.3, 0.1]):
        super().__init__()
        self.is_use = is_use
        self.mse = nn.MSELoss()
        self.weights = weights  # 各尺度损失权重
        self.warmup = warmup * 809 # 预热步数
        self.cur_epoch = 0
        self.end_epoch = end_epoch * 809
        total_steps = self.end_epoch - self.warmup
        self.scheduler = CosineAnnealingScheduler(
            max_steps=total_steps - self.warmup,
            min_weight=min_weight,
            max_weight=max_weight,
            restart_cycle=10*809  # 每30 epoch重启一次
        )
        
    def forward(self, features):
        self.cur_epoch += 1
        if self.is_use is False or self.training is False or self.cur_epoch <= self.warmup or self.cur_epoch >= self.end_epoch:
            return .0
        current_weight = self.scheduler.get_weight()
        len_ = len(features)
        fpn_features = features[:len_//2]  # backbone特征
        pan_features = features[len_//2:]  # PAN特征
        # pan_features = features[len_//2:].detach()  # PAN特征
        scales = [fpn_features[i].shape[-1] for i in range(len_//2)]
        total_loss = 0
        for i, (fpn, pan) in enumerate(zip(fpn_features, pan_features)):
            # 计算当前尺度损失
            scale_loss = self.mse(fpn, pan) * self.weights[i]
            total_loss += scale_loss
            
        return current_weight*total_loss / len(scales)

