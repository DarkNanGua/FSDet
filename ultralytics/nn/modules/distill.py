from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import global_variables as g


class MultiScaleDistillLoss(nn.Module):
    def __init__(self, channel=[], is_use=False, warmup=0, end_epoch=200, 
                 min_weight=0.2, max_weight=1, weights=[1,0.7,0.5, 0.1],
                 threshold=0.6, epoch_interval=1):
        super().__init__()
        
        self.is_use = is_use
        self.weights = weights  # 各尺度损失权重 
        self.warmup = warmup  # 预热步数 
        self.cur_epoch = 0
        self.cur_step = 0
        self.end_epoch = end_epoch 
        self.threshold = threshold  # p5_scale的阈值 
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
        
        self.cv1 = nn.Conv2d(self.p5_channel, 1, kernel_size=1, stride=1, padding=0)
        

    
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
        


        # current_weight = self.scheduler.get_weight()
        current_weight = 0.2
        len_ = len(features)
        bb_features = features[:len_//2]  # backbone特征
        pan_features = features[len_//2:]  # PAN特征
        scales = [bb_features[i].shape[-1] for i in range(len_//2)]
        total_loss = 0



        for i, (bb, pan) in enumerate(zip(bb_features, pan_features)):
            B,C,H,W = bb.shape
            # 如果 H*W 大于 80*80，则随机采样 6400 个点进行计算
            if H * W > 80 * 80:
                # 采样点数 k，最多采样 6400（80*80），但不能超过总点数
                N = H * W
                k = min(80 * 80, N)

                # bb 不进行梯度传播
                bb = bb.detach()
                # 展平为 (B, N, C)
                bb = bb.view(B, C, N).permute(0, 2, 1)
                pan = pan.view(B, C, N).permute(0, 2, 1)

                device = bb.device
                # 为效率起见，对所有 batch 使用相同的随机索引采样；也可按需改为 per-sample 索引
                idx = torch.randperm(N, device=device)[:k]

                # 选取采样点 -> (B, k, C)
                bb_sample = bb[:, idx, :]
                pan_sample = pan[:, idx, :]

                # L2 归一化
                bb_sample = F.normalize(bb_sample, p=2, dim=2)
                pan_sample = F.normalize(pan_sample, p=2, dim=2)
                bb_cos_matrix = torch.matmul(bb_sample, bb_sample.transpose(1, 2))
                pan_cos_matrix = torch.matmul(pan_sample, pan_sample.transpose(1, 2))

                # loss
                scale_loss = nn.functional.mse_loss(bb_cos_matrix, pan_cos_matrix)
            else:
                # 如果H*W小于等于80*80
                # bb不进行梯度传播 
                bb = bb.detach()
                bb = bb.view(B, C, H*W).permute(0, 2, 1)
                pan = pan.view(B, C, H*W).permute(0, 2, 1)
                # L2 归一化
                bb = F.normalize(bb, p=2, dim=2)
                pan = F.normalize(pan, p=2, dim=2)            
                bb_cos_matrix = torch.matmul(bb, bb.transpose(1, 2))
                pan_cos_matrix = torch.matmul(pan, pan.transpose(1, 2))
                # loss
                scale_loss = nn.functional.mse_loss(bb_cos_matrix, pan_cos_matrix)
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
    
    