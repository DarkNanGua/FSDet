import torch
import torch.nn as nn

class SPD(nn.Module):   # SPD 层
    """
    这个模块实现了空间到深度的操作，它重新排列空间数据块到深度维度，
    通过块大小增加通道数并减少空间维度。在卷积神经网络中常用此方法保持
    下采样图像的高分辨率信息。
    """
    def __init__(self, block_size=2):
        """
        初始化 SPD 模块。
        
        参数:
            block_size (int): 每个块的大小。它定义了空间维度的下采样因子。
                              输出通道的数量将增加 block_size**2 倍。
        """
        super(SPD, self).__init__()
        self.block_size = block_size  # 块大小

    def forward(self, x):
        """
        在输入张量上应用空间到深度操作。
        
        参数:
            x (torch.Tensor): 形状为 (N, C, H, W) 的输入张量。
        
        返回:
            torch.Tensor: 重新排列块后的输出张量。如果块大小为 2，
                          输出张量的形状将为 (N, C*4, H/2, W/2)。
        """
        N, C, H, W = x.size()  # 输入张量的维度
        block_size = self.block_size  # 块大小

        # 确保高度和宽度可以被 block_size 整除
        assert H % block_size == 0 and W % block_size == 0, \
            f"空间维度必须能被 block_size 整除。得到的 H: {H}, W: {W}"

        # 将空间块重新排列到深度
        x_reshaped = x.view(N, C, H // block_size, block_size, W // block_size, block_size)
        x_permuted = x_reshaped.permute(0, 3, 5, 1, 2, 4).contiguous()
        out = x_permuted.view(N, C * block_size**2, H // block_size, W // block_size)