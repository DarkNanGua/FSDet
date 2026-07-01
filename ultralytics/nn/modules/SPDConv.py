import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

class SinCosPositionEmbedding(nn.Module):
    """只编码空间位置的 Sin/Cos 位置编码，不切分通道"""
    
    def __init__(self, max_cache_size=5):
        super().__init__()
        self.max_cache_size = max_cache_size
        self.pos_cache = {}

    def forward(self, x):
        B, C, H, W = x.shape
        cache_key = (H, W, x.device)

        if cache_key not in self.pos_cache:
            if len(self.pos_cache) >= self.max_cache_size:
                self.pos_cache.pop(next(iter(self.pos_cache)))
            self.pos_cache[cache_key] = self._generate_pos_embed(H, W, x.device)

        pos = self.pos_cache[cache_key]  # (1, 2, H, W)
        pos = pos.expand(B, -1, H, W)    # 扩展到 batch
        return x + pos.repeat(1, C // 2, 1, 1)  # 广播到输入通道数

    def _generate_pos_embed(self, H, W, device):
        # y: [-1, 1], shape (H, 1)
        y_pos = torch.linspace(-1, 1, H, device=device).unsqueeze(1).expand(H, W)
        # x: [-1, 1], shape (1, W)
        x_pos = torch.linspace(-1, 1, W, device=device).unsqueeze(0).expand(H, W)

        # 组合 (2, H, W)  -> y 在通道 0，x 在通道 1
        pos_embed = torch.stack((y_pos, x_pos), dim=0).unsqueeze(0)  # (1, 2, H, W)
        return pos_embed
# class CED(nn.Module):
#     def __init__(self, c1, c2, e=0.5):
#         super().__init__()
#         self.cv2 = Conv(c1 * 4, c2, k=1,
#                               s=1, p=0)

#         self.gconv = Conv(c1 * 4, c1 * 4,
#                                  k=3, s=1, p=1,
#                                  g=c1)
#         self.offset_embed = nn.Parameter(torch.randn(1, c1 * 4, 1, 1))

#     def forward(self, x):
#         x = torch.cat([
#             x[..., ::2, ::2],
#             x[..., 1::2, ::2],
#             x[..., ::2, 1::2],
#             x[..., 1::2, 1::2],
#         ], dim=1)
#         # x = x + self.offset_embed
#         # CHANNEL SHUFFLE
#         b, c, h, w = x.shape
#         x = x.view(b, 4, c // 4, h, w).permute(0, 2, 1, 3, 4).reshape(b, c, h, w)
#         x = self.gconv(x)
#         x = self.cv2(x)
#         return x
    
class CED(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.proj1 = Conv(c1 , c2 , k=1,
                              s=1, p=0)
        self.cv1 = Conv(c2 //2, c2//2, k=3,
                              s=1, p=1,g=c2//2)
        self.cv2 = Conv(c2//2, c2//2, k=5,
                              s=1, p=2,g=c2//2)
        self.cv3 = Conv(c2//2, c2//2, k=7,
                              s=1, p=3,g=c2//2)
        # self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.proj2 = Conv(c2//2*4, c2, k=3, s=1, p=1)

    def forward(self, x):
        x = self.proj1(x)
        a,b = x.split([x.shape[1]//2, x.shape[1]//2], dim=1)
        x3 = self.cv1(b)
        x5 = self.cv2(x3+b) 
        x7 = self.cv3(x5+b) 
        x = torch.cat([a,x3, x5,x7], dim=1)
        x = self.proj2(x)
        
        return x



class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
