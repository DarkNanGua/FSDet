import torch
from torch import nn

class EMAttC(nn.Module):
    def __init__(self, channels,out, factor=8):
      super(EMAttC, self).__init__()
      self.groups = factor
      assert channels // self.groups > 0
      self.softmax = nn.Softmax(-1)
      self.agp = nn.AdaptiveAvgPool2d((1, 1))
      self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
      self.pool_w = nn.AdaptiveAvgPool2d((1, None))
      self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
      self.gn_after = nn.Conv2d(channels // self.groups, out // self.groups, kernel_size=1, stride=1,padding=0)
      self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
      self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
      self.conv3x3_after = nn.Conv2d(channels // self.groups, out // self.groups, kernel_size=1, stride=1,padding=0)

    def forward(self, input):
      x = input[0]
      fuse_x = input[1]
      b_,c_,h_,w_ = fuse_x.size()
      b, c, h, w = x.size()
      group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
      fuse_x = fuse_x.reshape(b * self.groups, -1, h_, w_) 
      x_h = self.pool_h(group_x)
      x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
      hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
      x_h, x_w = torch.split(hw, [h, w], dim=2)
      x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
      x1 = self.gn_after(x1)
      x2 = self.conv3x3(group_x)
      x2 = self.conv3x3_after(x2)
      x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
      x12 = x2.reshape(b * self.groups, c_ // self.groups, -1)  # b*g, c//g, hw
      x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
      x22 = x1.reshape(b * self.groups, c_ // self.groups, -1)  # b*g, c//g, hw
      weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
      weights = nn.functional.interpolate(weights,size=(h_,w_),mode='nearest')
      return (fuse_x * weights.sigmoid()).reshape(b_, c_, h_, w_)