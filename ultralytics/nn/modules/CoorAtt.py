import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from ultralytics.nn.modules.DySample import DySample


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h 
        # out = a_h.expand_as(x) * a_w.expand_as(x) * identity

        return out
    
    
# class CoordAttMap(nn.Module):
#     def __init__(self, inp,oup, reduction=32):
#         super(CoordAttMap, self).__init__()

#         self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
#         self.pool_w = nn.AdaptiveMaxPool2d((1, None))

#         mip = max(8, inp // reduction)

#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = nn.Hardswish()

#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    
#     def forward(self, x):
#         identity = x

#         n, c, h, w = x[0].size()
#         n_, c_, h_, w_ = x[1].size()
        
#         x_h = self.pool_h(x[0])
#         x_w = self.pool_w(x[0]).permute(0, 1, 3, 2)
        
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)

#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)

#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#         map = a_w * a_h
#         map = self.normalize(map)
#         map = F.interpolate(map,size=(h_,w_),mode='bilinear')

#         out = identity[1] * map
#         # out = a_h.expand_as(x) * a_w.expand_as(x) * identity

#         return out
    
#     def normalize(self,x):
#         return (np.e**x - 1) / (np.e - 1)* 0.5 + 0.5
    
class CoordAttMap(nn.Module):
    def __init__(self, inp,oup, reduction=32):
        super(CoordAttMap, self).__init__()

        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    
    def forward(self, x):
        identity = x

        n, c, h, w = x[0].size()
        n_, c_, h_, w_ = x[1].size()
        x[0] = F.interpolate(x[0],size=(h_,w_),mode='nearest')
        
        x_h = self.pool_h(x[0])
        x_w = self.pool_w(x[0]).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h_, w_], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        map = a_w * a_h
        map = self.normalize(map)

        out = identity[1] * map
        # out = a_h.expand_as(x) * a_w.expand_as(x) * identity

        return out
    
    def normalize(self,x):
        return (np.e**x - 1) / (np.e - 1)* 0.5 + 0.5

class CoordAttMapC(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAttMapC, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
    

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # out = a_h.expand_as(x) * a_w.expand_as(x) * identity

        return [y,h,w]
    
class FuseAtt(nn.Module):
    def __init__(self, inp,oup):
        super(FuseAtt, self).__init__()
        self.conv_h = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)
                
    def forward(self,x):
        n, c, h, w = x[0][0].size()
        n_, c_, h_, w_ = x[1].size()
        
        map_h, map_w = torch.split(x[0][0], [x[0][1], x[0][2]], dim=2)
        map_w = map_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(map_h).sigmoid()
        a_w = self.conv_w(map_w).sigmoid()
        
        map = a_w * a_h
         
        map = F.interpolate(map,size=(h_,w_),mode='nearest')
        
        out = x[1] * map
        return out
        
class CoordAttMapD(nn.Module):
    """_summary_

    Args:
        x -> x[0]:来自上层特征图，x[1]:来自下层特征图
    """
    def __init__(self, inp,oup, reduction=32,option = "avg",up_style = "nearest",ur = 2):
        super(CoordAttMapD, self).__init__()
        assert option in ["mix","avg","max"],"option input invaild"
        self.option = option
        if option == "avg":
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        elif option == "max":
            self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
            self.pool_w = nn.AdaptiveMaxPool2d((1, None))
        elif option == "mix":
            self.maxpool_h = nn.AdaptiveMaxPool2d((None, 1))
            self.maxpool_w = nn.AdaptiveMaxPool2d((1, None))
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h1 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w1 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
        
        if option == "mix":
            self.conv2 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
            self.bn2 = nn.BatchNorm2d(mip)
            self.conv_h2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
            self.conv_w2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
            self.scale = nn.Parameter(torch.ones([]))
        
        
        assert up_style in [ 'nearest','linear','bilinear','bicubic','trilinear',"dysample"]\
            ,"up_style input invaild"
            
        if up_style == "dysample":
            self.upsample = DySample(oup,ur,style='pl',groups=8,dyscope=True)
        else:
            self.upsample = nn.Upsample(scale_factor=ur,mode=up_style)
    

    def forward(self, x):

        # x[0] = self.upsample(x[0])
        n, c, h, w = x[0].size()
        n_, c_, h_, w_ = x[1].size()
        x_h = self.pool_h(x[0])
        x_w = self.pool_w(x[0]).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h1(x_h).sigmoid()
        a_w = self.conv_w1(x_w).sigmoid()
        map = a_w * a_h
        
        if self.option == "mix":
            x_h2 = self.maxpool_h(x[0])
            x_w2 = self.maxpool_w(x[0]).permute(0, 1, 3, 2)
            
            y2 = torch.cat([x_h2, x_w2], dim=2)
            y2 = self.conv2(y2)
            y2 = self.bn2(y2)
            y2 = self.act(y2)

            x_h2, x_w2 = torch.split(y, [h, w], dim=2)
            x_w2 = x_w2.permute(0, 1, 3, 2)

            a_h2 = self.conv_h2(x_h2).sigmoid()
            a_w2 = self.conv_w2(x_w2).sigmoid()
            r = self.scale.sigmoid()
            # print(f"r={r}")
            map = r*map + (1-r)*a_w2 * a_h2
            
        # map = self.normalize(map)
        map = self.upsample(map)
        # map = nn.PixelShuffle()
        
        out = x[1]  * map + x[1]
        # out = a_h.expand_as(x) * a_w.expand_as(x) * identity

        return out
    
    def normalize(self,x):
        return (np.e**x - 1) / (np.e - 1)* 0.5 + 0.5
    