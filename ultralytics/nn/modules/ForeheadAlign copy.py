from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from ultralytics.nn.modules.DySample import DySample
from ultralytics.nn.modules.conv import AsymmetricConv

class CoordAttMap(nn.Module):
    """_summary_

    Args:
        x -> x[0]:来自上层特征图，x[1]:来自下层特征图
    """
    def __init__(self, inp,oup, reduction=32,option = "avg",up_style = "nearest",ur = 2,L=16):
        super(CoordAttMap, self).__init__()
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
        
        out = x[1]  * map 
        # out = a_h.expand_as(x) * a_w.expand_as(x) * identity

        return out
    
    def normalize(self,x):
        return (np.e**x - 1) / (np.e - 1)* 0.5 + 0.5
      
      

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() 
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class AsymmetricSKA(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5],nxm = False,reduction=16,group=1,L=32):
        super().__init__()
        if group == -1 :
            group = channel
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',AsymmetricConv(channel,channel,kernel_size=k,groups=1,bias=False,nxm = nxm)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('h_silu',nn.Hardswish()),
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)        



    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            # conv_outs.append(self.drop_path(conv(x)))
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c,1
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V
      
class ForeheadAlign(nn.Module):
    def __init__(self, inp, oup, rmap=32,option = "avg",up_style = "nearest",ur = 2,kernels=[1,3,5],nxm = False,rska=16):
        super(ForeheadAlign, self).__init__()
        self.coordattmap = CoordAttMap(inp,oup, rmap,option,up_style,ur)
        self.ska = AsymmetricSKA(oup,kernels,nxm,rska)
    def forward(self, x):
        out = self.coordattmap(x) + x[1]
        out = self.ska(out) 
        return out