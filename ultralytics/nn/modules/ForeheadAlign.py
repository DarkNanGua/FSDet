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
      
      

class AttnMap(nn.Module):
    def __init__(self, inp,oup,ur=2):
        super(AttnMap, self).__init__()
        self.conv1 = nn.Conv2d(inp, 1, kernel_size=5, stride=1, padding=2,bias=False)
        self.upsample = nn.Upsample(scale_factor=ur,mode="nearest")
        
    def forward(self, x):
        x1 = self.conv1(x[0]).softmax(dim=1)
        map = self.upsample(x1)
        y = x[1] * map
        return y
        

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
      
class SpatialGroupEnhance(nn.Module):

    def __init__(self, inp,oup, reduction=32,option = "avg",up_style = "nearest",ur = 2,L=16,groups=8):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()
        self.init_weights()
        self.upsample = nn.Upsample(scale_factor=ur,mode=up_style)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, g):
        x = g[0]
        b, c, h,w=x.shape
        x=x.view(b*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(b*self.groups,-1) #bs*g,h*w

        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True) + 1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,h,w) #bs,g,h*w
        
        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(b*self.groups,1,h,w) #bs*g,1,h*w
        b,c,h,w = g[1].shape
        x=g[1].view(b*self.groups,-1,h,w)*self.upsample(self.sig(t))
        return x.view(b,c,h,w)

import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
  
def save_all_feature_maps(tensor, save_dir="visualize/all_channels", prefix="feat", cmap='viridis'):
    """
    保存每个通道为单张无边框、无标题、无轴线的图像。
    tensor: shape [1, C, H, W]
    """
    os.makedirs(save_dir, exist_ok=True)
    tensor = tensor.detach().cpu().squeeze(0)  # [C, H, W]

    for i in range(tensor.shape[0]):
        feature = tensor[i]
        plt.figure(figsize=(feature.shape[1] / 100, feature.shape[0] / 100), dpi=100)
        plt.imshow(feature, cmap=cmap)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去掉边距
        plt.savefig(
            os.path.join(save_dir, f"{prefix}_channel_{i}.png"),
            bbox_inches='tight',  # 去掉空白边框
            pad_inches=0
        )
        plt.close()
    
class ForeheadAlign(nn.Module):
    def __init__(self, inp, oup, rmap=32,option = "avg",up_style = "nearest",ur = 2,kernels=[1,3,5],nxm = False,rska=16):
        super(ForeheadAlign, self).__init__()
        # self.coordattmap = CoordAttMap(inp,oup, rmap,option,up_style,ur)
        # self.coordattmap = SpatialGroupEnhance(inp,oup,ur=ur)
        # self.coordattmap = AttnMap(inp,oup,ur=ur)
        self.coordattmap = nn.Sequential(
            # nn.Conv2d(inp, 1, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Softmax2d(),
            # nn.Upsample(scale_factor=ur,mode=up_style),
        )
        self.ska = AsymmetricSKA(oup,kernels,nxm,rska)
        
    def forward(self, x):
        _, _, H,W = x[1].shape
        out = self.coordattmap(x[0])
        out = nn.functional.interpolate(out, size=(H,W), mode="nearest") * x[1] *10 
        # out = self.coordattmap(x) 
        # if out.shape[1] == 64:
        #     save_all_feature_maps(out, save_dir="visualize/attn2", prefix="aligned")
        out = self.ska(out)  
        # out = self.ska(x[1])  
        # out = x[1]
        return out