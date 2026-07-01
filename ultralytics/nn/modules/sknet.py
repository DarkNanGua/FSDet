import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

from ultralytics.nn.modules.conv import AsymmetricConv

# class SKAttention(nn.Module):

#     def __init__(self, channel=512,kernels=[1,3,5],reduction=16,group=1,L=32):
#         super().__init__()
#         if group == -1 :
#             group = channel
#         self.d=max(L,channel//reduction)
#         self.convs=nn.ModuleList([])
#         for k in kernels:
#             self.convs.append(
#                 nn.Sequential(OrderedDict([
#                     ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group,bias=False)),
#                     ('point_conv',nn.Sequential(
#                         nn.BatchNorm2d(channel),
#                         # nn.Hardswish(),
#                         # nn.Conv2d(channel, channel, kernel_size=1, bias=False)
#                     ) if group != 1 else nn.Identity()),
#                     ('bn',nn.BatchNorm2d(channel)),
#                     ('h_silu',nn.Hardswish()),

#                 ]))
#             )
#         self.fc=nn.Linear(channel,self.d)
#         self.fcs=nn.ModuleList([])
#         for i in range(len(kernels)):
#             self.fcs.append(nn.Linear(self.d,channel))
#         self.softmax=nn.Softmax(dim=0)



#     def forward(self, x):
#         bs, c, _, _ = x.size()
#         conv_outs=[]
#         ### split
#         for conv in self.convs:
#             conv_outs.append(conv(x))
#         feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

#         ### fuse
#         U=sum(conv_outs) #bs,c,h,w,残差求和

#         ### reduction channel
#         S=U.mean(-1).mean(-1) #bs,c,1
#         Z=self.fc(S) #bs,d

#         ### calculate attention weight
#         weights=[]
#         for fc in self.fcs:
#             weight=fc(Z)
#             weights.append(weight.view(bs,c,1,1)) #bs,channel
#         attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
#         attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

#         ### fuse
#         V=(attention_weughts*feats).sum(0)
#         return V

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

class SKAttention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5],nxm = True,reduction=16,group=1,L=32):
        super().__init__()
        if group == -1 :
            group = channel
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',AsymmetricConv(channel,channel,kernel_size=k,groups=group,bias=False,nxm = nxm)),
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
