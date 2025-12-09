import einops
import torch
from torch import nn as nn
from torch.nn import functional as F
from Util.Util import Conv_atten
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np


class Vidformer(nn.Module):
    def __init__(self,patch_size,image_size,in_channels,out_channels,emd_dim,drop_out,depth,heads,dim_head,mlp_dim):
        super().__init__()
        self.TransInit=TransInint(patch_size,image_size,in_channels,emd_dim,drop_out)
        self.depth=depth
        self.block=nn.ModuleList()
        self.stem=nn.Conv3d(in_channels,in_channels,kernel_size=3,padding=1)
        original_patch=patch_size
        self.trans = nn.Sequential(Transformer(emd_dim, heads, dim_head, mlp_dim, image_size, original_patch, drop_out))
        self.cnn = stem(out_channels)
        self.dim_decrease=nn.ModuleList()
        self.atten=nn.ModuleList()
        self.pool=nn.MaxPool3d(kernel_size=(1,2,2))
        for i in range (len(depth)):
            feature_size=list(map(lambda x:x//(2**(i+2)),image_size))
            feature_size[0]=image_size[0]
            feature_size=tuple(feature_size)
            patch_size = list(map(lambda x: x // (2 ** (i+2)), original_patch))
            patch_size[0] = 25
            patch_size = tuple(patch_size)
            if i!=0:
                self.atten.append(Conv_atten(heads=heads,in_channels=out_channels,kernel_size=(2*(4-i)+1)))
            for j in range(depth[i]):
                self.block.append(block(emd_dim, heads, dim_head, mlp_dim, drop_out, out_channels, out_channels,
                                        feature_size, image_size,original_patch,patch_size))
        self.Generator=rPPGGnerator(out_channels,emd_dim)
    def forward(self,x,echo,batch_size,epoch_test):
        x = self.stem(x)
        x1 = self.TransInit(x)
        x1=self.trans(x1)
        x=self.cnn(x)
        num=0
        for i in range(len(self.depth)):
            if i!=0:
                x=self.atten[i-1](x)
            for j in range(self.depth[i]):
                 x,x1=self.block[j+num](x,x1,batch_size)
            num=num+self.depth[i]
            x = self.pool(x)
        x=self.Generator(x,x1,echo,batch_size,epoch_test)
        return x


class TransInint(nn.Module):
    def __init__(self,patch_size,image_size,in_channels,emd_dim,drop_out):
        super().__init__()
        video_len, video_height, video_width = pair(image_size)
        self.patch_t, self.patch_h, self.patch_w = pair(patch_size)
        assert video_len % self.patch_t == 0 and video_height % self.patch_h == 0 and video_width % self.patch_w == 0
        patch_num = (video_len // self.patch_t) * (video_height // self.patch_h) * (video_width // self.patch_w)
        patch_dim = in_channels * self.patch_t * self.patch_h * self.patch_w
        self.to_embedding = nn.Sequential(
            nn.Linear(patch_dim, emd_dim))
        self.emd_dim = emd_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, patch_num, emd_dim))
        #self.cls_token = nn.Parameter(torch.randn(1, 1, emd_dim))
        self.dropout = nn.Dropout(drop_out)


    def forward(self,x1):
        x1=einops.rearrange(x1,'b c (t dt) (h dh) (w dw) -> b (t h w) (dt dh dw c)', dt=self.patch_t, dh=self.patch_h, dw=self.patch_w)
        x1 = self.to_embedding(x1)
        #cls_token = einops.repeat(self.cls_token, '() n d -> b n d', b=x1.shape[0])
        #x1 = torch.cat((cls_token, x1), dim=1)
        x1 = self.pos_embedding + x1
        x1 = self.dropout(x1)
        return x1

class stem(nn.Module):
    def __init__(self,channel):
        super(stem, self).__init__()
        self.stem1=nn.Sequential(nn.Conv3d(3,channel//4,kernel_size=(1,5,5),padding=(0,2,2)),
                                 nn.GroupNorm(num_channels=channel//4,num_groups=2),
                                 nn.GELU())
        self.pool1=nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2))
        self.stem2=nn.Sequential(nn.Conv3d(channel//4,channel//2,kernel_size=3,padding=1),
                                 nn.GroupNorm(num_channels=channel//2,num_groups=4),
                                 nn.GELU())
        self.pool2=nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2))
        self.stem3=nn.Sequential(nn.Conv3d(channel//2,channel,kernel_size=3,padding=1),
                                 nn.GroupNorm(num_channels=channel,num_groups=8),
                                 nn.GELU())
    def forward(self,x):
        x=self.stem1(x)
        x=self.pool1(x)
        x=self.stem2(x)
        x=self.pool2(x)
        x=self.stem3(x)
        return x

class block(nn.Module):
    def __init__(self,emd_dim,heads,dim_head,mlp_dim,drop_out,in_channels,out_channels,feature_size,image_size,original_patch,patch_size):
        super().__init__()
        self.trans=nn.Sequential(Transformer(emd_dim, heads, dim_head, mlp_dim,image_size,original_patch, drop_out)
                                 )
        self.cnn = nn.ModuleList()
        for _ in range(1):
            self.cnn.append(CNN_block(in_channels,out_channels))
        self.interaction1=Conv2Trans(out_channels,feature_size,emd_dim,out_channels)
        self.interaction2=Trans2Conv(emd_dim,out_channels)
        #self.conv1=nn.Conv3d(out_channels,out_channels,kernel_size=1,stride=1)


    def forward(self,x,x1,batch_size):
        for cnns in self.cnn:
            x=cnns(x)+x
        x1=self.trans(x1+self.interaction1(x,batch_size))
        x=self.interaction2(x,x1)
        #x=self.conv1(x)
        return x,x1


class CNN_block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.module=nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=out_channels//16,num_channels=out_channels),
                nn.GELU()
           )

    def forward(self,x):
        x=self.module(x)
        return x

class Trans2Conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv=nn.Conv3d(in_channels,out_channels,kernel_size=1)
        self.conv1=nn.Sequential(nn.Conv3d(out_channels,out_channels,kernel_size=3,padding=1),
                                 nn.GroupNorm(num_groups=out_channels//16,num_channels=out_channels),
                                 nn.GELU())
        self.in_channels=in_channels

    def forward(self,x,x1):
        x1=x1.reshape(x.shape[0],self.in_channels,80,8,-1)
        x1=F.upsample(x1,size=(x.shape[2],x.shape[3],x.shape[4]),mode='trilinear')
        x1=self.conv(x1)
        x=self.conv1(x1)+x
        return x



class Conv2Trans(nn.Module):
    def __init__(self,in_channels,feature_size,emd_dim,out_channels):
        super().__init__()
        self.feature_size=feature_size
        self.feature_t,self.feature_h,self.feature_w=pair(feature_size)
        self.patch_dim=self.feature_h*self.feature_t*self.feature_w*out_channels
        self.to_embedding = nn.Sequential(
            nn.Linear(self.patch_dim//640, emd_dim))
        self.lnorm=nn.LayerNorm(emd_dim)

    def forward(self,x,batch_size):
        x=x.reshape(batch_size,640,-1)
        x=self.to_embedding(x)
        x=self.lnorm(x)
        return x


class MHSA(nn.Module):
    def __init__(self,dim,heads,dim_head,dropout=0.):
        super().__init__()
        inner_dim=heads*dim_head
        project_out = not (heads == 1 and dim_head == dim)
        self.heads=heads
        self.scale=dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()



    def forward(self,x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # q, k, v   (b, h, n, dim_head(64))

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



def pair(t):
    return t if isinstance(t,tuple) else (t,t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, original_size,patch_size,dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.video_len, self.video_height, self.video_width = pair(original_size)
        self.patch_t, self.patch_h, self.patch_w = pair(patch_size)
        self.spa_attn=PreNorm(dim,MHSA(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.time_attn = PreNorm(dim,MHSA(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        self.norm=nn.LayerNorm(dim)

    def forward(self, x):
        x=einops.rearrange(x,'b (n nt) h -> (b nt) n h',nt=self.video_len//self.patch_t)
        x1 = self.spa_attn(x)+x
        x=einops.rearrange(x,'(b nt) (nh nw) h -> (b nh nw) nt h',nt=self.video_len//self.patch_t,nh=self.video_height//self.patch_h,nw=self.video_width//self.patch_w)
        x2 = self.time_attn(x)+x
        x2=einops.rearrange(x2,'(b nh nw) nt h -> b (nt nw nh) h', nt=self.video_len//self.patch_t,nh=self.video_height//self.patch_h,nw=self.video_width//self.patch_w)
        x1 = einops.rearrange(x1, '(b nt) n h -> b (n nt) h', nt=self.video_len//self.patch_t)
        x= self.ff(x1+x2) + self.norm(x1+x2)
        # x=self.ff(x)+x
        return x


class rPPGGnerator(nn.Module):
    def __init__(self,out_channels,emd_dim):
        super().__init__()
        self.Conv=nn.Conv3d(out_channels,1,kernel_size=1)
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.transGener=nn.Sequential(
            nn.Linear(250 * emd_dim, 250)
        )
        self.conv1d=nn.Conv1d(640,250,kernel_size=1)
        self.conv1d_1=nn.Conv1d(1,1,kernel_size=1)


    def forward(self,x,x1,echo,batch_size,epoch_test):
        x1=self.conv1d(x1)
        x1=torch.flatten(x1,1)
        x1=self.transGener(x1)
        x = self.Conv(x)
        x=self.pool(x)
        return x.squeeze(),x1.squeeze()
