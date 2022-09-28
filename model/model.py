''' 
@Author: Wei-Cheng Lin.
@Email: steven61413@gmail.com 

'''
import torch
import torch.nn as nn 
import torch.nn.functional as F
from common.config import opts
from common.utils import get_graph_feature, knn, gen_grid

def xavier_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class DenseGCN(nn.Module):
    def __init__(self, opt, k=20, d=1):
        super().__init__()

        self.k = k
        self.d = d
        dim = opt.dim    
        self.layer_1 = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size=[1, 1], stride=1),
            nn.BatchNorm2d(dim*2),
            nn.GELU()
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(dim*3, dim*6, kernel_size=[1, 1], stride=1),
            nn.BatchNorm2d(dim*6),
            nn.GELU()
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(dim*9, dim*18, kernel_size=[1, 1], stride=1),
            nn.BatchNorm2d(dim*18),
            nn.GELU()
            )

    def forward(self, x):
        if self.d > 1:
            _, idx = knn(x.squeeze(-1), self.k*self.d)
            idx = idx[:, :, ::self.d]
        else:
            _, idx = knn(x.squeeze(-1), self.k)

        y = get_graph_feature(x, idx=idx)
        y = torch.max(y, dim=-1, keepdim=True)[0]

        y_1 = self.layer_1(y)
        y_1 = torch.cat((y, y_1), 1)

        y_2 = self.layer_2(y_1)
        y_2 = torch.cat((y_2, y_1), 1)

        y_3 = self.layer_3(y_2)
        y_3 = torch.cat((y_3, y_2), 1)

        output = torch.cat((y, y_1, y_2, y_3), 1)

        return output

class Attention(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()

        layer = dim_1 // 4
  
        self.f = nn.Sequential(
            nn.Conv2d(dim_1, layer, kernel_size=1),
            nn.BatchNorm2d(layer),
            nn.GELU(),
            )
        self.g = nn.Sequential(
            nn.Conv2d(dim_2, layer, kernel_size=1),
            nn.BatchNorm2d(layer),
            nn.GELU(),
            )
        self.h = nn.Sequential(
            nn.Conv2d(dim_2, dim_1, kernel_size=1),
            nn.BatchNorm2d(dim_1),
            nn.GELU(),
            )

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, q, k):

        f = self.f(q)
        g = self.g(k)
        h = self.h(k)
        s = g.squeeze(-1).permute(0, 2, 1).contiguous() @ f.squeeze(-1)  # # [bs, N, N]

        beta = self.softmax(s)  # attention map
        o = beta @ h.squeeze(-1).permute(0, 2, 1).contiguous()  # [bs, N, N]*[bs, N, c]->[bs, N, c]
        o = o.unsqueeze(-1).permute(0, 2, 1, 3).contiguous()
        x = self.gamma * o + q

        return x

class Inception_GCN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.densegcn_1 = DenseGCN(opt, d=1)
        self.densegcn_2 = DenseGCN(opt, d=opt.d)

    def forward(self, x):

        incep_1 = self.densegcn_1(x) # b, 240, n, 1
        incep_2 = self.densegcn_2(x)

        bottleneck = torch.max(x, dim=1, keepdim=True)[0]
        
        output = torch.cat((incep_1, incep_2, bottleneck, x), dim=1)
        return output

class up_block(nn.Module):
    def __init__(self, opt, channels=None):
        super().__init__()
        self.grid = gen_grid(opt.up_ratio).unsqueeze(0).to(opt.device)
        self.in_channels = channels + 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, opt.feature_dim, kernel_size=1),
            nn.BatchNorm2d(opt.feature_dim),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(opt.feature_dim+self.in_channels, opt.feature_dim, kernel_size=1),
            nn.BatchNorm2d(opt.feature_dim),
            nn.GELU()
        )
        self.self_attn = Attention(dim_1=opt.feature_dim+self.in_channels, dim_2=opt.feature_dim+self.in_channels) # self-attention
    def forward(self, x):
        b, c, n, _ = x.shape
        grid = self.grid.clone()
        grid = self.grid.repeat((b, 1 ,n)).view([b, -1, 2]).unsqueeze(-1)
        grid = grid.transpose(1, 2).contiguous()
        x = x.repeat(1, 1, 4, 1)
        x = torch.cat((x, grid), dim=1)
        x = torch.cat((self.conv1(x), x), dim=1)
        x = self.self_attn(x, x)
        x = self.conv2(x)
        return x

class Auxiliary_network(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.up_ratio = opt.up_ratio
        self.mlp = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.Conv2d(256, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.GELU(),
        )
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))
    def forward(self, x):
        x1 = self.mlp(x)
        x1 = self.pooling(x1)
        output = x1.repeat(1, 1, x.shape[2]*self.up_ratio, 1)
        return output


class Multibranch_upunit(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.channel_1 = 64
        self.channel_2 = 128
        self.channel_3 = 256

        # Muti-scale dense features
        self.conv_1 = nn.Sequential(
            nn.Conv2d(484, self.channel_1, kernel_size=1),
            nn.BatchNorm2d(self.channel_1),
            nn.GELU(),
            ) # 484
        self.conv_2 = nn.Sequential(nn.Conv2d(484, self.channel_2, kernel_size=1),
            nn.BatchNorm2d(self.channel_2),
            nn.GELU(),
            )
        self.conv_3 = nn.Sequential(nn.Conv2d(484, self.channel_3, kernel_size=1),
            nn.BatchNorm2d(self.channel_3),
            nn.GELU(),
            ) 

        self.up_1 = up_block(opt=opt, channels=self.channel_1)
        self.up_2 = up_block(opt=opt, channels=self.channel_2)
        self.up_3 = up_block(opt=opt, channels=self.channel_3)

        self.attn = Attention(dim_1=opt.feature_dim, dim_2=512)

    def forward(self, x):

        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)

        up1 = self.up_1(x1)
        up2 = self.up_2(x2)
        up3 = self.up_3(x3)

        up = up1 + up2 + up3

        return up

class Enhanced_PCU(nn.Module):
    def __init__(self, opt=None):
        super(Enhanced_PCU, self).__init__()

        self.feature_block = Inception_GCN(opt)
        self.up_unit = Multibranch_upunit(opt)
        self.aux = Auxiliary_network(opt)
        self.attention_fusion = Attention(dim_1=opt.feature_dim, dim_2=512)

        self.recon = nn.Sequential(
            nn.Conv2d(opt.feature_dim, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=1),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        dense_feature = self.feature_block(x)
        up_feature = self.up_unit(dense_feature)

        global_feature = self.aux(x)
        
        fusion_features = self.attention_fusion(up_feature, global_feature)
        recon = self.recon(fusion_features)
        recon = recon.squeeze(-1)

        return recon


if __name__ == '__main__':
    opt = opts().parse()
    t = torch.randn(32, 3, 256).to(opt.device)

    model = Enhanced_PCU(opt).to(opt.device)

    output = model(t)
    print(sum(p.numel() for p in model.parameters()))
    print(f"Output shape : {output.shape}")


