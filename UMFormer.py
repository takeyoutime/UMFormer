import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
# from geoseg.models.VMUnet import vmamba
from geoseg.models.umamba_v12.ssm import Mamba_Block
import math
from functools import partial
from einops import rearrange
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
from typing import Callable
from geoseg.models.My.Att import Att, ChannelAttention
from thop import profile
import timm


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            # nn.Dropout(0.1)
        )


class Conformity(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conformity, self).__init__()
        self.Att = Att(in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.Att(x)
        x = self.conv(x)

        return x


class IndentityBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters, rate=1):
        super(IndentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            # nn.Conv2d(F1, F2, kernel_size, stride=1, dilation=rate, padding=rate, bias=False),
            # nn.BatchNorm2d(F2),
            # nn.ReLU(True),
            SeparableConvBNReLU(F1, F2, kernel_size, dilation=rate),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class GlobeAttention(nn.Module):
    def __init__(self, dim_out):
        super(GlobeAttention, self).__init__()
        self.branch1 = nn.Sequential(
            SeparableConvBNReLU(64, dim_out, kernel_size=3, stride=2),
            SeparableConvBNReLU(dim_out, dim_out, kernel_size=3, stride=2),
            SeparableConvBNReLU(dim_out, dim_out, kernel_size=3, stride=2)
        )
        self.branch2 = nn.Sequential(
            SeparableConvBNReLU(128, dim_out, kernel_size=3, stride=2),
            SeparableConvBNReLU(dim_out, dim_out, kernel_size=3, stride=2)
        )
        self.branch3 = nn.Sequential(
            SeparableConvBNReLU(256, dim_out, kernel_size=3, stride=2)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(512, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU6()
        )
        self.merge = nn.Sequential(
            nn.Conv2d(4 * dim_out, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU6()
        )
        self.resblock = nn.Sequential(
            IndentityBlock(in_channel=dim_out, kernel_size=3, filters=[dim_out, dim_out, dim_out]),
            IndentityBlock(in_channel=dim_out, kernel_size=3, filters=[dim_out, dim_out, dim_out])
        )
        self.transformer = SelfAttention()
        self.conv = nn.Conv2d(1280, dim_out, 1)

    def forward(self, x, skip_list):
        b, c, h, w = x.shape
        list1 = []
        list2 = []

        x1 = self.branch1(skip_list[0])
        x2 = self.branch2(skip_list[1])
        x3 = self.branch3(skip_list[2])
        x = self.branch4(x)

        # CNN
        merge = self.merge(torch.cat([x, x1, x2, x3], dim=1))
        merge = self.resblock(merge)

        # Transformer
        list1.append(x)
        list1.append(x3)
        list1.append(x2)
        list1.append(x1)

        for i in range(len(list1)):
            for j in range(len(list1)):
                if i <= j:
                    att = self.transformer(list1[i], list1[j])
                    list2.append(att)

        for j in range(len(list2)):
            list2[j] = list2[j].view(b, 128, h, w)

        out = self.conv(torch.concat(list2, dim=1))

        return out + merge


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.conv = SeparableConvBNReLU(256, 384, 3)

    def forward(self, x, y):
        b, c, h, w = x.shape
        fm = self.conv(torch.concat([x, y], dim=1))

        Q, K, V = rearrange(fm, 'b (qkv c) h w -> qkv b h c w'
                            , qkv=3, b=b, c=128, h=h, w=w)

        dots = (Q @ K.transpose(-2, -1))
        attn = dots.softmax(dim=-1)
        attn = attn @ V
        attn = attn.view(b, c, h, w)

        return attn


class LocalFeature(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1):
        super(LocalFeature, self).__init__()
        self.branch1_1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.SiLU()
        )
        self.branch1_2 = nn.Sequential(
            SeparableConvBNReLU(dim_in, dim_out, kernel_size=3)
        )
        self.branch2 = IndentityBlock(in_channel=dim_in, kernel_size=3, filters=[dim_in, dim_out, dim_out], rate=1)
        self.branch3 = IndentityBlock(in_channel=dim_in, kernel_size=3, filters=[dim_in, dim_out, dim_out], rate=2)
        self.branch4 = IndentityBlock(in_channel=dim_in, kernel_size=3, filters=[dim_in, dim_out, dim_out], rate=3)
        self.conv = nn.Conv2d(3 * dim_in, dim_out, kernel_size=1)
        self.channel = ChannelAttention(dim_out)

    def forward(self, x):
        x_branch1_1 = self.branch1_1(x)
        x_branch1_2 = self.branch1_2(x)
        x_branch1 = x_branch1_1 + x_branch1_2

        x_branch2 = self.branch2(x_branch1)
        x_branch3 = self.branch3(x_branch2)
        x_branch4 = self.branch4(x_branch3)
        out = self.conv(torch.cat([x_branch2, x_branch3, x_branch4], dim=1))

        att = self.channel(out)
        out = out * att

        return out + x_branch1


class FeatureEmbedding(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeatureEmbedding, self).__init__()
        self.conv1 = SeparableConvBNReLU(in_channel, out_channel, kernel_size=3)
        self.conv2 = SeparableConvBNReLU(out_channel, out_channel, kernel_size=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        sum = x + y
        sum = self.conv1(sum)
        sim = F.cosine_similarity(x, y, dim=1, eps=1e-8).unsqueeze(1)
        sim = self.sigmoid(sim)
        out = sum * sim
        out = self.conv2(out)

        return out + sum


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        # self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = Mamba_Block(depth=1, embed_dim=hidden_dim // 4, out_dim=hidden_dim // 4,
                                          bimamba_type='None')
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x1, x2, x3, x4 = torch.chunk(input, 4, dim=1)

        x1 = self.drop_path(self.self_attention(x1))
        x2 = self.drop_path(self.self_attention(x2))
        x3 = self.drop_path(self.self_attention(x3))
        x4 = self.drop_path(self.self_attention(x4))

        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            conformity_dim,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            channel=0,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        self.localfeature = LocalFeature(dim, dim)
        self.featureembedding = FeatureEmbedding(dim, dim)

        self.conv1 = nn.Conv2d(128, 64, 1)

        if channel == 0:
            # self.channel_down = self.conv1
            self.channel_down = None
        else:
            self.channel_down = Conformity(conformity_dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        if self.channel_down:
            x = self.channel_down(x)

        globe = x
        local = self.localfeature(x)

        for blk in self.blocks:
            if self.use_checkpoint:
                globe = checkpoint.checkpoint(blk, globe)
            else:
                globe = blk(globe)

        x = self.featureembedding(globe, local) + globe
        x = F.interpolate(x, (h * 2, w * 2), None, 'bilinear', True)

        return x


class MambaDecoder(nn.Module):
    def __init__(self, num_classes, depths_decoder=[1, 1, 1, 1], dims=[64, 128, 256, 512],
                 dims_decoder=[128, 64, 64, 64], conformity_dim=[0, 384, 192, 128],
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 use_checkpoint=False):
        super(MambaDecoder, self).__init__()
        self.Decoder_layer = nn.ModuleList()
        self.num_layers = len(depths_decoder)
        self.init_weight()
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                conformity_dim=conformity_dim[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                channel=i_layer,
                use_checkpoint=use_checkpoint
            )
            self.Decoder_layer.append(layer)
        self.finalconv = nn.Conv2d(dims_decoder[-1], num_classes, 1)

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, skip_list):
        b, c, h, w = x.shape
        for inx, layer_decoder in enumerate(self.Decoder_layer):
            if inx == 0:
                x = layer_decoder(x)
            else:
                middle = torch.cat([x, skip_list[-inx]], dim=1)
                x = layer_decoder(middle)
        x = self.finalconv(F.interpolate(x, (h * 32, w * 32), None, 'bilinear', True))
        return x


class Mambaformer(nn.Module):
    def __init__(self, num_classes):
        super(Mambaformer, self).__init__()
        self.backbone = timm.create_model('swsl_resnet18', features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=True)

        self.GlobeAttention = GlobeAttention(128)

        self.mambadecoder = MambaDecoder(num_classes)

    def forward(self, x):
        skip_list = []
        res1, res2, res3, res4 = self.backbone(x)
        skip_list.append(res1)
        skip_list.append(res2)
        skip_list.append(res3)

        mid = self.GlobeAttention(res4, skip_list)

        out = self.mambadecoder(mid, skip_list)
        return out


device = torch.device('cuda:0')
if __name__ == '__main__':
    net = Mambaformer(6)
    net = net.cuda()
    dummy_input = torch.randn(1, 3, 1024, 1024).to(device)
    flops, params = profile(net, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    print('***************************')
    # x = torch.rand(2, 3, 512, 512).to(device)
    total = sum([param.nelement() for param in net.parameters()])  # 计算总参数量
    print("model size:", total / 1000 / 1000, "M")
    print('***************************')
    y = net(dummy_input)
    print(y.size())
