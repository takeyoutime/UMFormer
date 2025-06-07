import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from einops import rearrange, repeat

import timm
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange

# from .local_mamba import Mamba_Block
from ssm import Mamba_Block
from trans import TransformerEncoder, TransformerEncoderLayer


class ConvBNSiLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            # nn.ReLU6()
            nn.SiLU(inplace=True)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class Residual_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, if_downsample=False, **kwargs):
        super(Residual_Block, self).__init__()
        self.if_downsample = if_downsample
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel))

    def forward(self, x):
        identity = x
        if self.if_downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class SpatialAttnLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.avgpool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.maxpool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.avgpool_w = nn.AdaptiveAvgPool2d((1, None))
        self.maxpool_w = nn.AdaptiveMaxPool2d((1, None))

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(inplace=True),
        )
        self.norm = nn.BatchNorm2d(d_model)
        self.norm1 = nn.GroupNorm(16, d_model)
        self.norm2 = nn.GroupNorm(16, d_model)

        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(d_model, d_model, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x, v):
        B, C, H, W = v.shape

        x_h = (self.avgpool_h(x) + self.maxpool_h(x)).squeeze(3).permute(0, 2, 1)
        x_w = (self.avgpool_w(x) + self.maxpool_w(x)).squeeze(2).permute(0, 2, 1)
        q = self.mlp(x_h)
        k = self.mlp(x_w).transpose(-1, -2)

        weight_score = torch.matmul(q, k)
        weight_probs = nn.ReLU(inplace=True)(weight_score).unsqueeze(1)

        src2 = weight_probs * v

        src = v + self.norm(src2)
        src = self.norm1(src)
        src2 = self.ffn(src)
        src = src + src2
        src = self.norm2(src)

        return src.reshape(B, C, H, W)


class SpaceAttn(nn.Module):
    def __init__(self, d_model, num_layers, norm=None):
        super().__init__()
        SAL = SpatialAttnLayer(d_model)
        self.layers = nn.ModuleList([copy.deepcopy(SAL) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src1, src2=None) -> torch.Tensor:
        if src2 is None:
            src2 = src1
        output = src1
        for layer in self.layers:
            output = layer(output, src2)

        if self.norm is not None:
            output = self.norm(output)

        return output


class MF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, img_size=16, depth=4, bimamba_type='v2', dropout=0.1):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.mamba = Mamba_Block(img_size, depth=depth, embed_dim=decode_channels, out_dim=decode_channels,
                                 bimamba_type=bimamba_type)
        self.post_conv = Residual_Block(decode_channels, decode_channels)
        self.attn = SpaceAttn(decode_channels, 1)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        res = self.pre_conv(res)
        x = self.mamba(x, self.attn(res))
        x = self.post_conv(x)

        return x


# class FeatureDetailDistiller(nn.Module):
#     def __init__(self, in_channels=64, decode_channels=64, dropout=0.1):
#         super().__init__()
#         self.num_channels = decode_channels
#         self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
#         self.attn = SpaceAttn(decode_channels, 1)

#         self.sa = Conv(2, 1, kernel_size=7)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.ca = nn.Sequential(
#             nn.Conv2d(decode_channels, decode_channels // 8, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(decode_channels // 8, decode_channels, 1, padding=0, bias=True),
#         )
#         self.pa = nn.Conv2d(2 * decode_channels, decode_channels, 7, padding=3, padding_mode='reflect', groups=decode_channels, bias=True)
#         self.sigmoid = nn.Sigmoid()

#         self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.eps = 1e-8

#         self.proj = Residual_Block(decode_channels*2, decode_channels, if_downsample=True)
#         self.conv = Conv(decode_channels, decode_channels, kernel_size=1)


#     def forward(self, x, res):
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
#         res = self.attn(self.pre_conv(res))
#         x_f = self.proj(torch.cat([x, res], dim=1))
#         identity = x

#         x_avg = torch.mean(x_f, dim=1, keepdim=True)
#         x_max, _ = torch.max(x_f, dim=1, keepdim=True)
#         x_sa = torch.cat([x_avg, x_max], dim=1)
#         x_sa = self.sa(x_sa)

#         x_gap = self.gap(x_f)
#         x_ca = self.ca(x_gap)

#         weights = nn.ReLU()(self.weights)
#         fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
#         x_a = fuse_weights[0] * x_sa + fuse_weights[1] * x_ca

#         x1 = x.unsqueeze(dim=2) # B, C, 1, H, W
#         x_a = x_a.unsqueeze(dim=2) # B, C, 1, H, W
#         x2 = torch.cat([x1, x_a], dim=2) # B, C, 2, H, W
#         x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
#         w = self.pa(x2)
#         w = self.sigmoid(w)

#         x = x_f + w * res + (1 - w) * identity
#         x = self.conv(x)

#         return x


# class FeatureDetailDistiller(nn.Module):
#     def __init__(self, in_channels=64, decode_channels=64, dropout=0.1):
#         super().__init__()
#         self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
#         self.post_conv = Residual_Block(decode_channels * 2, decode_channels, if_downsample=True)
#         self.attn = SpaceAttn(decode_channels, 1)
#         self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.eps = 1e-8

#     def forward(self, x, res):
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
#         identity = x
#         res = self.pre_conv(res)
#         x = torch.cat([x, self.attn(res)], dim=1)
#         weights = nn.ReLU()(self.weights)
#         fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
#         x = fuse_weights[0] * self.post_conv(x) + fuse_weights[1] * identity

#         return x

class FeatureDetailDistiller(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64, dropout=0.1):
        super().__init__()
        self.channel = decode_channels
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.attn = SpaceAttn(decode_channels, 1)

        self.conv_f = ConvBN(decode_channels * 2, decode_channels * 2, kernel_size=1)

        self.conv1x1 = ConvBNSiLU(decode_channels // 2, decode_channels // 4, kernel_size=1)
        self.conv3x3 = ConvBNSiLU(decode_channels // 2, decode_channels // 4, kernel_size=3)
        self.conv5x5 = ConvBNSiLU(decode_channels // 2, decode_channels // 4, kernel_size=3, dilation=2)
        self.conv7x7 = ConvBNSiLU(decode_channels // 2, decode_channels // 4, kernel_size=3, dilation=3)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = Residual_Block(decode_channels, decode_channels, if_downsample=True)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        identity = x
        res = self.pre_conv(res)
        x = torch.cat([x, self.attn(res)], dim=1)
        x = self.conv_f(x)
        x1 = self.conv1x1(x[:, :self.channel // 2, :, :])
        x2 = self.conv3x3(x[:, self.channel // 2: self.channel, :, :])
        x3 = self.conv5x5(x[:, self.channel:self.channel // 2 * 3, :, :])
        x4 = self.conv7x7(x[:, self.channel // 2 * 3:, :, :])
        x = torch.cat([x1, x2, x3, x4], dim=1)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * identity + fuse_weights[1] * x
        x = self.post_conv(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNSiLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels,
                 decode_channels,
                 num_classes,
                 depth=4,
                 bimamba_type='v2',
                 dropout=0.1,
                 window_size=8
                 ):
        super().__init__()
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            decode_channels,
            nhead=8,
            dim_feedforward=decode_channels * 2,
            dropout=dropout,
            activation='silu')
        self.attn = TransformerEncoder(copy.deepcopy(encoder_layer), 3)
        # self.attn = SpaceAttn(decode_channels, 3)

        self.p3 = MF(encoder_channels[-2], decode_channels, img_size=32, depth=depth, bimamba_type=bimamba_type)
        self.p2 = MF(encoder_channels[-3], decode_channels, img_size=64, depth=depth, bimamba_type=bimamba_type)
        self.p1 = FeatureDetailDistiller(encoder_channels[-4], decode_channels)

        self.h = AuxHead(decode_channels, num_classes)

        self.segmentation_head = nn.Sequential(
            ConvBNSiLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, num_classes, kernel_size=1)
        )
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.pre_conv(res4)
        # x = self.attn(x)
        B, C, H, W = x.shape
        x = self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)
        if self.training:
            # 这个地方会增加参数量，res4通道更多
            ah = self.h(x)

        x = self.p3(x, res3)
        x = self.p2(x, res2)
        x = self.p1(x, res1)

        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        if self.training:
            return x, ah
        else:
            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class UMamba(nn.Module):
    def __init__(self,
                 decode_channels=128,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 num_classes=6,
                 depth=4,
                 bimamba_type='v2',
                 ):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()

        self.decoder = Decoder(encoder_channels, decode_channels, num_classes, depth, bimamba_type)

    def forward(self, x):
        _, _, H, W = x.shape
        res1, res2, res3, res4 = self.backbone(x)

        if self.training:
            x, ah = self.decoder(res1, res2, res3, res4, H, W)

            return [x, ah]
        else:
            x = self.decoder(res1, res2, res3, res4, H, W)

            return x


if __name__ == '__main__':
    data = torch.rand(1, 3, 512, 512).to('cuda:0')
    net = UMamba(num_classes=6).to('cuda:0')

    # weights_dict = torch.load('vim_t_midclstok_ft_78p3acc.pth', map_location='cuda:0')
    # load_weights_dict = {k: v for k, v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
    # print(model.load_state_dict(load_weights_dict, strict=False))

    # model.load_state_dict(torch.load('vim_t_midclstok_ft_78p3acc.pth', map_location='cuda:0'))

    out = net(data)
    print(out[0].shape)
