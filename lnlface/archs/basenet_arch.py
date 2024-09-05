import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.spectral_norm as SpectralNorm
from math import sqrt
from lnlface.archs.vision_lstm2 import VisionLSTM2
from basicsr.utils.registry import ARCH_REGISTRY

def calc_mean_std_4D(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization_4D(content_feat,
                                       style_feat):  # content_feat is ref feature, style is degradate feature
    size = content_feat.size()
    style_mean, style_std = calc_mean_std_4D(style_feat)
    content_mean, content_std = calc_mean_std_4D(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def convU(in_channels, out_channels, conv_layer, norm_layer, kernel_size=3, stride=1, dilation=1, bias=True):
    return nn.Sequential(
        SpectralNorm(conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                padding=((kernel_size - 1) // 2) * dilation, bias=bias)),
        nn.LeakyReLU(0.2),
        SpectralNorm(conv_layer(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                padding=((kernel_size - 1) // 2) * dilation, bias=bias)),
    )


class MSDilateBlock(nn.Module):
    def __init__(self, in_channels, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, kernel_size=3,
                 dilation=[1, 1, 1, 1], bias=True):
        super(MSDilateBlock, self).__init__()
        self.conv1 = convU(in_channels, in_channels, conv_layer, norm_layer, kernel_size, dilation=dilation[0],
                           bias=bias)
        self.conv2 = convU(in_channels, in_channels, conv_layer, norm_layer, kernel_size, dilation=dilation[1],
                           bias=bias)
        self.conv3 = convU(in_channels, in_channels, conv_layer, norm_layer, kernel_size, dilation=dilation[2],
                           bias=bias)
        self.conv4 = convU(in_channels, in_channels, conv_layer, norm_layer, kernel_size, dilation=dilation[3],
                           bias=bias)
        self.convi = SpectralNorm(
            conv_layer(in_channels * 4, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                       bias=bias))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.convi(cat) + x
        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)

    def forward(self, input, style):
        style_mean, style_std = calc_mean_std_4D(style)
        out = self.norm(input)
        size = input.size()
        out = style_std.expand(size) * out + style_mean.expand(size)
        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        if noise is None:
            b, c, h, w = image.shape
            noise = image.new_empty(b, 1, h, w).normal_()
        return image + self.weight * noise


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class ChannelAttention(nn.Module):
    def __init__(self, base_filters, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        kernel_size = 1
        conv = default_conv
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv(base_filters, base_filters // squeeze_factor, kernel_size),
            nn.ReLU(inplace=True),
            conv(base_filters // squeeze_factor, base_filters, kernel_size),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return y * x


class DegenerateAware(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DegenerateAware, self).__init__()

        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, in_channels * 2, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(in_channels * 2, in_channels * 2, 3, 1, 1))
        )
        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels * 4, in_channels * 4, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(in_channels * 4, in_channels * 4, 3, 1, 1))
        )
        self.conv3 = convU(in_channels * 8, out_channels, nn.Conv2d, nn.BatchNorm2d, kernel_size=3, stride=1)
        self.main = VisionLSTM2()

    def forward(self, x1, x2, x):
        x1 = self.conv1(x1)
        x2 = torch.cat((x1, x2), dim=1)
        x2 = self.conv2(x2)
        x3 = torch.cat((x2, x), dim=1)
        x3 = self.conv3(x3)
        out = self.main(x3)
        return out


class FusionUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, ngf, scale=8, kernel_size=3, padding=1):
        super(FusionUpBlock, self).__init__()
        self.quantup = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            SpectralNorm(nn.Conv2d(ngf * 4, in_channel, kernel_size, padding=padding)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(in_channel, in_channel, kernel_size, padding=padding)),
        )
        self.convup = nn.Sequential(
            nn.Conv2d(out_channel, out_channel * 4, 1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding),
        )
        self.ScaleModel1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, out_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1))
        )
        self.ShiftModel1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, out_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)),
        )
        self.ScaleModel2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1))
        )
        self.ShiftModel2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)),
        )
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.attention = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, in_channel // 2, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(in_channel // 2, in_channel // 4, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(in_channel // 4, in_channel // 2, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(in_channel // 2, in_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(in_channel, out_channel, 3, 1, 1)),
        )
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)),
        )
        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)),
        )
        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(out_channel * 2, out_channel, kernel_size, padding=padding)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)),
        )

        self.main = nn.Sequential(
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            UpResBlock(out_channel),
            UpResBlock(out_channel),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)),
        )

    def forward(self, x, upin, quant):
        quantstyle = self.lrelu1(self.quantup(quant))
        Shift1 = self.ShiftModel1(quantstyle)
        Scale1 = self.ScaleModel1(quantstyle)
        upin = self.lrelu1(self.conv1(upin))
        upin = upin * Scale1 + Shift1

        x = self.lrelu1(self.conv2(x))
        upin = torch.cat((upin, x), dim=1)
        upin = self.lrelu1(self.conv3(upin))
        att = self.attention(quantstyle)
        upin = upin * att + upin

        Shift2 = self.ShiftModel2(upin)
        Scale2 = self.ScaleModel2(upin)

        x = x * Scale2 + Shift2
        x = self.convup(x)

        outup = self.main(x) + x

        return outup


class StyledUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, upsample=False, noise_inject=False):
        super().__init__()

        self.noise_inject = noise_inject
        if upsample:
            self.conv1 = nn.Sequential(
                nn.Conv2d(out_channel, out_channel * 4, 1, bias=False),
                nn.PixelShuffle(2),
                nn.Conv2d(out_channel, out_channel, kernel_size),
            )
        else:
            self.conv1 = nn.Sequential(
                SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)),
                nn.LeakyReLU(0.2),
                SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)),
            )
        self.convup = nn.Sequential(
            nn.Conv2d(out_channel, out_channel * 4, 1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)
        )
        if self.noise_inject:
            self.noise1 = NoiseInjection(out_channel)

        self.lrelu1 = nn.LeakyReLU(0.2)

        self.ScaleModel1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, out_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1))
        )
        self.ShiftModel1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, out_channel, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)),
        )

    def forward(self, input, style):
        out = self.conv1(input)
        out = self.lrelu1(out)
        Shift1 = self.ShiftModel1(style)
        Scale1 = self.ScaleModel1(style)
        out = out * Scale1 + Shift1
        if self.noise_inject:
            out = self.noise1(out, noise=None)
        outup = self.convup(out)
        return outup


####################################################################
###############Face Dictionary Generator
####################################################################
def AttentionBlock(in_channel):
    return nn.Sequential(
        SpectralNorm(nn.Conv2d(in_channel, in_channel, 3, 1, 1)),
        nn.LeakyReLU(0.2),
        SpectralNorm(nn.Conv2d(in_channel, in_channel, 3, 1, 1)),
    )


class DilateResBlock(nn.Module):
    def __init__(self, dim, dilation=[5, 3]):
        super(DilateResBlock, self).__init__()
        self.Res = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim, dim, 3, 1, ((3 - 1) // 2) * dilation[0], dilation[0])),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim, dim, 3, 1, ((3 - 1) // 2) * dilation[1], dilation[1])),
        )

    def forward(self, x):
        out = x + self.Res(x)
        return out


class KeyValue(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Sequential(
            SpectralNorm(nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(keydim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)),
        )
        self.Value = nn.Sequential(
            SpectralNorm(nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(valdim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)),
        )

    def forward(self, x):
        return self.Key(x), self.Value(x)


class MaskAttention(nn.Module):
    def __init__(self, indim):
        super(MaskAttention, self).__init__()
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(indim, indim // 3, kernel_size=(3, 3), padding=(1, 1), stride=1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(indim // 3, indim // 3, kernel_size=(3, 3), padding=(1, 1), stride=1)),
        )
        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(indim, indim // 3, kernel_size=(3, 3), padding=(1, 1), stride=1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(indim // 3, indim // 3, kernel_size=(3, 3), padding=(1, 1), stride=1)),
        )
        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(indim, indim // 3, kernel_size=(3, 3), padding=(1, 1), stride=1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(indim // 3, indim // 3, kernel_size=(3, 3), padding=(1, 1), stride=1)),
        )
        self.convCat = nn.Sequential(
            SpectralNorm(nn.Conv2d(indim // 3 * 3, indim, kernel_size=(3, 3), padding=(1, 1), stride=1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(indim, indim, kernel_size=(3, 3), padding=(1, 1), stride=1)),
        )

    def forward(self, x, y, z):
        c1 = self.conv1(x)
        c2 = self.conv2(y)
        c3 = self.conv3(z)
        return self.convCat(torch.cat([c1, c2, c3], dim=1))


class Query(nn.Module):
    def __init__(self, indim, quedim):
        super(Query, self).__init__()
        self.Query = nn.Sequential(
            SpectralNorm(nn.Conv2d(indim, quedim, kernel_size=(3, 3), padding=(1, 1), stride=1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(quedim, quedim, kernel_size=(3, 3), padding=(1, 1), stride=1)),
        )

    def forward(self, x):
        return self.Query(x)


def roi_align_self(input, location, target_size):
    return torch.cat([F.interpolate(input[i:i + 1, :, location[i, 1]:location[i, 3], location[i, 0]:location[i, 2]],
                                    (target_size, target_size), mode='bilinear', align_corners=False) for i in
                      range(input.size(0))], 0)


class FeatureExtractor(nn.Module):
    def __init__(self, ngf=64, key_scale=4):  #
        super().__init__()

        self.key_scale = 4
        self.part_sizes = np.array([80, 80, 50, 110])  #
        self.feature_sizes = np.array([256, 128, 64])  #

        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, ngf, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(ngf, ngf, 3, 1, 1)),
        )
        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf, ngf, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(ngf, ngf, 3, 1, 1))
        )
        self.res1 = DilateResBlock(ngf, [5, 3])
        self.res2 = DilateResBlock(ngf, [5, 3])

        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf, ngf * 2, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1)),
        )
        self.conv4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1))
        )
        self.res3 = DilateResBlock(ngf * 2, [3, 1])
        self.res4 = DilateResBlock(ngf * 2, [3, 1])

        self.conv5 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1)),
        )
        self.conv6 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1))
        )
        self.res5 = DilateResBlock(ngf * 4, [1, 1])
        self.res6 = DilateResBlock(ngf * 4, [1, 1])

        self.LE_256_Q = Query(ngf, ngf // self.key_scale)
        self.RE_256_Q = Query(ngf, ngf // self.key_scale)
        self.MO_256_Q = Query(ngf, ngf // self.key_scale)
        self.LE_128_Q = Query(ngf * 2, ngf * 2 // self.key_scale)
        self.RE_128_Q = Query(ngf * 2, ngf * 2 // self.key_scale)
        self.MO_128_Q = Query(ngf * 2, ngf * 2 // self.key_scale)
        self.LE_64_Q = Query(ngf * 4, ngf * 4 // self.key_scale)
        self.RE_64_Q = Query(ngf * 4, ngf * 4 // self.key_scale)
        self.MO_64_Q = Query(ngf * 4, ngf * 4 // self.key_scale)

    def forward(self, img, locs):
        le_location = locs[:, 0, :].int().cpu().numpy()
        re_location = locs[:, 1, :].int().cpu().numpy()
        no_location = locs[:, 2, :].int().cpu().numpy()
        mo_location = locs[:, 3, :].int().cpu().numpy()

        f1_0 = self.conv1(img)
        f1_1 = self.res1(f1_0)
        f2_0 = self.conv2(f1_1)
        f2_1 = self.res2(f2_0)

        f3_0 = self.conv3(f2_1)
        f3_1 = self.res3(f3_0)
        f4_0 = self.conv4(f3_1)
        f4_1 = self.res4(f4_0)

        f5_0 = self.conv5(f4_1)
        f5_1 = self.res5(f5_0)
        f6_0 = self.conv6(f5_1)
        f6_1 = self.res6(f6_0)

        ####ROI Align
        le_part_256 = roi_align_self(f2_1.clone(), le_location // 2, self.part_sizes[0] // 2)
        re_part_256 = roi_align_self(f2_1.clone(), re_location // 2, self.part_sizes[1] // 2)
        mo_part_256 = roi_align_self(f2_1.clone(), mo_location // 2, self.part_sizes[3] // 2)

        le_part_128 = roi_align_self(f4_1.clone(), le_location // 4, self.part_sizes[0] // 4)
        re_part_128 = roi_align_self(f4_1.clone(), re_location // 4, self.part_sizes[1] // 4)
        mo_part_128 = roi_align_self(f4_1.clone(), mo_location // 4, self.part_sizes[3] // 4)

        le_part_64 = roi_align_self(f6_1.clone(), le_location // 8, self.part_sizes[0] // 8)
        re_part_64 = roi_align_self(f6_1.clone(), re_location // 8, self.part_sizes[1] // 8)
        mo_part_64 = roi_align_self(f6_1.clone(), mo_location // 8, self.part_sizes[3] // 8)

        le_256_q = self.LE_256_Q(le_part_256)
        re_256_q = self.RE_256_Q(re_part_256)
        mo_256_q = self.MO_256_Q(mo_part_256)

        le_128_q = self.LE_128_Q(le_part_128)
        re_128_q = self.RE_128_Q(re_part_128)
        mo_128_q = self.MO_128_Q(mo_part_128)

        le_64_q = self.LE_64_Q(le_part_64)
        re_64_q = self.RE_64_Q(re_part_64)
        mo_64_q = self.MO_64_Q(mo_part_64)

        return {'f256': f2_1, 'f128': f4_1, 'f64': f6_1, \
                'le256': le_part_256, 're256': re_part_256, 'mo256': mo_part_256, \
                'le128': le_part_128, 're128': re_part_128, 'mo128': mo_part_128, \
                'le64': le_part_64, 're64': re_part_64, 'mo64': mo_part_64, \
                'le_256_q': le_256_q, 're_256_q': re_256_q, 'mo_256_q': mo_256_q, \
                'le_128_q': le_128_q, 're_128_q': re_128_q, 'mo_128_q': mo_128_q, \
                'le_64_q': le_64_q, 're_64_q': re_64_q, 'mo_64_q': mo_64_q}


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings

        min_value, min_encoding_indices = torch.min(d, dim=1)

        min_encoding_indices = min_encoding_indices.unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # .........\end

        # with:
        # .........\start
        # min_encoding_indices = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
               torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices, d)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


@ARCH_REGISTRY.register()
class LNLFaceNet(nn.Module):
    def __init__(self, ngf=64, banks_num=128, n_embed=1024, embed_dim=256, function_num=4, fixed_init=True):
        super().__init__()
        self.part_sizes = np.array([80, 80, 50, 110])  # size for 512
        self.feature_sizes = np.array([256, 128, 64])  # size for 512

        self.banks_num = banks_num
        self.key_scale = 4

        self.E_lq = FeatureExtractor(key_scale=self.key_scale)
        self.E_hq = FeatureExtractor(key_scale=self.key_scale)

        self.LE_256_KV = KeyValue(ngf, ngf // self.key_scale, ngf)
        self.RE_256_KV = KeyValue(ngf, ngf // self.key_scale, ngf)
        self.MO_256_KV = KeyValue(ngf, ngf // self.key_scale, ngf)

        self.LE_128_KV = KeyValue(ngf * 2, ngf * 2 // self.key_scale, ngf * 2)
        self.RE_128_KV = KeyValue(ngf * 2, ngf * 2 // self.key_scale, ngf * 2)
        self.MO_128_KV = KeyValue(ngf * 2, ngf * 2 // self.key_scale, ngf * 2)

        self.LE_64_KV = KeyValue(ngf * 4, ngf * 4 // self.key_scale, ngf * 4)
        self.RE_64_KV = KeyValue(ngf * 4, ngf * 4 // self.key_scale, ngf * 4)
        self.MO_64_KV = KeyValue(ngf * 4, ngf * 4 // self.key_scale, ngf * 4)

        self.LE_256_Attention = AttentionBlock(64)
        self.RE_256_Attention = AttentionBlock(64)
        self.MO_256_Attention = AttentionBlock(64)

        self.LE_128_Attention = AttentionBlock(128)
        self.RE_128_Attention = AttentionBlock(128)
        self.MO_128_Attention = AttentionBlock(128)

        self.LE_64_Attention = AttentionBlock(256)
        self.RE_64_Attention = AttentionBlock(256)
        self.MO_64_Attention = AttentionBlock(256)

        self.LE_256_Mask = MaskAttention(64)
        self.RE_256_Mask = MaskAttention(64)
        self.MO_256_Mask = MaskAttention(64)

        self.LE_128_Mask = MaskAttention(128)
        self.RE_128_Mask = MaskAttention(128)
        self.MO_128_Mask = MaskAttention(128)

        self.LE_64_Mask = MaskAttention(256)
        self.RE_64_Mask = MaskAttention(256)
        self.MO_64_Mask = MaskAttention(256)

        # define generic memory, revise register_buffer to register_parameter for backward update
        self.register_buffer('le_256_mem_key', torch.randn(128, 16, 40, 40))
        self.register_buffer('re_256_mem_key', torch.randn(128, 16, 40, 40))
        self.register_buffer('mo_256_mem_key', torch.randn(128, 16, 55, 55))
        self.register_buffer('le_256_mem_value', torch.randn(128, 64, 40, 40))
        self.register_buffer('re_256_mem_value', torch.randn(128, 64, 40, 40))
        self.register_buffer('mo_256_mem_value', torch.randn(128, 64, 55, 55))

        self.register_buffer('le_128_mem_key', torch.randn(128, 32, 20, 20))
        self.register_buffer('re_128_mem_key', torch.randn(128, 32, 20, 20))
        self.register_buffer('mo_128_mem_key', torch.randn(128, 32, 27, 27))
        self.register_buffer('le_128_mem_value', torch.randn(128, 128, 20, 20))
        self.register_buffer('re_128_mem_value', torch.randn(128, 128, 20, 20))
        self.register_buffer('mo_128_mem_value', torch.randn(128, 128, 27, 27))

        self.register_buffer('le_64_mem_key', torch.randn(128, 64, 10, 10))
        self.register_buffer('re_64_mem_key', torch.randn(128, 64, 10, 10))
        self.register_buffer('mo_64_mem_key', torch.randn(128, 64, 13, 13))
        self.register_buffer('le_64_mem_value', torch.randn(128, 256, 10, 10))
        self.register_buffer('re_64_mem_value', torch.randn(128, 256, 10, 10))
        self.register_buffer('mo_64_mem_value', torch.randn(128, 256, 13, 13))

        self.MSDilate = MSDilateBlock(ngf * 4, dilation=[4, 3, 2, 1])

        self.fusion_up1 = StyledUpBlock(ngf * 4, ngf * 2, noise_inject=False)

        self.deg_aware = DegenerateAware(ngf, ngf * 4)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.post_quant = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            SpectralNorm(nn.Conv2d(ngf * 4, ngf * 4, 3, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            SpectralNorm(nn.Conv2d(ngf * 4, ngf * 4, 3, padding=1)),
            nn.LeakyReLU(0.2),
        )

        self.fusion_up2 = FusionUpBlock(ngf * 2, ngf, ngf, scale=2)
        self.fusion_up3 = FusionUpBlock(ngf, ngf, ngf, scale=4)
        self.fusion_up4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf, ngf, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            UpResBlock(ngf),
            UpResBlock(ngf),
            SpectralNorm(nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1)),
            nn.Sigmoid()
        )

        if fixed_init:
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
            # for _, param in self.E_lq.named_parameters():
            #     param.requires_grad = False
            # for _, param in self.E_hq.named_parameters():
            #     param.requires_grad = False
            for v in self.find_variables("KV"):
                for _, param in getattr(self, v).named_parameters():
                    param.requires_grad = False
            for v in self.find_variables("Attention"):
                for _, param in getattr(self, v).named_parameters():
                    param.requires_grad = False
            for v in self.find_variables("Mask"):
                for _, param in getattr(self, v).named_parameters():
                    param.requires_grad = False


    def find_variables(self, suffix):
        attributes = dir(self)
        variables = [attr for attr in attributes if attr.endswith(suffix)]
        return variables


    def readMem(self, k, v, q):
        sim = F.conv2d(q, k)
        score = F.softmax(sim / sqrt(sim.size(1)), dim=1)  # B * S * 1 * 1 6*128
        sb, sn, sw, sh = score.size()
        s_m = score.view(sb, -1).unsqueeze(1)  # 2*1*M
        vb, vn, vw, vh = v.size()
        v_in = v.view(vb, -1).repeat(sb, 1, 1)  # 2*M*(c*w*h)
        mem_out = torch.bmm(s_m, v_in).squeeze(1).view(sb, vn, vw, vh)
        max_inds = torch.argmax(score, dim=1).squeeze()
        return mem_out, max_inds

    def enhancer(self, fs_in):
        le_256_q = fs_in['le_256_q']
        re_256_q = fs_in['re_256_q']
        mo_256_q = fs_in['mo_256_q']

        le_128_q = fs_in['le_128_q']
        re_128_q = fs_in['re_128_q']
        mo_128_q = fs_in['mo_128_q']

        le_64_q = fs_in['le_64_q']
        re_64_q = fs_in['re_64_q']
        mo_64_q = fs_in['mo_64_q']

        ####for 256
        le_256_mem_g, le_256_inds = self.readMem(self.le_256_mem_key, self.le_256_mem_value, le_256_q)
        re_256_mem_g, re_256_inds = self.readMem(self.re_256_mem_key, self.re_256_mem_value, re_256_q)
        mo_256_mem_g, mo_256_inds = self.readMem(self.mo_256_mem_key, self.mo_256_mem_value, mo_256_q)

        le_128_mem_g, le_128_inds = self.readMem(self.le_128_mem_key, self.le_128_mem_value, le_128_q)
        re_128_mem_g, re_128_inds = self.readMem(self.re_128_mem_key, self.re_128_mem_value, re_128_q)
        mo_128_mem_g, mo_128_inds = self.readMem(self.mo_128_mem_key, self.mo_128_mem_value, mo_128_q)

        le_64_mem_g, le_64_inds = self.readMem(self.le_64_mem_key, self.le_64_mem_value, le_64_q)
        re_64_mem_g, re_64_inds = self.readMem(self.re_64_mem_key, self.re_64_mem_value, re_64_q)
        mo_64_mem_g, mo_64_inds = self.readMem(self.mo_64_mem_key, self.mo_64_mem_value, mo_64_q)

        le_256_mem = le_256_mem_g
        re_256_mem = re_256_mem_g
        mo_256_mem = mo_256_mem_g
        le_128_mem = le_128_mem_g
        re_128_mem = re_128_mem_g
        mo_128_mem = mo_128_mem_g
        le_64_mem = le_64_mem_g
        re_64_mem = re_64_mem_g
        mo_64_mem = mo_64_mem_g

        le_256_mem_norm = adaptive_instance_normalization_4D(le_256_mem, fs_in['le256'])
        re_256_mem_norm = adaptive_instance_normalization_4D(re_256_mem, fs_in['re256'])
        mo_256_mem_norm = adaptive_instance_normalization_4D(mo_256_mem, fs_in['mo256'])

        ####for 128
        le_128_mem_norm = adaptive_instance_normalization_4D(le_128_mem, fs_in['le128'])
        re_128_mem_norm = adaptive_instance_normalization_4D(re_128_mem, fs_in['re128'])
        mo_128_mem_norm = adaptive_instance_normalization_4D(mo_128_mem, fs_in['mo128'])

        ####for 64
        le_64_mem_norm = adaptive_instance_normalization_4D(le_64_mem, fs_in['le64'])
        re_64_mem_norm = adaptive_instance_normalization_4D(re_64_mem, fs_in['re64'])
        mo_64_mem_norm = adaptive_instance_normalization_4D(mo_64_mem, fs_in['mo64'])

        EnMem256 = {'LE256Norm': le_256_mem_norm, 'RE256Norm': re_256_mem_norm, 'MO256Norm': mo_256_mem_norm}
        EnMem128 = {'LE128Norm': le_128_mem_norm, 'RE128Norm': re_128_mem_norm, 'MO128Norm': mo_128_mem_norm}
        EnMem64 = {'LE64Norm': le_64_mem_norm, 'RE64Norm': re_64_mem_norm, 'MO64Norm': mo_64_mem_norm}
        Ind256 = {'LE': le_256_inds, 'RE': re_256_inds, 'MO': mo_256_inds}
        Ind128 = {'LE': le_128_inds, 'RE': re_128_inds, 'MO': mo_128_inds}
        Ind64 = {'LE': le_64_inds, 'RE': re_64_inds, 'MO': mo_64_inds}
        return EnMem256, EnMem128, EnMem64, Ind256, Ind128, Ind64

    def reconstruct(self, fs_in, locs, memstar):
        le_256_mem_norm, re_256_mem_norm, mo_256_mem_norm = memstar[0]['LE256Norm'], memstar[0]['RE256Norm'], \
                                                            memstar[0]['MO256Norm']
        le_128_mem_norm, re_128_mem_norm, mo_128_mem_norm = memstar[1]['LE128Norm'], memstar[1]['RE128Norm'], \
                                                            memstar[1]['MO128Norm']
        le_64_mem_norm, re_64_mem_norm, mo_64_mem_norm = memstar[2]['LE64Norm'], memstar[2]['RE64Norm'], memstar[2][
            'MO64Norm']

        le_256_final = self.LE_256_Attention(le_256_mem_norm - fs_in['le256']) * le_256_mem_norm + fs_in['le256']
        re_256_final = self.RE_256_Attention(re_256_mem_norm - fs_in['re256']) * re_256_mem_norm + fs_in['re256']
        mo_256_final = self.MO_256_Attention(mo_256_mem_norm - fs_in['mo256']) * mo_256_mem_norm + fs_in['mo256']

        le_128_final = self.LE_128_Attention(le_128_mem_norm - fs_in['le128']) * le_128_mem_norm + fs_in['le128']
        re_128_final = self.RE_128_Attention(re_128_mem_norm - fs_in['re128']) * re_128_mem_norm + fs_in['re128']
        mo_128_final = self.MO_128_Attention(mo_128_mem_norm - fs_in['mo128']) * mo_128_mem_norm + fs_in['mo128']

        le_64_final = self.LE_64_Attention(le_64_mem_norm - fs_in['le64']) * le_64_mem_norm + fs_in['le64']
        re_64_final = self.RE_64_Attention(re_64_mem_norm - fs_in['re64']) * re_64_mem_norm + fs_in['re64']
        mo_64_final = self.MO_64_Attention(mo_64_mem_norm - fs_in['mo64']) * mo_64_mem_norm + fs_in['mo64']

        le_location = locs[:, 0, :]
        re_location = locs[:, 1, :]
        mo_location = locs[:, 3, :]
        le_location = le_location.cpu().int().numpy()
        re_location = re_location.cpu().int().numpy()
        mo_location = mo_location.cpu().int().numpy()

        up_in_256 = fs_in['f256'].clone()  # * 0
        up_in_128 = fs_in['f128'].clone()  # * 0
        up_in_64 = fs_in['f64'].clone()  # * 0

        for i in range(fs_in['f256'].size(0)):
            up_in_256[i:i + 1, :, le_location[i, 1] // 2:le_location[i, 3] // 2,
            le_location[i, 0] // 2:le_location[i, 2] // 2] = F.interpolate(le_256_final[i:i + 1, :, :, :].clone(), (
                le_location[i, 3] // 2 - le_location[i, 1] // 2, le_location[i, 2] // 2 - le_location[i, 0] // 2),
                                                                           mode='bilinear', align_corners=False)
            up_in_256[i:i + 1, :, re_location[i, 1] // 2:re_location[i, 3] // 2,
            re_location[i, 0] // 2:re_location[i, 2] // 2] = F.interpolate(re_256_final[i:i + 1, :, :, :].clone(), (
                re_location[i, 3] // 2 - re_location[i, 1] // 2, re_location[i, 2] // 2 - re_location[i, 0] // 2),
                                                                           mode='bilinear', align_corners=False)
            up_in_256[i:i + 1, :, mo_location[i, 1] // 2:mo_location[i, 3] // 2,
            mo_location[i, 0] // 2:mo_location[i, 2] // 2] = F.interpolate(mo_256_final[i:i + 1, :, :, :].clone(), (
                mo_location[i, 3] // 2 - mo_location[i, 1] // 2, mo_location[i, 2] // 2 - mo_location[i, 0] // 2),
                                                                           mode='bilinear', align_corners=False)

            up_in_128[i:i + 1, :, le_location[i, 1] // 4:le_location[i, 3] // 4,
            le_location[i, 0] // 4:le_location[i, 2] // 4] = F.interpolate(le_128_final[i:i + 1, :, :, :].clone(), (
                le_location[i, 3] // 4 - le_location[i, 1] // 4, le_location[i, 2] // 4 - le_location[i, 0] // 4),
                                                                           mode='bilinear', align_corners=False)
            up_in_128[i:i + 1, :, re_location[i, 1] // 4:re_location[i, 3] // 4,
            re_location[i, 0] // 4:re_location[i, 2] // 4] = F.interpolate(re_128_final[i:i + 1, :, :, :].clone(), (
                re_location[i, 3] // 4 - re_location[i, 1] // 4, re_location[i, 2] // 4 - re_location[i, 0] // 4),
                                                                           mode='bilinear', align_corners=False)
            up_in_128[i:i + 1, :, mo_location[i, 1] // 4:mo_location[i, 3] // 4,
            mo_location[i, 0] // 4:mo_location[i, 2] // 4] = F.interpolate(mo_128_final[i:i + 1, :, :, :].clone(), (
                mo_location[i, 3] // 4 - mo_location[i, 1] // 4, mo_location[i, 2] // 4 - mo_location[i, 0] // 4),
                                                                           mode='bilinear', align_corners=False)

            up_in_64[i:i + 1, :, le_location[i, 1] // 8:le_location[i, 3] // 8,
            le_location[i, 0] // 8:le_location[i, 2] // 8] = F.interpolate(le_64_final[i:i + 1, :, :, :].clone(), (
                le_location[i, 3] // 8 - le_location[i, 1] // 8, le_location[i, 2] // 8 - le_location[i, 0] // 8),
                                                                           mode='bilinear', align_corners=False)
            up_in_64[i:i + 1, :, re_location[i, 1] // 8:re_location[i, 3] // 8,
            re_location[i, 0] // 8:re_location[i, 2] // 8] = F.interpolate(re_64_final[i:i + 1, :, :, :].clone(), (
                re_location[i, 3] // 8 - re_location[i, 1] // 8, re_location[i, 2] // 8 - re_location[i, 0] // 8),
                                                                           mode='bilinear', align_corners=False)
            up_in_64[i:i + 1, :, mo_location[i, 1] // 8:mo_location[i, 3] // 8,
            mo_location[i, 0] // 8:mo_location[i, 2] // 8] = F.interpolate(mo_64_final[i:i + 1, :, :, :].clone(), (
                mo_location[i, 3] // 8 - mo_location[i, 1] // 8, mo_location[i, 2] // 8 - mo_location[i, 0] // 8),
                                                                           mode='bilinear', align_corners=False)

        ms_in_64 = self.MSDilate(fs_in['f64'].clone())

        h = self.deg_aware(fs_in['f256'].clone(), fs_in['f128'].clone(), ms_in_64)
        quant, emb_loss, info = self.quantize(h)
        quant = self.post_quant(quant)

        fea_up1 = self.fusion_up1(quant, up_in_64)
        fea_up2 = self.fusion_up2(fea_up1, up_in_128, quant)
        fea_up3 = self.fusion_up3(fea_up2, up_in_256, quant)
        output = self.fusion_up4(fea_up3)  #
        return output

    def forward(self, lq, loc):
        fs_in = self.E_lq(lq, loc)  # low quality images
        MemNorm256, MemNorm128, MemNorm64, Ind256, Ind128, Ind64 = self.enhancer(fs_in)
        Out = self.reconstruct(fs_in, loc, memstar=[MemNorm256, MemNorm128, MemNorm64])
        return Out


class UpResBlock(nn.Module):
    def __init__(self, dim, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d):
        super(UpResBlock, self).__init__()
        self.Model = nn.Sequential(
            SpectralNorm(conv_layer(dim, dim, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(conv_layer(dim, dim, 3, 1, 1)),
        )

    def forward(self, x):
        out = x + self.Model(x)
        return out


if __name__ == '__main__':
    file = "/root/yanwd/dataset/FFHQ_512_locations/00012.txt"
    model = LNLFaceNet()
    ipt = torch.randn(1, 3, 512, 512)
    loc = torch.tensor(np.loadtxt(file)).unsqueeze(0)
    out = model(ipt, loc)
    print(out.shape)





    # def check_frozen_layers(model):
    #     frozen_layers = []
    #     for name, param in model.named_parameters():
    #         if not param.requires_grad:
    #             layer_name = name.split('.')[0]
    #             frozen_layers.append(layer_name)
    #     return frozen_layers


    # frozen_layers = check_frozen_layers(model)


    '''
    init_weights from per-train weights
    '''
    # param_name = [name for name, _ in model.named_parameters()]
    #
    # weights_DMD = torch.load('/root/yanwd/projects/LNLFace/DMDNet.pth')
    # weights_RES = torch.load('/root/yanwd/projects/LNLFace/RestoreFormer++.ckpt')
    #
    # for name, param in weights_DMD.items():
    #     if name in param_name:
    #         print(name)
    #         model.state_dict()[name].copy_(param)
    #
    # for name, buffer in weights_DMD.items():
    #     if name in model.state_dict() and '_mem_' in name:
    #         print(name)
    #         model.state_dict()[name].copy_(buffer)
    #
    # model.state_dict()["quantize.embedding.weight"].copy_(weights_RES["state_dict"]["vqvae.quantize.embedding.weight"])
    #
    # torch.save(model.state_dict(), 'init_weights.pth')
    #
    # my_weights = torch.load('./init_weights.pth')



