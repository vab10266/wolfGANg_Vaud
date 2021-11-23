import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.autograd import Variable
import math

import numpy as np


def initialize_weights(layer, mean=0.0, std=0.02):
    if isinstance(layer, (nn.Conv3d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight, mean, std)
    elif isinstance(layer, (nn.Linear, nn.BatchNorm2d)):
        nn.init.normal_(layer.weight, mean, std)
        nn.init.constant_(layer.bias, 0)

class Normal(nn.Module):
    """Samples from a Normal distribution using the reparameterization trick.
    """

    def __init__(self, mu=0, sigma=1):
        super(Normal, self).__init__()
        self.normalization = Variable(torch.Tensor([np.log(2 * np.pi)]))

        self.mu = Variable(torch.Tensor([mu]))
        self.logsigma = Variable(torch.Tensor([math.log(sigma)]))

    def _check_inputs(self, size, mu_logsigma):
        if size is None and mu_logsigma is None:
            raise ValueError(
                'Either one of size or params should be provided.')
        elif size is not None and mu_logsigma is not None:
            mu = mu_logsigma.select(-1, 0).expand(size)
            logsigma = mu_logsigma.select(-1, 1).expand(size)
            return mu, logsigma
        elif size is not None:
            mu = self.mu.expand(size)
            logsigma = self.logsigma.expand(size)
            return mu, logsigma
        elif mu_logsigma is not None:
            mu = mu_logsigma.select(-1, 0)
            logsigma = mu_logsigma.select(-1, 1)
            return mu, logsigma
        else:
            raise ValueError(
                'Given invalid inputs: size={}, mu_logsigma={})'.format(
                    size, mu_logsigma))

    def sample(self, size=None, params=None):
        mu, logsigma = self._check_inputs(size, params)
        std_z = Variable(torch.randn(mu.size()).type_as(mu.data))
        sample = std_z * torch.exp(logsigma) + mu
        return sample

    def log_density(self, sample, params=None):
        if params is not None:
            mu, logsigma = self._check_inputs(None, params)
        else:
            mu, logsigma = self._check_inputs(sample.size(), None)
            mu = mu.type_as(sample)
            logsigma = logsigma.type_as(sample)

        c = self.normalization.type_as(sample.data)
        inv_sigma = torch.exp(-logsigma)
        tmp = (sample - mu) * inv_sigma
        return -0.5 * (tmp * tmp + 2 * logsigma + c)

    def NLL(self, params, sample_params=None):
        """Analytically computes
            E_N(mu_2,sigma_2^2) [ - log N(mu_1, sigma_1^2) ]
        If mu_2, and sigma_2^2 are not provided, defaults to entropy.
        """
        mu, logsigma = self._check_inputs(None, params)
        if sample_params is not None:
            sample_mu, sample_logsigma = self._check_inputs(None, sample_params)
        else:
            sample_mu, sample_logsigma = mu, logsigma

        c = self.normalization.type_as(sample_mu.data)
        nll = logsigma.mul(-2).exp() * (sample_mu - mu).pow(2) \
            + torch.exp(sample_logsigma.mul(2) - logsigma.mul(2)) + 2 * logsigma + c
        return nll.mul(0.5)

    def kld(self, params):
        """Computes KL(q||p) where q is the given distribution and p
        is the standard Normal distribution.
        """
        mu, logsigma = self._check_inputs(None, params)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mean^2 - sigma^2)
        kld = logsigma.mul(2).add(1) - mu.pow(2) - logsigma.exp().pow(2)
        kld.mul_(-0.5)
        return kld

    def get_params(self):
        return torch.cat([self.mu, self.logsigma])

    @property
    def nparams(self):
        return 2

    @property
    def ndim(self):
        return 1

    @property
    def is_reparameterizable(self):
        return True

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' ({:.3f}, {:.3f})'.format(
            self.mu.data[0], self.logsigma.exp().data[0])
        return tmpstr

class Reshape(nn.Module):
    def __init__(self, shape=[32, 1, 1]):
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        batch_size = x.size(0)
        # print("reshape: ", *x.shape, "\t to ", x.shape[0], *self.shape)
        return x.view(batch_size, *self.shape)


class UnFlatten(nn.Module):
    def __init__(self, block_size):
        super(UnFlatten, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        # print("unflat")
        # print(x.shape)
        # print(x.view(x.size(0), -1, self.block_size, self.block_size).shape)
        return x.view(x.size(0), -1, self.block_size, self.block_size)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def cos_similarity(weight):
    weight = weight / weight.norm(dim=-1).unsqueeze(-1)
    cos_distance = torch.mm(weight, weight.transpose(1,0))
    return cos_distance.pow(2).mean()

class OrthorConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride=1, padding=0, bias=True, groups=1):
        super(OrthorConv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.groups = groups
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=bias, groups=groups)
        self.opt_orth = optim.Adam(self.parameters(), lr=1e-5, betas=(0.5, 0.99))
        self.out_channel = out_channel
        self.in_channel = in_channel

    def orthogonal_update(self):
        self.zero_grad()
        loss = cos_similarity(self.conv.weight.view(self.in_channel*self.out_channel//self.groups, -1))
        loss.backward()
        self.opt_orth.step()

    def forward(self, feat):
        # print("orthoconv: ", feat.shape)
        if self.training:
            self.orthogonal_update()
        return self.conv(feat)

class OrthorTransform(nn.Module):
    def __init__(self, c_dim, feat_hw):
        super(OrthorTransform, self).__init__()

        self.c_dim = c_dim
        self.feat_hw = feat_hw
        self.weight = nn.Parameter(torch.randn(1, c_dim, 5, 1)) #feat_hw, feat_hw))
        self.opt_orth = optim.Adam(self.parameters(), lr=1e-4, betas=(0.5, 0.99))

    def orthogonal_update(self):
        self.zero_grad()
        loss = cos_similarity(self.weight.view( self.c_dim, -1))
        loss.backward()
        self.opt_orth.step()

    def forward(self, feat):
        if self.training:
            self.orthogonal_update()
        pred = feat * self.weight.expand_as(feat)
        return pred.mean(-1).mean(-1)


# Q module that utilzes the orthogonal regularized conv and transformer layers
class CodeReduction(nn.Module):
    def __init__(self, c_dim, feat_c, feat_hw, prob=True):
        super(CodeReduction, self).__init__()
        if prob:
            c_dim *= 2
        self.c_dim = c_dim        
        self.feat_c, self.feat_hw, self.prob = feat_c, feat_hw, prob
        self.main = nn.Sequential(
            nn.Conv2d(feat_c, c_dim, 3, 1, 1, bias=True, groups=1),
            nn.LeakyReLU(0.1),
            OrthorConv2d(c_dim, c_dim, 4, 2, 1, bias=True, groups=c_dim)
        )

        self.trans = OrthorTransform(c_dim=c_dim, feat_hw=feat_hw//2)
    
    def forward(self, feat):
        # print(self.c_dim, self.feat_c, self.feat_hw, self.prob)
        # input shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar//4, n_pitches//12)
        # print("Reduction: ", feat.shape)
        main = self.main(feat)
        # print("main: ", main.shape)
        pred_c = self.trans( main )
        # print("pred_c: ", pred_c.shape)
        return pred_c.view(feat.size(0), self.c_dim)


class ChannelAttentionMask(nn.Module):
    def __init__(self, c_dim, feat_c, feat_hw):
        super().__init__()
        self.feat_c = feat_c
        self.feat_hw = feat_hw

        self.instance_attention = nn.Parameter(torch.randn(1,feat_c,feat_hw*feat_hw))
        self.channel_attention = nn.Sequential(
            nn.Linear(c_dim, feat_c), nn.ReLU(), nn.Linear(feat_c, feat_c), UnFlatten(1)
        )
    def forward(self, c):
        #instance_mask = torch.softmax(self.instance_attention, dim=-1).view(1, self.feat_c, self.feat_hw, self.feat_hw)
        channel_mask = self.channel_attention(c)
        return self.feat_c*channel_mask


class Upscale2d(nn.Module):
    def __init__(self, factor):
        super(Upscale2d, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor

    def forward(self, x):
        if self.factor == 1:
            return x
        s = x.size()
        x = x.view(-1, s[1], s[2], 1, s[3], 1)
        x = x.expand(-1, s[1], s[2], self.factor, s[3], self.factor)
        x = x.contiguous().view(-1, s[1], s[2] * self.factor, s[3] * self.factor)
        return x



class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.main = nn.Sequential(
                        Upscale2d(factor=2),
                        nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True),
                        nn.BatchNorm2d(out_channel),
                        nn.LeakyReLU(0.1),
                    )
    
    def forward(self, x):
        return self.main(x)


class DownConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel ):
        super().__init__()

        self.main = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True),
                nn.BatchNorm2d(out_channel), 
                nn.AvgPool2d(2, 2),
                nn.LeakyReLU(0.2),)

    def forward(self, x):
        return self.main(x)

 
class OOGANInput(nn.Module):
    """
    The OOGAN input module, the compteting-free input
    ...

    Attributes
    ----------
    c_dim : int
        number of dimensions in control vector c
    z_dim : int
        number of dimensions in noise vector z
    feat_4 : int
        feature's channel dimension at 4x4 level
    feat_8 : int
        feature's channel dimension at 8x8 level

    Methods
    -------
    forward(c=None, z=None): Tensor
        returns the feature map
    """
    def __init__(self, c_dim, z_dim, feat_4, feat_8):
        super().__init__()

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.feat_4 = feat_4
        self.feat_8 = feat_8

        self.init_noise = nn.Parameter(torch.randn(1, feat_4, 4, 4))
        
        self.from_c_4 = nn.Sequential(
            UnFlatten(1), 
            nn.ConvTranspose2d(c_dim, feat_4, 4, 1, 0, bias=True),
            nn.BatchNorm2d(feat_4), 
            nn.LeakyReLU(0.01))

        self.from_c_8 = UpConvBlock(feat_4, feat_8)
        
        self.z_dim = z_dim
        if z_dim > 0:
            self.attn_from_c = ChannelAttentionMask(c_dim, feat_8, 8)
            self.from_z_8 = nn.Sequential(
                UnFlatten(1), 
                nn.ConvTranspose2d(z_dim, z_dim//2, 4, 1, 0, bias=True),
                nn.BatchNorm2d(z_dim//2), 
                nn.LeakyReLU(0.01),
                UpConvBlock(z_dim//2, feat_8)
                )

    def forward(self, c, z=None):
        #feat = self.init_noise.expand(c.size(0), -1, -1, -1)
        # print("in: ", 
        # self.c_dim,
        # self.z_dim,
        # self.feat_4,
        # self.feat_8)

        feat = self.from_c_4(c) #+ feat
        feat = self.from_c_8(feat)
        if self.z_dim>0 and z is not None:
            attn_from_c = self.attn_from_c(c)
            feat = attn_from_c*self.from_z_8(z) + feat
        return feat
