import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from utils import init_weights, get_padding
from hparams import Hparams



class ResBlock1(torch.nn.Module):
    
    def __init__(self, hp : Hparams, channels, kernel_size = 3, dilation = (1, 3, 5)):
        
        super(ResBlock1, self).__init__()
        self.hp = hp
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation = dilation[0],
                               padding = get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation = dilation[1],
                               padding = get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation = dilation[2],
                               padding = get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)
        
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation = 1,
                               padding = get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation = 1,
                               padding = get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation = 1,
                               padding = get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        
        
    def forward(self, x):
        
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.hp.lrelu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.hp.lrelu_slope)
            xt = c2(xt)
            x = xt + x
        
        return x
    
    
    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)
        
        
        
        
class ResBlock2(torch.nn.Module):
    
    def __init__(self):
        super(ResBlock2, self).__init__()



class Generator(torch.nn.Module):
    
    def __init__(self, hp : Hparams):
        super(Generator, self).__init__()
        self.hp = hp
        self.num_kernels = len(hp.resblock_kernel_sizes)
        self.num_upsamples = len(hp.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(hp.n_mels, hp.upsample_initial_channel, 7, 1, padding = 3))
        resblock = ResBlock1 if hp.resblock == '1' else ResBlock2
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(hp.upsample_rates, hp.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(
                    hp.upsample_initial_channel // (2 ** i),
                    hp.upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding = (k - u) // 2
                )
            ))
            
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hp.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(hp.resblock_kernel_sizes, hp.resblock_dilation_sizes)):
                self.resblocks.append(resblock(hp, ch, k, d))
                
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding = 3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        
        
    def forward(self, x):
        
        # [B, M, T] -> [B, M1, T]
        x = self.conv_pre(x)
        
        # 再每个resblock中维度不会遍，resblock是为了收集全局信息
        # [B, 512, 32] -> 
        # [B, 256, 256] -> 3 * ResBlock -> [B, 256, 256] ->
        # [B, 128, 2048] -> 3 * ResBlock -> [B, 256, 2048] ->
        # [B, 64, 4096] -> 3 * ResBlock -> [B, 64, 4096] ->
        # [B, 32, 8192] -> 3 * ResBlock -> [B, 32, 8192] ->
        # [B, 1, 8192] 
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.hp.lrelu_slope)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
        
        
    def remove_weight_norm(self):
        print('Removing weight norm ...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        
        
        
class DiscriminatorS(torch.nn.Module):
    
    def __init__(self, hp : Hparams, use_spectral_norm = False):
        super(DiscriminatorS, self).__init__()
        self.hp = hp
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding = 7)),
            norm_f(Conv1d(128, 128, 41, 2, groups = 4, padding = 20)),
            norm_f(Conv1d(128, 256, 41, 2, groups = 16, padding = 20)),
            norm_f(Conv1d(256, 512, 41, 4, groups = 16, padding = 20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups = 16, padding = 20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups = 16, padding = 20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding = 2))
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding = 1))
        
    def forward(self, x):
        
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.hp.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap
        
        
class MultiScaleDiscriminator(torch.nn.Module):
    
    def __init__(self, hp : Hparams):
        super(MultiScaleDiscriminator, self).__init__()
        self.hp = hp
        self.discriminators = nn.ModuleList([
            DiscriminatorS(hp, use_spectral_norm = True),
            DiscriminatorS(hp),
            DiscriminatorS(hp)
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding = 2),
            AvgPool1d(4, 2, padding = 2)
        ])
        
    def forward(self, y, y_hat):
        
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [[] for i in range(4)]
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y  = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
            
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    
    
    
class DiscriminatorP(torch.nn.Module):
    
    def __init__(self, hp : Hparams, period, use_spectral_norm = False):
        super(DiscriminatorP, self).__init__()
        self.hp = hp
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (hp.DP_kernel_size, 1), (hp.DP_stride, 1), padding = (get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (hp.DP_kernel_size, 1), (hp.DP_stride, 1), padding = (get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (hp.DP_kernel_size, 1), (hp.DP_stride, 1), padding = (get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (hp.DP_kernel_size, 1), (hp.DP_stride, 1), padding = (get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (hp.DP_kernel_size, 1), 1, padding = (2, 0)))
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding = (1, 0)))
     
   
    def forward(self, x):
        
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), 'reflect')
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)
        
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.hp.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap
    
    
class MultiPeriodDiscriminator(torch.nn.Module):
    
    def __init__(self, hp : Hparams):
        super(MultiPeriodDiscriminator, self).__init__()
        self.hp = hp
        self.discriminators = nn.ModuleList([
            DiscriminatorP(hp, 2),
            DiscriminatorP(hp, 3),
            DiscriminatorP(hp, 5),
            DiscriminatorP(hp, 7),
            DiscriminatorP(hp, 11)
        ])
        
    def forward(self, y, y_hat):
        
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [[] for i in range(4)]
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    
    

def feature_loss(fmap_r, fmap_g):
    
    loss = 0.
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))    
    
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    
    loss = 0.
    r_losses, g_losses = [], []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
        
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    
    loss = 0.
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += 1
        
    return loss, gen_losses

        
            
if __name__ == '__main__':
    inp = torch.randn(1, 80, 32)
    hp = Hparams()
    g = Generator(hp)
    g_out = g(inp)
    print(g_out.shape)
    #d = MultiScaleDiscriminator(hp)
    #d = DiscriminatorS(hp)
    #a, b, c, d = d(g_out, g_out)
    #print(a[1].shape)
    d = MultiPeriodDiscriminator(hp)
    a, b, c, d = d(g_out, g_out)
    print(a[0].shape, a[1].shape, a[2].shape, a[3].shape, a[4].shape)
    
    
            