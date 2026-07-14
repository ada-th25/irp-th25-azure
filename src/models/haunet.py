import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def make_model(goes_channels=3, cov_channels=2, downscale_factor=4,
               cov_refine_channels=32, cov_window_size=8):
    """
    Factory function for the wildfire-adapted HAUNet.
 
    Args:
        goes_channels: number of GOES input bands (default 3: C07, C14, TFI)
        cov_channels:  number of covariate bands (default 2: DEM, land cover)
        downscale_factor: internal resolution reduction applied before the
            main encoder/attention stack (must be a power of 2).
        cov_refine_channels: channel width of the auxiliary full-resolution
            covariate pyramid (kept narrow, this branch is convolution +
            windowed-attention only, not the main cost driver).
        cov_window_size: window size (pixels) for windowed spatial
            self-attention applied to covariate features at each pyramid
            scale. Cost scales with window_size^4, not full resolution^4.
 
    Returns:
        HAUNet instance configured for 256x256 -> 256x256 fire mask prediction.
    """
    return HAUNet(
        goes_channels=goes_channels,
        cov_channels=cov_channels,
        width=96,
        enc_blk_nums=[5, 5],
        dec_blk_nums=[5, 5],
        middle_blk_num=10,
        downscale_factor=downscale_factor,
        cov_refine_channels=cov_refine_channels,
        cov_window_size=cov_window_size,
    )

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)

class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class Reconstruct(nn.Module):
    def __init__(self, scale_factor):
        super(Reconstruct, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        if self.scale_factor!=1:
            x = nn.Upsample(scale_factor=self.scale_factor)(x)
        return x

class qkvblock(nn.Module):
    def __init__(self, c, num_heads=2, FFN_Expand=2):
        super().__init__()
        self.num_heads = num_heads
        self.kv = nn.Conv2d(c * 3, c * 6, kernel_size=1)
        self.kv_dwconv = nn.Conv2d(c * 6, c * 6, 3, padding=1, groups=c * 6)

        self.q = nn.Conv2d(c, c, kernel_size=1)
        self.q_dwconv = nn.Conv2d(c, c, 3, padding=1, groups=c)

        self.q1 = nn.Conv2d(c, c, kernel_size=1)
        self.q1_dwconv = nn.Conv2d(c, c, 3, padding=1, groups=c)

        self.q2 = nn.Conv2d(c, c, kernel_size=1)
        self.q2_dwconv = nn.Conv2d(c, c, 3, padding=1, groups=c)

        self.project_out = nn.Conv2d(c, c, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out1 = nn.Conv2d(c, c, kernel_size=1)
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out2 = nn.Conv2d(c, c, kernel_size=1)
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.conv4_1 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        self.conv5_1 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.conv4_2 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=True)

        self.conv5_2 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                                 groups=1, bias=True)

        self.normq = LayerNorm2d(c)
        self.normq1 = LayerNorm2d(c)
        self.normq2 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.norm2_1 = LayerNorm2d(c)
        self.norm2_2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta1 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma1 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta2 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.relu=nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encs):
        enc0 = encs[0]
        enc1= nn.Upsample(scale_factor=2)(encs[1])
        enc2= nn.Upsample(scale_factor=4)(encs[2])
        q = self.normq(enc0)
        q1 = self.normq1(enc1)
        q2 = self.normq2(enc2)

        kv_attn = torch.cat((q, q1, q2), dim=1)
        kv = self.kv_dwconv(self.kv(kv_attn))
        k, v = kv.chunk(2, dim=1)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = torch.nn.functional.normalize(k, dim=-1)


        q = self.q_dwconv(self.q(q))
        b, c_q, h, w = q.shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn=self.relu(attn)
        attn = self.softmax(attn)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        x = self.project_out(out)
        y = enc0 + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        out0 = y + x * self.gamma


        q1 = self.q1_dwconv(self.q1(q1))
        b, c_q, h, w = q1.shape
        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        attn1 = (q1 @ k.transpose(-2, -1)) * self.temperature1
        attn1 = self.relu(attn1)
        # attn1 = attn1.softmax(dim=-1)
        attn1 = self.softmax(attn1)
        out1 = (attn1 @ v)
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        x1 = self.project_out1(out1)
        y1 = enc1 + x1 * self.beta1
        x1 = self.conv4_1(self.norm2_1(y1))
        x1 = self.sg(x1)
        x1 = self.conv5_1(x1)
        out1 = y1 + x1 * self.gamma1


        q2 = self.q2_dwconv(self.q2(q2))
        b, c_q, h, w = q2.shape
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        attn2 = (q2 @ k.transpose(-2, -1)) * self.temperature2
        attn2 = self.relu(attn2)
        attn2 = self.softmax(attn2)
        out2 = (attn2 @ v)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        x2 = self.project_out1(out2)
        y2 = enc2 + x2 * self.beta2
        x2 = self.conv4_2(self.norm2_2(y2))
        x2 = self.sg(x2)
        x2 = self.conv5_2(x2)
        out2 = y2 + x2 * self.gamma2
        out1 = nn.Upsample(scale_factor=0.5)(out1)
        out2 = nn.Upsample(scale_factor=0.25)(out2)
        outs = []
        outs.append(out0)
        outs.append(out1)
        outs.append(out2)
        return outs

class lateral_nafblock(nn.Module):
    def __init__(self, c,num_heads=3,num_block=1):
        super().__init__()
        self.num_heads=num_heads
        self.qkv=nn.Sequential(
                    *[qkvblock(c) for _ in range(num_block)]
                )
    def forward(self, encs):
        outs=encs
        for qkv in self.qkv:
            outs=qkv(outs)
        return outs

class S_CEMBlock(nn.Module):
    def __init__(self, c, DW_Expand=2,num_heads=3, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(c, c * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(c * 3, c * 3, kernel_size=3, stride=1, padding=1, groups=c * 3)

        self.project_out = nn.Conv2d(c, c, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out2 = nn.Conv2d(c, c, kernel_size=1)
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta2 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.relu=nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        qs = q.clone().permute(0, 1, 3, 2)
        ks = k.clone().permute(0, 1, 3, 2)
        vs = v.clone().permute(0, 1, 3, 2)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn=self.relu(attn)
        attn = self.softmax(attn)

        outc = (attn @ v)

        qs = torch.nn.functional.normalize(qs, dim=-1)
        ks = torch.nn.functional.normalize(ks, dim=-1)

        attns = (qs @ ks.transpose(-2, -1)) * self.temperature2
        attns=self.relu(attns)
        attns = self.softmax(attns)
        outs = (attns @ vs)
        outs = outs.permute(0, 1, 3, 2)

        outc = rearrange(outc, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        outs = rearrange(outs, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        xc = self.project_out(outc)
        xc = self.dropout1(xc)
        xs = self.project_out2(outs)
        xs = self.dropout1(xs)

        y = inp + xc * self.beta+ xs * self.beta2

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class CEMBlock(nn.Module):
    def __init__(self, c, DW_Expand=2,num_heads=3, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(c, c * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3)
        self.project_out = nn.Conv2d(c, c, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.ReLU()
        self.softmax=nn.Softmax(dim=-1)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.relu(attn)
        attn=self.softmax(attn)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        x = self.project_out(out)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class WindowedSCEMBlock(nn.Module):
    def __init__(self, c, window_size=8, num_heads=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.window_size = window_size
        self.block = S_CEMBlock(c, num_heads=num_heads, FFN_Expand=FFN_Expand,
                                 drop_out_rate=drop_out_rate)
 
    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
 
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x_p = F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else x
        Hp, Wp = x_p.shape[-2:]
 
        # Partition into non-overlapping windows:
        # (B, C, Hp, Wp) -> (B * nH * nW, C, ws, ws)
        x_p = x_p.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x_p = x_p.permute(0, 2, 4, 1, 3, 5).contiguous()
        x_p = x_p.view(-1, C, ws, ws)
 
        x_p = self.block(x_p)   # full channel+spatial attention, but only within each window
 
        # Reassemble windows back into the full feature map.
        x_p = x_p.view(B, Hp // ws, Wp // ws, C, ws, ws)
        x_p = x_p.permute(0, 3, 1, 4, 2, 5).contiguous()
        x_p = x_p.view(B, C, Hp, Wp)
 
        return x_p[:, :, :H, :W]
    
class CovariatePyramidEncoder(nn.Module):
    def __init__(self, cov_channels, refine_channels, n_stages, window_size=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(cov_channels, refine_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            WindowedSCEMBlock(refine_channels, window_size=window_size),
        )
        self.down_stages = nn.ModuleList()
        for _ in range(n_stages):
            self.down_stages.append(nn.Sequential(
                nn.Conv2d(refine_channels, refine_channels, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                WindowedSCEMBlock(refine_channels, window_size=window_size),
            ))
 
    def forward(self, cov):
        """Returns a list of features [full_res, H/2, H/4, ..., H/2^n_stages]."""
        feats = [self.stem(cov)]
        c = feats[0]
        for stage in self.down_stages:
            c = stage(c)
            feats.append(c)
        return feats

# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class HAUNet(nn.Module):
    """
    Wildfire-adapted HAUNet.
 
    Input  : (B, goes_channels + cov_channels, 256, 256)
             GOES bands must be bilinearly upsampled to 256x256 BEFORE
             being passed in. Covariates are already at 30m / 256x256.
    Output : (B, 1, 256, 256)  —  raw fire mask logit (no sigmoid).
             Apply torch.sigmoid() for inference.
    """
 
    def __init__(
        self,
        goes_channels = 3,
        cov_channels = 2,
        width = 96,
        middle_blk_num = 10,
        enc_blk_nums = [5, 5],
        dec_blk_nums = [5, 5],
        heads = [1, 2, 4],
        downscale_factor = 4,
        cov_refine_channels = 32,
        cov_window_size = 8,
    ):
        super().__init__()

        assert downscale_factor >= 1 and (downscale_factor & (downscale_factor - 1)) == 0, \
            "downscale_factor must be a power of 2 (1, 2, 4, 8, ...)"
 
        self.goes_channels = goes_channels
        self.cov_channels = cov_channels
        self.downscale_factor = downscale_factor
        self.n_refine_stages = int(math.log2(downscale_factor)) if downscale_factor > 1 else 0
 
        in_channels = goes_channels + cov_channels
 
        self.intro = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
 
        chan = width
        ii   = 0
        for numii, num in enumerate(enc_blk_nums):
            if numii < 1:
                self.encoders.append(nn.Sequential(
                    *[S_CEMBlock(chan, num_heads=heads[ii]) for _ in range(num)]
                ))
            else:
                self.encoders.append(nn.Sequential(
                    *[CEMBlock(chan, num_heads=heads[ii]) for _ in range(num)]
                ))
            self.downs.append(nn.Conv2d(chan, chan, 2, 2))
            ii += 1
 
        self.lateral_nafblock = lateral_nafblock(chan)
        self.enc_middle_blks  = nn.Sequential(
            *[CEMBlock(chan, num_heads=heads[ii]) for _ in range(middle_blk_num // 2)]
        )
        self.dec_middle_blks  = nn.Sequential(
            *[CEMBlock(chan, num_heads=heads[ii]) for _ in range(middle_blk_num // 2)]
        )
 
        ii = 0
        for numii, num in enumerate(dec_blk_nums):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(chan, chan, kernel_size=2, stride=2)
            ))
            if numii < 1:
                self.decoders.append(nn.Sequential(
                    *[CEMBlock(chan, num_heads=heads[1 - ii]) for _ in range(num)]
                ))
            else:
                self.decoders.append(nn.Sequential(
                    *[S_CEMBlock(chan, num_heads=heads[1 - ii]) for _ in range(num)]
                ))
            ii += 1
 
        self.dec_blk_nums = dec_blk_nums
        self.padder_size  = 2 ** len(self.encoders)

        self.cov_pyramid = CovariatePyramidEncoder(
            cov_channels=cov_channels,
            refine_channels=cov_refine_channels,
            n_stages=max(0, self.n_refine_stages - 1),#n_stages=self.n_refine_stages,
            window_size=cov_window_size,
        )

        self.refine_ups = nn.ModuleList([
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2)
            for _ in range(self.n_refine_stages)
        ])
        self.refine_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width + cov_refine_channels, width, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.n_refine_stages)
        ])

        self.output_head = nn.Conv2d(
            in_channels=width,
            out_channels=1,
            kernel_size=3, padding=1, stride=1, bias=True
        )

        PRIOR_FIRE_FRACTION = 0.0085
        prior_logit = math.log(PRIOR_FIRE_FRACTION / (1 - PRIOR_FIRE_FRACTION))
        nn.init.constant_(self.output_head.bias, prior_logit)
 
    def forward(self, inp):
        """
        Args:
            inp: (B, goes_channels + cov_channels, H, W)
                 GOES bands bilinearly upsampled to (H, W) = (256, 256).
                 Covariates at native 30m resolution, same spatial extent.
        Returns:
            (B, 1, H, W) fire mask logit: apply sigmoid for probabilities.
        """

        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)   # pad up to a multiple of padder_size (4)

        cov = inp[:, self.goes_channels:, :, :]
 
        low_h = max(1, H // self.downscale_factor)
        low_w = max(1, W // self.downscale_factor)
        inp_low = F.interpolate(inp, size=(low_h, low_w), mode='bilinear', align_corners=False)
        inp_low = self.check_image_size(inp_low)   # pad up to a multiple of padder_size
 
        x = self.intro(inp_low)
 
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
 
        x = self.enc_middle_blks(x)
        encs.append(x)

        outs = self.lateral_nafblock(encs)
        x = outs[-1]
        x = self.dec_middle_blks(x)

        outs2 = outs[:2]
        for decoder, up, enc_skip in zip(self.decoders, self.ups, outs2[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
 
        x = x[:, :, :low_h, :low_w]   # crop back to unpadded low-res size

        # feats[0] = full res (H, W); feats[-1] = low-res (~low_h, low_w)
        cov_feats = self.cov_pyramid(cov)

        # Progressive coarse-to-fine refinement, fusing covariates at each scale
        for i in range(self.n_refine_stages):
            x = self.refine_ups[i](x)                              # upsample by 2x
            cov_feat = cov_feats[self.n_refine_stages - 1 - i]      # matching scale, coarse->fine
            if x.shape[-2:] != cov_feat.shape[-2:]:
                x = F.interpolate(x, size=cov_feat.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, cov_feat], dim=1)
            x = self.refine_fuse[i](x)
 
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
 
        x = self.output_head(x)   # (B, 1, H, W)

        return x
 
    def check_image_size(self, x):
        """Pad input to be divisible by padder_size (= 2^num_encoder_stages)."""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)

if __name__ == '__main__':
    import time
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
 
    net = make_model(goes_channels=3, cov_channels=2).to(device)
    count_parameters(net)
 
    # Simulate a batch of 2 samples at the wildfire input resolution
    # GOES (3 ch) + covariates (2 ch), all at 256×256
    x = torch.rand(2, 5, 256, 256).to(device)
 
    with torch.no_grad():
        y = net(x)
 
    print(f"Input shape:  {x.shape}")   # (2, 5, 256, 256)
    print(f"Output shape: {y.shape}")   # (2, 1, 256, 256) fire mask logits
 
    # Verify output range is unbounded (logits, not probabilities)
    print(f"Output min/max (logits): {y.min():.3f} / {y.max():.3f}")
    print(f"Output after sigmoid  :  {y.sigmoid().min():.3f} / {y.sigmoid().max():.3f}")
 
    # Quick timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = net(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f"Mean forward pass: {(time.time() - t0) / 10 * 1000:.1f} ms")

