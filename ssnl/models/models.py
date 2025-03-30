from .basic_layers import *
from .DCT_layer import *
from .DWT_layer import *


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encode = nn.Sequential(
            ResBlock(3, 64, 1), 
            ResBlock(64, 64, 1), 
            ResBlock(64, 64, 1), 
            ResBlock(64, 64, 1), 
            ResBlock(64, 64, 1), 
            ResBlock(64, 64, 1)
        )
    
    def forward(self, imgs):
        output = self.encode(imgs)

        return output


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decode = nn.Sequential(
            ResBlock(64, 64, 1), 
            ResBlock(64, 64, 1), 
            ResBlock(64, 64, 1), 
            ResBlock(64, 64, 1), 
            ResBlock(64, 64, 1), 
            ResBlock(64, 64, 1),
            nn.Conv2d(64, 3, 3, stride=1, padding=1, bias=False),
        )
    
    def forward(self, imgs):
        output = self.decode(imgs)

        return output
    

class TransformLayer(nn.Module):
    def __init__(self):
        super(TransformLayer, self).__init__()

        self.transform = nn.Sequential(
            ConvINRelu(64, 64, 3), 
            ConvINRelu(64, 64, 3), 
        )
    
    def forward(self, imgs):
        output = self.transform(imgs)

        return output
    


class Encoder_Unet(nn.Module):
    def __init__(self):
        super(Encoder_Unet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ResBlock(3, 16)
        self.Conv2 = ResBlock(16, 32)
        self.Conv3 = ResBlock(32, 64)
    
    def forward(self, imgs):
        # 16*128*128
        x1 = self.Conv1(imgs)
        # 32*64*64
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        # 64*32*32
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # 64*16*16
        x4 = self.Maxpool(x3)

        return x1, x2, x3, x4


class Decoder_Unet(nn.Module):
    def __init__(self):
        super(Decoder_Unet, self).__init__()

        self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.up4 = UpConvBNRelu(128, 64)
        self.conv3 = ResBlock(128, 64)

        self.up3 = UpConvBNRelu(64, 32)
        self.conv2 = ResBlock(64, 32)

        self.up2 = UpConvBNRelu(32, 16)
        self.conv1 = ResBlock(32, 16)

        self.conv_last = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
        
    
    def forward(self, cover_imgs, x1, x2, x3, x4):
        d4 = self.Globalpool(x4)
        d4 = d4.repeat(1,1,4,4)

        d3 = self.up4(torch.cat([x4, d4], dim=1))
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.conv3(d3)

        d2 = self.up3(d3)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.conv2(d2)

        d1 = self.up2(d2)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.conv1(d1)

        imgs = cover_imgs + self.conv_last(d1)

        return imgs
    

class TransformLayer_Unet(nn.Module):
    def __init__(self):
        super(TransformLayer_Unet, self).__init__()

        self.transform1 = nn.Sequential(
            ConvINRelu(16, 16, 3), 
            ConvINRelu(16, 16, 3), 
        )

        self.transform2 = nn.Sequential(
            ConvINRelu(32, 32, 3), 
            ConvINRelu(32, 32, 3), 
        )

        self.transform3 = nn.Sequential(
            ConvINRelu(64, 64, 3), 
            ConvINRelu(64, 64, 3), 
        )

    
    def forward(self, x1, x2, x3, x4):
        t1 = self.transform1(x1)
        t2 = self.transform2(x2)
        t3 = self.transform3(x3)

        return t1, t2, t3, x4

from .basic_layers import ResidualDenseBlock_out as DB

class Noise_INN_block(nn.Module):
    def __init__(self, clamp=2.0):
        super().__init__()

        self.clamp = clamp
        self.r = DB(input=3, output=9)
        self.y = DB(input=3, output=9)
        self.f = DB(input=9, output=3)


    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):

        x1, x2 = x[0], x[1]

        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2

            s1, t1 = self.r(y1), self.y(y1)

            y2 = torch.exp(s1) * x2 + t1

            out = [y1, y2]

        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / torch.exp(s1)

            t2 = self.f(y2)
            y1 = x1 - t2

            out = [y1, y2]
        return out


from .Haar import HaarDownsampling

class INN_OSN(nn.Module):
    def __init__(self):
        super(INN_OSN, self).__init__()

        self.inv1 = Noise_INN_block()
        self.inv2 = Noise_INN_block()
        self.inv3 = Noise_INN_block()
        self.inv4 = Noise_INN_block()
        self.inv5 = Noise_INN_block()
        self.inv6 = Noise_INN_block()
        self.inv7 = Noise_INN_block()
        self.inv8 = Noise_INN_block()

    def forward(self, x, rev=False):

        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)
        else:

            out = self.inv8(x, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)
        return out


class INL(nn.Module):
    def __init__(self):
        super(INL, self).__init__()
        self.model = INN_OSN()
        self.haar = HaarDownsampling(3)

    def forward(self, x, rev=False):

        if not rev:
            out = self.haar(x)
            out = self.model(out)
            out = self.haar(out, rev=True)

        else:
            out = self.haar(x)
            out = self.model(out, rev=True)
            out = self.haar(out, rev=True)

        return out
    

class Discriminator(nn.Module):
    """
    鉴别器，鉴别传入图像是否带有水印
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(3, 64)]
        layers.append(DoubleConvBNRelu(64, 64))
        
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(64, 1)

    def forward(self, img):
        # (b, c, 1, 1)
        x = self.layers(img)
        # (b, c)
        x.squeeze_(-1).squeeze_(-1)
        x = self.linear(x)
        return x




class Unet_OSN(nn.Module):
    def __init__(self):
        super(Unet_OSN, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ResBlock(3, 16)
        self.Conv2 = ResBlock(16, 32)
        self.Conv3 = ResBlock(32, 64)

        self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.up4 = UpConvBNRelu(128, 64)
        self.conv6 = ResBlock(128, 64)

        self.up3 = UpConvBNRelu(64, 32)
        self.conv5 = ResBlock(64, 32)

        self.up2 = UpConvBNRelu(32, 16)
        self.conv4 = ResBlock(32, 16)

        self.conv_last = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
    
    def forward(self, cover_imgs):
        # encoder
        # 16*128*128
        x1 = self.Conv1(cover_imgs)
        # 32*64*64
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        # 64*32*32
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # 64*16*16
        x4 = self.Maxpool(x3)


        # decoder
        d4 = self.Globalpool(x4)
        d4 = d4.repeat(1,1,4,4)

        d3 = self.up4(torch.cat([x4, d4], dim=1))
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.conv6(d3)

        d2 = self.up3(d3)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.conv5(d2)

        d1 = self.up2(d2)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.conv4(d1)

        imgs = cover_imgs + self.conv_last(d1)

        return imgs



class Unet_OSN_DB(nn.Module):
    def __init__(self):
        super(Unet_OSN_DB, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DB(3, 16)
        self.Conv2 = DB(16, 32)
        self.Conv3 = DB(32, 64)

        self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.up4 = UpConvBNRelu(128, 64)
        self.conv6 = DB(128, 64)

        self.up3 = UpConvBNRelu(64, 32)
        self.conv5 = DB(64, 32)

        self.up2 = UpConvBNRelu(32, 16)
        self.conv4 = DB(32, 16)

        self.conv_last = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
    
    def forward(self, cover_imgs):
        # encoder
        # 16*128*128
        x1 = self.Conv1(cover_imgs)
        # 32*64*64
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        # 64*32*32
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # 64*16*16
        x4 = self.Maxpool(x3)


        # decoder
        d4 = self.Globalpool(x4)
        d4 = d4.repeat(1,1,4,4)

        d3 = self.up4(torch.cat([x4, d4], dim=1))
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.conv6(d3)

        d2 = self.up3(d3)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.conv5(d2)

        d1 = self.up2(d2)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.conv4(d1)

        imgs = cover_imgs + self.conv_last(d1)

        return imgs




## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 64,
        num_blocks = [2,2,2,2], 
        num_refinement_blocks = 2,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

