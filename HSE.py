import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms
from einops import rearrange

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x2 = torch.stack([x, x1], dim=0)
        out, _ = torch.max(x2, dim=0)
        return out

##layer_norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Feature_extract(nn.Module):
    '''
    特征提取模块
    '''
    def __init__(self, in_channels, out_channels):
        super(Feature_extract, self).__init__()
        self.SFEB1 = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels/2), kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(int(out_channels/2)),
            FReLU(int(out_channels/2)),
            nn.Conv2d(int(out_channels/2), int(out_channels/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(out_channels/2)),
            FReLU(int(out_channels/2)),
        )
        self.SFEB2= nn.Sequential(
            nn.Conv2d(int(out_channels/2), out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            FReLU(out_channels),
            nn.Conv2d(out_channels,  out_channels, kernel_size=3, stride=1, padding=1),)

    def forward(self, x):        
        high_x = self.SFEB1(x)
        x = self.SFEB2(high_x)
        return high_x, x

class IFP(nn.Module):

    def __init__(self, channels):        
        super(IFP, self).__init__()
        self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
        nn.Tanh(),
        )

    def forward(self, x):
        return (self.conv_block(x) + 1) / 2


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class FGM(nn.Module):

    def __init__(self, inc):

        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
        self.conv1 = nn.Sequential(nn.Conv2d(4 * inc,  int(inc/2), kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(int(inc/2)),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(int(inc/2), int(inc/2), kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(int(inc/2)),
                                        nn.ReLU(inplace=True)
                                        )
        
        self.conv2 = BBasicConv2d(int(inc/2), int(inc/2), 3, 1, 1)
        
        self.convbnrelu = nn.Sequential(nn.Conv2d(3 * int(inc/2),  int(inc/2), kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(int(inc/2)),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(int(inc/2), int(inc/2), kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(int(inc/2)),
                                        )

        self.SA = SpatialAttention(3)
        self.CA = ChannelAttention(32)

    def forward(self, x3, x4, x5, x6):
        B, C, H, W,= x4.shape
        output_size = (H, W)

        x3 = self.avg_pool(x3, output_size)
        x5 = F.interpolate(x5, size=(H, W), mode='bilinear', align_corners=False)
        x6 = F.interpolate(x6, size=(H, W), mode='bilinear', align_corners=False)
        x3456 = torch.cat([x3, x4, x5, x6], 1) #(8,256,60,80)
        x_r = self.conv1(x3456) #(8,192,60,80)
        x_f = self.conv2(x_r)
        x_ca = self.CA(x_r)
        x_sa = self.SA(x_r)
        x_ca = x_ca * x_r
        x_sa = x_sa * x_r
        out = self.convbnrelu(torch.cat((x_f, x_ca, x_sa),dim=1))

        return out



class FDM(nn.Module):
    def __init__(self, local_dim, inj_dim):
        super().__init__()
        
        self.avg_pool = nn.functional.adaptive_avg_pool2d
        self.att_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1) 
        self.sigmoid = nn.Sigmoid()
        
        self.inj_val = nn.Sequential(
            nn.Conv2d(inj_dim, local_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(local_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(local_dim, local_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(local_dim),
        )

        self.agg = nn.Sequential(
            nn.Conv2d(local_dim, local_dim, kernel_size = 3, stride=1, padding = 1),
            nn.BatchNorm2d(local_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(local_dim, local_dim, kernel_size = 3, stride=1, padding = 1),
            nn.BatchNorm2d(local_dim),
        )

    def forward(self, x_local, x_inj):
        _, _, H, W = x_local.shape
        _, _, h, w = x_inj.shape
        if H > h:
            x_inj = F.interpolate(x_inj, size=(H, W), mode='bilinear', align_corners=False)
        elif H < h:
            x_inj = self.avg_pool(x_inj, (H, W))
        
        x_inj_w, _ = torch.max(x_inj, dim=1, keepdim=True)
        x_inj_w = self.sigmoid(self.att_conv(x_inj_w))

        x_inj_v = self.inj_val(x_inj)
        out = x_local * x_inj_w + x_inj_v
        out = self.agg(out)
        return out


class MSIA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.norm1 = LayerNorm(dim)
        #self.norm2 = LayerNorm(dim)
        #self.norm3 = LayerNorm(dim)
        self.rgb_qkv = BBasicConv2d(dim, 3*dim, 3, 1, 1)
        self.thermal_qkv = BBasicConv2d(dim, 3*dim, 3, 1, 1)
        self.semantic_q = BBasicConv2d(dim, dim, 3, 1, 1)
        self.fus_conv = BBasicConv2d(2*dim, dim, 3, 1, 1)

    def forward(self, rgb, thermal, inj):
        b,c,h,w = rgb.shape
        inj = F.interpolate(inj, size=(h, w), mode='bilinear', align_corners=False)
        #rgb_norm = self.norm1(rgb)
        #thermal_norm_ = self.norm2(thermal)
        #inj_norm = self.norm3(inj)
        rgb_qkv = self.rgb_qkv(rgb)
        rgb_q,rgb_k,rgb_v = rgb_qkv.chunk(3,dim=1)
        rgb_q = rearrange(rgb_q, 'b c h w -> b c (h w)')
        rgb_k = rearrange(rgb_k, 'b c h w -> b (h w) c')
        rgb_v = rearrange(rgb_v, 'b c h w -> b c (h w)')
        thermal_qkv = self.thermal_qkv(thermal)
        thermal_q, thermal_k, thermal_v = thermal_qkv.chunk(3, dim=1)
        thermal_q = rearrange(thermal_q, 'b c h w -> b c (h w)')
        thermal_k = rearrange(thermal_k, 'b c h w -> b (h w) c')
        thermal_v = rearrange(thermal_v, 'b c h w -> b c (h w)')
        inj_q = self.semantic_q(inj)
        inj_q = rearrange(inj_q, 'b c h w -> b c (h w)')
      
        inj_rgb_qk = inj_q @ rgb_k  # b c c
        inj_thermal_qk = inj_q @ thermal_k # b c c
        inj_qk = inj_rgb_qk + inj_thermal_qk
        inj_att = inj_qk.softmax(dim=-1)
        inj_rgb_map = inj_att @ rgb_v
        inj_thermal_map = inj_att @ thermal_v
        rgb_qk = rgb_q @ rgb_k
        rgb_att = rgb_qk.softmax(dim=-1)
        rgb_map = rgb_att @ rgb_v
        thermal_qk = thermal_q @ thermal_k
        thermal_att = thermal_qk.softmax(dim=-1)
        thermal_map = thermal_att @ thermal_v
        
        rgb_map = rearrange(rgb_map, 'b c (h w) -> b c h w',h=h,w=w)
        thermal_map = rearrange(thermal_map, 'b c (h w) -> b c h w',h=h,w=w)
        inj_rgb_map = rearrange(inj_rgb_map, 'b c (h w) -> b c h w',h=h,w=w)
        inj_thermal_map = rearrange(inj_thermal_map, 'b c (h w) -> b c h w',h=h,w=w)
        
        rgb_map = rgb_map + inj_thermal_map + rgb
        thermal_map = thermal_map + inj_rgb_map + thermal
        fus = torch.cat([rgb_map,thermal_map],dim=1)
        fus = self.fus_conv(fus)
        return fus

class SGDN(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.rgb_block1 = BBasicConv2d(2*in_c, out_c, 3, 1, 1)
        self.rgb_block2 = BBasicConv2d(3*in_c, out_c, 3, 1, 1)
        self.rgb_block3 = BBasicConv2d(3*in_c, out_c, 3, 1, 1)
        self.thermal_block1 = BBasicConv2d(2 * in_c, out_c, 3, 1, 1)
        self.thermal_block2 = BBasicConv2d(2*in_c, out_c, 3, 1, 1)
        self.thermal_block3 = BBasicConv2d(2*in_c, out_c, 3, 1, 1)
        
    def forward(self, rgb, thermal, inj):
        b,c,h,w = rgb.shape
        inj = F.interpolate(inj, size=(h, w), mode='bilinear', align_corners=False)
        rgb1 = self.rgb_block1(torch.cat([rgb,thermal],dim=1))
        thermal1 = self.thermal_block1(torch.cat([thermal,inj],dim=1))
        
        rgb2 = self.rgb_block2(torch.cat([rgb,rgb1,thermal1],dim=1))
        thermal2 = self.thermal_block2(torch.cat([thermal1,inj],dim=1))
        
        rgb3 = self.rgb_block3(torch.cat([rgb1,rgb2,thermal2],dim=1))
        thermal3 = self.thermal_block3(torch.cat([thermal2,inj],dim=1))
        return rgb3,thermal3
        
class PSF(nn.Module):

    def __init__(self, n_classes):
        super(PSF, self).__init__()
        self.num_resnet_layers = 34
        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)

        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)

        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)

        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)

        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)

        self.dims = [32, 32, 64, 64, 64, 64]
        self.decoder_dim_rec = 32        
        self.decoder_dim_seg = 64
        
        
        ########  Thermal ENCODER  ########
        self.encoder_thermal_conv1 = Feature_extract(1, 64)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer3 = resnet_raw_model1.layer1
        self.encoder_thermal_layer4 = resnet_raw_model1.layer2
        self.encoder_thermal_layer5 = resnet_raw_model1.layer3
        self.encoder_thermal_layer6 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########
        self.encoder_rgb_conv1 = Feature_extract(3, 64)
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer3 = resnet_raw_model2.layer1
        self.encoder_rgb_layer4 = resnet_raw_model2.layer2
        self.encoder_rgb_layer5 = resnet_raw_model2.layer3
        self.encoder_rgb_layer6 = resnet_raw_model2.layer4

        self.high_fuse6 = SSFM(512, 64, 128)
        self.high_fuse5 = SSFM(256, 64, 128)
        self.high_fuse4 = SSFM(128, 64, 128)
        self.high_fuse3 = SSFM(64, 64, 128)

        self.modality_interaction1 = SGDN(32,32)
        self.modality_interaction2 = SGDN(32,32)
        self.semantic_interaction3 =  MSIA(32)
        
        self.fgm = FGM(64)
        self.fdm_x3 = FDM(64, 32)
        self.fdm_x4 = FDM(64, 32)


        self.fgm = FGM(64)
        self.fdm_x5 = FDM(64, 32)
        self.fdm_x6 = FDM(64, 32)


        self.to_fused_seg = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, self.decoder_dim_seg, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor = 2 ** i, mode='bilinear', align_corners=True)
        ) for i, dim in enumerate(self.dims[2:])])

        self.channal_rec = nn.Conv2d(self.decoder_dim_seg, self.decoder_dim_rec, 3, 1, 1)
        self.seg_decoder = S2PM(4 * self.decoder_dim_seg, self.decoder_dim_seg)
        self.seg_rec_decoder = DSRM(self.decoder_dim_rec, self.decoder_dim_rec)
        self.sem_decoder = Semantic_Head(feature=self.decoder_dim_seg, n_classes=n_classes)
        self.pred_fusion = IFP([self.decoder_dim_rec, 1])   

        
    def forward(self, rgb, depth):
        rgb = rgb
        thermal = depth[:, :1, ...]

        vobose = False

        # encoder
        ######################################################################

        if vobose: print("rgb.size() original: ", rgb.size())  # (480, 640)
        if vobose: print("thermal.size() original: ", thermal.size())  # (480, 640)

        ######################################################################

        rgb1, rgb2 = self.encoder_rgb_conv1(rgb)  # (240, 320)
        rgb2 = self.encoder_rgb_bn1(rgb2)  # (240, 320)
        rgb2 = self.encoder_rgb_relu(rgb2) # (240, 320)

        thermal1, thermal2 = self.encoder_thermal_conv1(thermal)  # (240, 320)
        thermal2 = self.encoder_thermal_bn1(thermal2)  # (240, 320)
        thermal2 = self.encoder_thermal_relu(thermal2) # (240, 320)

        ######################################################################
        rgb3 = self.encoder_rgb_maxpool(rgb2)  # (120, 160)
        thermal3 = self.encoder_thermal_maxpool(thermal2) # (120, 160)
        rgb3 = self.encoder_rgb_layer3(rgb3) # (120, 160)
        thermal3 = self.encoder_thermal_layer3(thermal3)   # (120, 160)

        ######################################################################
        rgb4= self.encoder_rgb_layer4(rgb3)  # (60, 80)
        thermal4 = self.encoder_thermal_layer4(thermal3)  # (60, 80)

        ######################################################################
        rgb5 = self.encoder_rgb_layer5(rgb4)  # (30, 40)
        thermal5 = self.encoder_thermal_layer5(thermal4) # (30, 40)

        ######################################################################
        rgb6 = self.encoder_rgb_layer6(rgb5)  # (30, 40)
        thermal6 = self.encoder_thermal_layer6(thermal5) # (30, 40)

       ## fused featrue
        B, C, H, W = rgb1.shape
        fused_f6 = self.high_fuse6(rgb6,thermal6)
        fused_f5 = self.high_fuse5(rgb5,thermal5)
        fused_f4 = self.high_fuse4(rgb4,thermal4)
        fused_f3 = self.high_fuse3(rgb3, thermal3)


        rgb_in=rgb1
        thermal_in=thermal1
        f_inj1 = self.fgm(fused_f3, fused_f4, fused_f5, fused_f6)
        rgb1, thermal1 = self.modality_interaction1(rgb1, thermal1, f_inj1)
        rgb_out=rgb1
        thermal_out=thermal1
        
        fused_f3 = self.fdm_x3(fused_f3, f_inj1)
        fused_f4 = self.fdm_x4(fused_f4, f_inj1)

        f_inj2 = self.fgm(fused_f3, fused_f4, fused_f5, fused_f6)
        rgb1, thermal1 = self.modality_interaction2(rgb1, thermal1, f_inj2)
        fused_f5 = self.fdm_x5(fused_f5, f_inj2)
        fused_f6 = self.fdm_x6(fused_f6, f_inj2)

        encoded_featrues_seg = [fused_f3, fused_f4, fused_f5, fused_f6]
        seg_fused_f = [to_fused(output) for output, to_fused in zip(encoded_featrues_seg, self.to_fused_seg)] 
        seg_f = torch.cat(seg_fused_f, dim=1)

        ## sparse scene understanding 
        seg_f = self.seg_decoder(seg_f)  
        semantic_out = self.sem_decoder(seg_f)

        rec_seg_f = self.channal_rec(seg_f)
        fus = self.semantic_interaction3(rgb1, thermal1, rec_seg_f)
        
        ## image reconstruction
        rec_f = self.seg_rec_decoder(fus)
        fused_img =  self.pred_fusion(rec_f)
        
        #f_inj1 = F.interpolate(f_inj1, size=(H, W), mode='bilinear', align_corners=False)
        
        return semantic_out, fused_img#, rgb_in, thermal_in, rgb_out, thermal_out#, f_inj1


class SSFM(nn.Module):
    def __init__(self, in_C, out_C, cat_C):
        super(SSFM, self).__init__()
        self.out_c = out_C
        self.rgb_pconv = nn.Conv2d(in_C, 3*out_C, 1)
        self.rgb_dconv = nn.Conv2d(3*out_C, 3*out_C, 3, stride=1, padding=1, groups=3*out_C)
        self.depth_pconv = nn.Conv2d(in_C, 3*out_C, 1)
        self.depth_dconv = nn.Conv2d(3*out_C, 3*out_C, 3, stride=1, padding=1, groups=3*out_C)
        self.conv1 = nn.Conv2d(in_C,out_C,1)
        self.conv2 = nn.Conv2d(in_C,out_C,1)
        
        self.rgb_depth_conv = nn.Sequential(nn.Conv2d(2, 16, 3, 1, 1),
                                            nn.BatchNorm2d(16),
                                            FReLU(16),
                                            nn.Conv2d(16, 2, 3, 1, 1),
                                            nn.BatchNorm2d(2),
                                            nn.Sigmoid(),
                                            )

        self.fus_conv = BBasicConv2d(cat_C, out_C, 3, 1, 1)

    def forward(self, rgb, depth):
        rgb_temp = self.rgb_pconv(rgb)
        rgb_temp = self.rgb_dconv(rgb_temp)
        rgb_q,rgb_k,rgb_v = torch.split(rgb_temp,self.out_c,dim=1)
        depth_temp = self.depth_pconv(depth)
        depth_temp = self.depth_dconv(depth_temp)
        depth_q, depth_k, depth_v = torch.split(depth_temp, self.out_c, dim=1)
        
        b, c, h, w = rgb_q.shape
        rgb_q = rearrange(rgb_q, 'b c h w -> b c (h w)')
        rgb_k = rearrange(rgb_k, 'b c h w -> b (h w) c')
        rgb_v = rearrange(rgb_v, 'b c h w -> b c (h w)')
        depth_q = rearrange(depth_q, 'b c h w -> b c (h w)')
        depth_k = rearrange(depth_k, 'b c h w -> b (h w) c')
        depth_v = rearrange(depth_v, 'b c h w -> b c (h w)')

        rgb_qk = rgb_q @ depth_k #b c c
        depth_qk = depth_q @ rgb_k  # b c c
        rgb_qk = torch.unsqueeze(rgb_qk,dim=1)
        depth_qk = torch.unsqueeze(depth_qk,dim=1)
        rgb_depth_qk = torch.cat([rgb_qk,depth_qk],dim=1)
        rgb_depth_qk = self.rgb_depth_conv(rgb_depth_qk)
        rgb_qk, depth_qk = torch.split(rgb_depth_qk, 1, dim=1)
        rgb_qk = torch.squeeze(rgb_qk,dim=1)
        depth_qk = torch.squeeze(depth_qk,dim=1)

        rgb_qk = rgb_qk.softmax(dim=-1)
        depth_qk = depth_qk.softmax(dim=-1)
        rgb_att = rgb_qk @ rgb_v   # b c hw
        depth_att = depth_qk @ depth_v # b c hw
        rgb_att = rearrange(rgb_att, 'b c (h w) -> b c h w', h=h, w=w)
        depth_att = rearrange(depth_att, 'b c (h w) -> b c h w', h=h, w=w)
        rgb = self.conv1(rgb)
        depth = self.conv2(depth)
        rgb_refine = depth_att + rgb
        depth_refine = rgb_att + depth
        fus = torch.cat([rgb_refine,depth_refine],dim=1)
        fus = self.fus_conv(fus)

        return fus


class BBasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BBasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

#########################################################################################################    Inception


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU6(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x





#########################################################################################################     decoder

        

class S2PM(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(S2PM, self).__init__()
        self.block1 = nn.Sequential(
            BBasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )
        self.block2 = nn.Sequential(
            BBasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )
        self.block3 = nn.Sequential(
            BBasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        out = self.block3(x2)
        return out

class DSRM(nn.Module):
    def __init__(self, in_channel=32, out_channel=32):
        super(DSRM, self).__init__()
        self.block1 = nn.Sequential(
            BBasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )
        self.block2 = nn.Sequential(
            BBasicConv2d(2*out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )
        self.block3 = nn.Sequential(
            BBasicConv2d(3 * out_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )
        
        self.block4 = nn.Sequential(
            BBasicConv2d(4 * out_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(torch.cat([x, x1],dim=1))
        x3 = self.block3(torch.cat([x, x1, x2], dim=1))
        out = self.block4(torch.cat([x, x1, x2, x3], dim=1))
        return out

class Semantic_Head(nn.Module):
    '''This path plays the role of a classifier and is responsible for predicting the results of semantic segmentation, binary segmentation and edge segmentation'''
    def __init__(self, feature=64, n_classes=9):
        super(Semantic_Head, self).__init__()
        
        self.semantic_conv1 = ConvBNReLU(feature, feature, kernel_size=3)
        #self.semantic_conv2 = ConvBNReLU(feature, feature, kernel_size=3)
        #self.semantic_conv3 = ConvBNReLU(feature, int(feature/2), kernel_size=3)
        #self.semantic_conv4 = ConvBNReLU(int(feature/2), int(feature/2), kernel_size=3)
        self.semantic_conv5 = nn.Conv2d(feature, n_classes, kernel_size=3, stride=1, padding=1)

        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, feat):

        feat_sematic = self.semantic_conv1(feat)
        #feat_sematic = self.semantic_conv2(feat_sematic)
        feat_sematic = self.up2x(feat_sematic)
        #feat_sematic = self.semantic_conv3(feat_sematic)
        #feat_sematic = self.semantic_conv4(feat_sematic) + feat_sematic
        semantic_out = self.semantic_conv5(feat_sematic)
        semantic_out = self.up2x(semantic_out)

        return semantic_out

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1, bn=True, relu=True):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        # padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                              bias=False if bn else True)
        self.bn = bn
        if bn:
            self.bnop = nn.BatchNorm2d(out_planes)
        self.relu = relu
        if relu:
            self.reluop = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bnop(x)
        if self.relu:
            x = self.reluop(x)
        return x
        
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x