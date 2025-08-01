import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca_weight = self.ca(x)
        x = x * ca_weight
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa_weight = self.sa(sa_input)
        x = x * sa_weight
        
        return x

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for enhanced feature learning"""
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1),
                nn.BatchNorm2d(growth_rate),
                nn.ReLU(inplace=True)
            ))
        self.conv1x1 = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        out = torch.cat(features, dim=1)
        out = self.bn(self.conv1x1(out))
        return out + x

class EnhancedDecoder(nn.Module):
    """Enhanced decoder block with attention and dense connections"""
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.has_skip = skip_channels > 0
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

        if self.has_skip:
            self.skip_attention = nn.Sequential(
                nn.Conv2d(skip_channels, skip_channels, 1),
                nn.BatchNorm2d(skip_channels),
                nn.Sigmoid()
            )
            self.conv_after_concat = nn.Conv2d(out_channels + skip_channels, out_channels, 1)

        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Dataset-specific growth rate will be handled in config
        self.rdb = ResidualDenseBlock(out_channels, growth_rate=32)
        
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, skip=None):
        x = self.upconv(x)

        if self.has_skip and skip is not None:
            skip_weights = self.skip_attention(skip)
            skip = skip * skip_weights
            x = torch.cat([x, skip], dim=1)
            x = self.conv_after_concat(x)

        x = self.double_conv(x)
        x = self.rdb(x)
        x = x * self.channel_att(x)

        return x

class EdgeDetectionHead(nn.Module):
    """Edge detection head for boundary-aware segmentation"""
    def __init__(self, in_channels):
        super().__init__()
        self.edge_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.edge_head(x)

class BoundaryMedSeg(nn.Module):
    """
    BoundaryMedSeg: Boundary-Guided Multi-Domain Medical Image Segmentation
    
    Unified architecture supporting:
    - BUSI: 1-channel → 1-class (grayscale ultrasound)
    - ISIC 2018: 3-channel → 1-class (RGB dermoscopy)
    - BraTS 2020: 4-channel → 4-class (multimodal MRI)
    - CVC-ClinicDB: 3-channel → 1-class (RGB endoscopy)
    - Kvasir-SEG: 3-channel → 1-class (RGB endoscopy)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.boundary_guidance_weight = config.boundary_guidance_weight
        
        # Input processing based on dataset
        if config.input_channels != 3:
            self.stem_conv = nn.Sequential(
                nn.Conv2d(config.input_channels, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, 1)
            )
        else:
            self.stem_conv = None
        
        # PVTv2 backbone encoder
        self.encoder = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)
        encoder_channels = [64, 128, 320, 512]
        fpn_channels = 256

        # Feature Pyramid Network
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, fpn_channels, 1),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True)
            ) for ch in encoder_channels
        ])

        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])

        # Enhanced decoder pathway
        self.decoder1 = EnhancedDecoder(fpn_channels, fpn_channels, skip_channels=fpn_channels)
        self.decoder2 = EnhancedDecoder(fpn_channels, fpn_channels, skip_channels=fpn_channels)
        self.decoder3 = EnhancedDecoder(fpn_channels, fpn_channels, skip_channels=fpn_channels)
        self.decoder4 = EnhancedDecoder(fpn_channels, fpn_channels // 2, skip_channels=0)

        # Global attention
        self.cbam = CBAMBlock(fpn_channels // 2)

        # Final segmentation head
        self.final_conv = nn.Sequential(
            nn.Conv2d(fpn_channels // 2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, config.num_classes, 1)
        )

        # Boundary detection head
        self.boundary_head = EdgeDetectionHead(fpn_channels // 2)
        
        # Regularization
        self.dropout = nn.Dropout2d(config.dropout_rate)

    def forward(self, x):
        input_size = x.shape[2:]
        
        # Convert input channels to RGB if needed
        if self.stem_conv is not None:
            x = self.stem_conv(x)
        
        # Encoder pathway
        features = self.encoder(x)

        # Feature Pyramid Network
        fpn_features = [lateral(features[i]) for i, lateral in enumerate(self.lateral_convs)]

        # Top-down pathway
        p5 = self.smooth_convs[3](fpn_features[3])
        p4 = self.smooth_convs[2](
            fpn_features[2] + F.interpolate(p5, size=features[2].shape[2:], mode='bilinear', align_corners=True)
        )
        p3 = self.smooth_convs[1](
            fpn_features[1] + F.interpolate(p4, size=features[1].shape[2:], mode='bilinear', align_corners=True)
        )
        p2 = self.smooth_convs[0](
            fpn_features[0] + F.interpolate(p3, size=features[0].shape[2:], mode='bilinear', align_corners=True)
        )

        # Apply dropout
        p5, p4, p3, p2 = self.dropout(p5), self.dropout(p4), self.dropout(p3), self.dropout(p2)

        # Enhanced decoder pathway
        d1 = self.decoder1(p5, p4)
        d2 = self.decoder2(d1, p3)
        d3 = self.decoder3(d2, p2)
        d4 = self.decoder4(d3)

        # Apply global attention
        d4 = self.cbam(d4)

        # Generate predictions
        seg_logits = self.final_conv(d4)
        boundary_logits = self.boundary_head(d4)

        # Resize to input dimensions
        if seg_logits.shape[2:] != input_size:
            seg_logits = F.interpolate(seg_logits, size=input_size, mode='bilinear', align_corners=True)
            boundary_logits = F.interpolate(boundary_logits, size=input_size, mode='bilinear', align_corners=True)

        # Boundary-guided refinement
        boundary_guidance = torch.sigmoid(boundary_logits)
        
        if self.config.num_classes > 1:
            # Multi-class: expand boundary guidance
            boundary_guidance = boundary_guidance.expand(-1, self.config.num_classes, -1, -1)
        
        refined_seg_logits = seg_logits + self.boundary_guidance_weight * seg_logits * boundary_guidance

        return refined_seg_logits, boundary_logits
