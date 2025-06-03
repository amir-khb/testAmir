import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .attention import ChannelAttention, SpatialAttention


class DCRN(nn.Module):
    """Dual-Channel Residual Network"""

    def __init__(self, input_channels=103, patch_size=7, n_classes=None):
        super(DCRN, self).__init__()
        self.feature_dim = input_channels
        self.sz = patch_size

        # Spectral path as shown in the image
        # 1×1×7, 64 -> 1×1×7, 64 -> 1×1×7, 512 -> 1×1×21, 512 -> 1×1×21, 1024
        self.spec_conv1 = nn.Conv2d(input_channels, 64, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn1 = nn.BatchNorm2d(64)

        self.spec_conv2 = nn.Conv2d(64, 64, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn2 = nn.BatchNorm2d(64)

        self.spec_conv3 = nn.Conv2d(64, 512, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn3 = nn.BatchNorm2d(512)

        self.spec_conv4 = nn.Conv2d(512, 512, kernel_size=(1, 21), padding=(0, 10))
        self.spec_bn4 = nn.BatchNorm2d(512)

        self.spec_conv5 = nn.Conv2d(512, 1024, kernel_size=(1, 21), padding=(0, 10))
        self.spec_bn5 = nn.BatchNorm2d(1024)

        # Spatial path as shown in the image
        # Initial 3x3 conv followed by ResNet50 layers
        self.spat_conv1 = nn.Conv2d(input_channels, 3, kernel_size=3, padding=1)
        self.spat_bn1 = nn.BatchNorm2d(3)

        # Use ResNet50 layers
        resnet = models.resnet50(pretrained=True)

        # Extract the layers from ResNet50
        self.layer1 = resnet.layer1  # ResNet50 layer1
        self.layer2 = resnet.layer2  # ResNet50 layer2
        self.layer3 = resnet.layer3  # ResNet50 layer3
        self.layer4 = resnet.layer4  # ResNet50 layer4

        # Modified adapter with stride=1 instead of stride=2 to preserve spatial dimensions
        self.adapter = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),  # Changed stride from 2 to 1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # Changed stride from 2 to 1
        )

        # Sizes for feature combination
        self.spectral_size = 1024
        self.spatial_size = 2048  # ResNet50's final layer outputs 2048 channels
        self.total_output_channels = 3072  # From the image (total output channels)

        # Attention mechanisms (keep as in original)
        self.ca = ChannelAttention(self.total_output_channels)
        self.sa = SpatialAttention()

        # Projection for final feature dimension
        self.projection = nn.Conv2d(self.spectral_size + self.spatial_size,
                                    self.total_output_channels,
                                    kernel_size=1)

        # Disentanglement parameters
        self.spec_bn_1d = nn.BatchNorm1d(self.spectral_size)
        self.spat_bn_1d = nn.BatchNorm1d(self.spatial_size)

        # Upsample layer to restore spatial dimensions for spatial path
        self.upsample = nn.Upsample(size=(patch_size, patch_size), mode='bilinear', align_corners=False)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m not in resnet.modules():
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) and m not in resnet.modules():
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)

        # Spectral pathway
        spec = self.spec_conv1(x)
        spec = F.relu(self.spec_bn1(spec))

        spec = self.spec_conv2(spec)
        spec = F.relu(self.spec_bn2(spec))

        spec = self.spec_conv3(spec)
        spec = F.relu(self.spec_bn3(spec))

        spec = self.spec_conv4(spec)
        spec = F.relu(self.spec_bn4(spec))

        spec = self.spec_conv5(spec)
        spec = F.relu(self.spec_bn5(spec))

        # Spatial pathway
        spat = self.spat_conv1(x)
        spat = F.relu(self.spat_bn1(spat))

        # Pass through ResNet50 layers with modified adapter (stride=1)
        spat = self.adapter(spat)
        spat = self.layer1(spat)
        spat = self.layer2(spat)
        spat = self.layer3(spat)
        spat = self.layer4(spat)

        # Upsample spatial features to match original dimensions
        spat = self.upsample(spat)

        # Keep separate features for disentanglement
        pooled_spec = F.adaptive_avg_pool2d(spec, 1).view(batch_size, -1)
        pooled_spat = F.adaptive_avg_pool2d(spat, 1).view(batch_size, -1)

        # Normalize for better generalization
        if self.training:
            pooled_spec = self.spec_bn_1d(pooled_spec)
            pooled_spat = self.spat_bn_1d(pooled_spat)

        # Combine features
        combined = torch.cat((spec, spat), 1)

        # Project to desired output dimension (3072 channels)
        combined = self.projection(combined)

        # Keep original combined for residual connection
        combined_residual = combined

        # Apply attention with residual (keeping the attention mechanism as requested)
        combined = self.ca(combined) * combined + 0.2 * combined_residual
        combined = self.sa(combined) * combined + 0.2 * combined_residual

        # Save for feature analysis
        pixel_features = combined

        # Pool to get final features
        pooled = F.adaptive_avg_pool2d(combined, 1).view(batch_size, -1)

        return {
            'pixel': pixel_features,
            'features': pooled,
            'spectral_features': pooled_spec,
            'spatial_features': pooled_spat
        }