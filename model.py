import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class CheXNetFPNGrayscale(nn.Module):
    def __init__(self, num_classes=10):
        super(CheXNetFPNGrayscale, self).__init__()
        # Load pre-trained DenseNet
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        # Modify the first convolutional layer to accept grayscale (1-channel) input
        self.features = densenet.features
        self.features.conv0 = nn.Conv2d(
            1,  # Single channel for grayscale
            64,  # Output channels remain the same
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Copy weights from the original conv0 (RGB) layer to initialize the new layer
        with torch.no_grad():
            self.features.conv0.weight = nn.Parameter(
                self.features.conv0.weight[:, 0:1, :, :]  # Copy weights for the first channel
            )

        # FPN layers
        self.conv6 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv5 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.out_conv = nn.Conv2d(256, num_classes, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

    def forward(self, x):
        # DenseNet feature extraction
        c3 = self.features[:6](x)
        c4 = self.features[6:8](c3)
        c5 = self.features[8:10](c4)
        c6 = self.features[10:](c5)

        # FPN feature aggregation
        p6 = self.conv6(c6)
        p5 = self.conv5(c5) + F.interpolate(p6, size=c5.shape[2:], mode='nearest')
        p4 = self.conv4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = self.conv3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')

        # Final output
        out = self.out_conv(p3)
        out = self.global_avg_pool(out)  # Apply global average pooling
        out = out.view(out.size(0), -1)  # Flatten
        return torch.sigmoid(out)
