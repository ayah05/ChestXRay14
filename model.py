import torch
import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        resnet = models.resnet50(weights='IMAGENET1K_V1')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_base = nn.Sequential(*list(resnet.children())[:-2])
        self.additional_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_fc = nn.Linear(2048, 2048)

        self.combined_fc = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        x = self.resnet_base(image)
        x = self.additional_maxpool(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.image_fc(x)

        output = self.combined_fc(x)
        return output