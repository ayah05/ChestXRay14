import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, num_classes=10, metadata_features=6):
        super(ResNet50, self).__init__()
        # Load pretrained ResNet-50
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept grayscale input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Add an extra max-pooling layer after the first bottleneck block
        self.extra_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Extract ResNet feature layers up to the last pooling layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # Adaptive pooling to ensure the output size matches the paper (7x7 -> 1x1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for image features
        self.image_fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Fully connected layer for metadata
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x, metadata):
        # Forward pass for image data
        x = self.features(x)  # ResNet feature extraction
        x = self.extra_pooling(x)  # Extra pooling layer after the first bottleneck block
        x = self.avgpool(x)  # Adaptive pooling to 1x1
        x = torch.flatten(x, 1)  # Flatten to vector
        x = self.image_fc(x)  # Fully connected layer for image features

        # Forward pass for metadata
        metadata = self.metadata_fc(metadata)

        # Concatenate image and metadata features
        combined = torch.cat([x, metadata], dim=1)

        # Classification
        return self.classifier(combined)
