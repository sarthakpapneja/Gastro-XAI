import torch.nn as nn
from torchvision import models
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

class GastroClassifier(nn.Module):
    def __init__(self, num_classes=8, freeze_backbone=True):
        """
        Args:
            num_classes (int): Number of output classes (8 for Kvasir v2).
            freeze_backbone (bool): If True, freeze ResNet50 weights.
        """
        super(GastroClassifier, self).__init__()
        # Load pretrained ResNet50
        print(f"Loading ResNet50 with num_classes={num_classes}...")
        self.backbone = models.resnet50(pretrained=True)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_last_conv_layer(self):
        # Return the last convolutional layer for Grad-CAM
        return self.backbone.layer4[2].conv3 

if __name__ == "__main__":
    model = GastroClassifier(num_classes=8)
    print(model)
