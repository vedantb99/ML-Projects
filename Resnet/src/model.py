import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet18_model(num_classes=10, pretrained=False):
    """
    Returns a ResNet-18 model adapted for CIFAR-10.
    - num_classes: number of output classes (10 for CIFAR-10)
    - pretrained: whether to load pretrained weights (on ImageNet)
    """

    # Load the standard ResNet-18 architecture
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

    # Modify the final fully connected layer
    in_features = model.fc.in_features   # input dimension of the last FC layer
    model.fc = nn.Linear(in_features, num_classes)

    return model
