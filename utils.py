import torch
import torchvision
import torch.nn as nn


def get_resnet_based_model(freeze_resnet=False, CUDA=True, num_classes=8):
    model = torchvision.models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = not freeze_resnet

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, num_classes)

    device = torch.device("cuda:0" if CUDA else "cpu")
    model = model.to(device)

    return model

