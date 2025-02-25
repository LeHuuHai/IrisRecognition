import os

import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import VGG16_Weights


# Xây dựng mô hình
class FeaturesExtraction:
    def __init__(self):
        # Sử dụng VGG16 pre-trained
        self.vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.vgg16 = nn.Sequential(*list(self.vgg16.children())[:-1])
        self.vgg16.eval()
        self.flatten = nn.Flatten()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extraction(self, image):
        image = Image.fromarray(image).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)  # Thêm một chiều batch
        with torch.no_grad():  # Không tính gradient khi suy luận
            features = self.vgg16(image_tensor)
        features_flatten = self.flatten(features)
        features_numpy = features_flatten.squeeze().cpu().numpy()  # Chuyển tensor thành numpy array và loại bỏ chiều batch
        return features_numpy

