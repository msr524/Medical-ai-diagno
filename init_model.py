import torch
import torch.nn as nn
from torchvision import models
import os

os.makedirs("models", exist_ok=True)
model_path = "models/resnet18_medical.pth"

if not os.path.exists(model_path):
    print("Initializing base ResNet18 weights for testing...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )
    torch.save(model.state_dict(), model_path)
    print(f"Successfully saved base model weights to {model_path}")
else:
    print("Model weights already exist.")