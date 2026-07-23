import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class MedicalAIEngine:
    def __init__(self, model_path="models/model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["NORMAL", "PNEUMONIA"]
        
        # Load model structure & weights
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Model weights not found at {model_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Target layer for ResNet18 Grad-CAM (last convolutional layer of layer4)
        self.target_layers = [self.model.layer4[-1]]
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_and_generate_cam(self, image_path: str, output_cam_path: str = "static/outputs/cam_result.jpg"):
        os.makedirs(os.path.dirname(output_cam_path), exist_ok=True)
        
        # 1. Load and process image for model inference
        original_image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
        
        # 2. Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            
        predicted_class = self.classes[predicted_idx.item()]
        confidence_score = confidence.item()
        
        # 3. Grad-CAM generation
        rgb_img = np.array(original_image.resize((224, 224))) / 255.0
        
        with GradCAM(model=self.model, target_layers=self.target_layers) as cam:
            targets = [ClassifierOutputTarget(predicted_idx.item())]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
        # Save heatmap result image
        plt.imsave(output_cam_path, visualization)
        
        return predicted_class, confidence_score, output_cam_path