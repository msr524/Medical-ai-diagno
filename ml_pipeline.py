import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from torchvision import models, transforms

MODEL_PATH = "models/model.pt"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # alphabetical order — matches ImageFolder/training

device = torch.device("cpu")

# 1. Rebuild the EXACT same architecture used in train.py
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

# 2. Load YOUR real trained weights (not a fresh pretrained model)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()  # Mandatory for correct inference

# 3. Preprocessing must match train.py's "val" transform exactly
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# Hook the last conv block of torchvision resnet18 (final feature map before avgpool)
target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

def predict_and_generate_gradcam(image_path: str, output_gradcam_path: str):
    global activations, gradients

    original_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(original_image).unsqueeze(0).to(device)

    model.zero_grad()
    outputs = model(input_tensor)

    probabilities = F.softmax(outputs, dim=-1)
    confidence, predicted_idx = torch.max(probabilities, dim=-1)

    pred_class_id = predicted_idx.item()
    conf_score = confidence.item()
    predicted_class_name = CLASS_NAMES[pred_class_id]

    try:
        target_score = outputs[0, pred_class_id]
        target_score.backward()

        if gradients is not None and activations is not None:
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            activations_tensor = activations.detach()[0]
            for i in range(activations_tensor.shape[0]):
                activations_tensor[i, :, :] *= pooled_gradients[i]

            heatmap = torch.mean(activations_tensor, dim=0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)
            else:
                heatmap = np.zeros_like(heatmap)

            img_np = np.array(original_image.resize((224, 224)))

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img_np)
            ax.imshow(heatmap, cmap='jet', alpha=0.4, extent=(0, 224, 224, 0))
            ax.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

            os.makedirs(os.path.dirname(output_gradcam_path), exist_ok=True)
            plt.savefig(output_gradcam_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close(fig)
        else:
            original_image.save(output_gradcam_path)
    except Exception as cam_err:
        print(f"Grad-CAM generation failed: {str(cam_err)}")
        original_image.save(output_gradcam_path)

    return predicted_class_name, conf_score