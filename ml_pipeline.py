import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 1. Configuration & Model Loading (Synchronized with Training)
MODEL_CHECKPOINT = "microsoft/resnet-18"
processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

device = torch.device("cpu")
model = AutoModelForImageClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=2, ignore_mismatched_sizes=True)
model.to(device)
model.eval()  # Mandatory for correct inference

# Container for storing gradients and activations for Grad-CAM
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# Register hooks on Hugging Face ResNet's final convolutional layer (layer4)
try:
    # For microsoft/resnet-18 on Hugging Face, the final block is model.resnet.encoder.stages.3.layers.1.conv2 or similar
    # Let's target the last stage's final layer safely:
    target_layer = model.resnet.encoder.stages[-1].layers[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)
except Exception as e:
    print(f"Warning: Could not register Grad-CAM hooks: {str(e)}")

def predict_and_generate_gradcam(image_path: str, output_gradcam_path: str):
    global activations, gradients
    
    # Load input image
    original_image = Image.open(image_path).convert("RGB")
    
    # 2. Synchronized Preprocessing via AutoImageProcessor
    inputs = processor(images=original_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 3. Forward pass with gradient tracking enabled for Grad-CAM
    model.zero_grad()
    outputs = model(**inputs)
    logits = outputs.logits

    # 4. Correct Probability Conversion (Single Softmax)
    probabilities = F.softmax(logits, dim=-1)
    confidence, predicted_idx = torch.max(probabilities, dim=-1)
    
    pred_class_id = predicted_idx.item()
    conf_score = confidence.item()

    id2label = {0: "NORMAL", 1: "PNEUMONIA"}
    predicted_class_name = id2label.get(pred_class_id, "UNKNOWN")

    # 5. Grad-CAM Generation
    try:
        # Score for target class
        target_score = logits[0, pred_class_id]
        target_score.backward()

        if gradients is not None and activations is not None:
            # Pool gradients across spatial dimensions
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            
            # Weight activations by gradients
            activations_tensor = activations.detach()[0]
            for i in range(activations_tensor.shape[0]):
                activations_tensor[i, :, :] *= pooled_gradients[i]
                
            # Average across channels to create heatmap
            heatmap = torch.mean(activations_tensor, dim=0).cpu().numpy()
            
            # ReLU on heatmap
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)
            else:
                heatmap = np.zeros_like(heatmap)

            # Overlay heatmap on original image using matplotlib
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
            # Fallback if hooks didn't capture tensors
            original_image.save(output_gradcam_path)
            
    except Exception as cam_err:
        print(f"Grad-CAM generation failed: {str(cam_err)}")
        original_image.save(output_gradcam_path)

    return predicted_class_name, conf_score