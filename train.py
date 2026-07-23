import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from pathlib import Path

def train_model():
    # 1. Hyperparameters & Paths
    DATA_DIR = Path("data/chest_xray")
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)
    
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(using_device := f"Using device: {DEVICE}")

    # 2. Data Augmentation & Normalization
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Check if dataset has images
    train_path = DATA_DIR / "train"
    if not any(train_path.iterdir()) if train_path.exists() else True:
        print("❌ Training data directory is empty. Please populate data/chest_xray/train first using prepare_data.py guidelines.")
        return

    image_datasets = {
        x: datasets.ImageFolder(str(DATA_DIR / x), data_transforms[x])
        for x in ["train", "val"]
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        for x in ["train", "val"]
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    print(f"Classes found: {class_names}")

    # 3. Model Setup (ResNet18 Transfer Learning)
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace final fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 30)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    # 5. Save Checkpoint
    output_path = MODEL_DIR / "model.pt"
    torch.save(model.state_dict(), output_path)
    print(f"\n✅ Training complete! Model saved successfully to {output_path}")

if __name__ == "__main__":
    train_model()