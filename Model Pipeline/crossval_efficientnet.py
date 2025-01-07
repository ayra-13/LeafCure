import os
import random
import torch
import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# Directories
# source_dir = "drive/MyDrive/Rice_Leaf_AUG/Rice_Leaf_AUG/Rice_Leaf_AUG"
data_dir = "RiceDataProcessed/"
# Model Definition
def create_model(num_classes):
    model = models.efficientnet_b0(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model
# Ratios
train_val_ratio = 0.9  # Train + Validation

def split_data(source_dir, output_dir, train_val_ratio):
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if image_files:
            random.shuffle(image_files)
            num_images = len(image_files)
            train_val_end = int(train_val_ratio * num_images)

            splits = {
                'train_val': image_files[:train_val_end],
                'test': image_files[train_val_end:],
            }

            for split, split_files in splits.items():
                split_dir = os.path.join(output_dir, split, os.path.relpath(root, source_dir))
                os.makedirs(split_dir, exist_ok=True)
                for image in split_files:
                    shutil.copy2(os.path.join(root, image), os.path.join(split_dir, image))

# split_data(source_dir, data_dir, train_val_ratio)
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Load datasets
train_val_data = datasets.ImageFolder(os.path.join(data_dir, 'train_val'), transform=transform)
test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
def cross_validate_and_save_models(dataset, num_folds=5):
    """
    Perform cross-validation and save models for each fold.

    Args:
        dataset: The training and validation dataset.
        num_folds: Number of folds for cross-validation.

    Returns:
        str: The path to the best model file.
    """
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    best_model_state = None
    best_val_accuracy = 0
    best_model_path = None

    labels = np.array([label for _, label in dataset])
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\nFold {fold + 1}/{num_folds}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        model = models.efficientnet_b0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, len(dataset.classes))
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

        for epoch in range(20):  # Adjust number of epochs as needed
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            scheduler.step()
            train_loss = running_loss / len(train_loader)

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_accuracy = correct / total
            print(f"Epoch {epoch + 1:02d} | Training Loss: {train_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

        model_path = f"model_fold_{fold + 1}_val_acc_{val_accuracy:.4f}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model for Fold {fold + 1} saved as {model_path}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            best_model_path = model_path

        print(f"Fold {fold + 1} Complete: Best Validation Accuracy = {val_accuracy:.4f}")

    # Save the best model
    print(f"\nBest model saved as {best_model_path} with accuracy {best_val_accuracy:.4f}")
    return best_model_path
def test_model(model_path, test_data):
    """
    Test the best model on the test dataset.

    Args:
        model_path: Path to the saved model file.
        test_data: The test dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(pretrained=False)  # Load without pretrained weights
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(test_data.classes))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")

# Run Cross-Validation
best_model_path = cross_validate_and_save_models(train_val_data)

# Test the Best Model
test_model(best_model_path, test_data)