# -*- coding: utf-8 -*-
"""labelme1250k-dinov2-all-methods.ipynb"""

import numpy as np
import pandas as pd
import torch
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from PIL import Image
from transformers import AutoModel, AutoConfig, AutoImageProcessor
from tqdm import tqdm
import gc

# -----------------------------
# SETUP
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

dataset_path = "/home/shabbeer/Research/datasets/LabelMe1250K"

# -----------------------------
# DATASET
# -----------------------------
class LabelMeDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []

        # Read classes
        classes_file = os.path.join(root, 'classes.txt')
        with open(classes_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        data_dir = os.path.join(root, 'train' if train else 'test')
        annotation_file = os.path.join(data_dir, 'annotation.txt')

        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        # Collect all image paths
        image_paths = []
        subdirs = range(40) if train else range(10)
        for i in subdirs:
            subdir = f"{i:04d}"
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.exists(subdir_path):
                for img_file in os.listdir(subdir_path):
                    if img_file.endswith('.jpg'):
                        image_paths.append(os.path.join(subdir_path, img_file))

        image_name_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            img_name = parts[0]
            scores = list(map(float, parts[1:13]))
            max_score = max(scores)
            class_idx = 12 if max_score < 0.5 else scores.index(max_score)
            if img_name in image_name_to_path:
                self.data.append(image_name_to_path[img_name])
                self.targets.append(class_idx)

        print(f"Loaded {len(self.data)} images for {'train' if train else 'test'}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, target = self.data[idx], self.targets[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target

# -----------------------------
# MODEL
# -----------------------------
class DINOv2WithClassifier(nn.Module):
    def __init__(self, backbone, num_classes=13):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.hidden_size, num_classes)
    
    def forward(self, x):
        outputs = self.backbone(x)
        features = outputs.last_hidden_state[:, 0, :]
        return self.classifier(features)
MODEL_NAME = "facebook/dinov2-small-imagenet1k-1-layer"
def load_dinov2_model(pretrained=False, num_classes=13):
    if pretrained:
        print(f"Loading pretrained {MODEL_NAME}")
        backbone = AutoModel.from_pretrained(MODEL_NAME)
    else:
        print("Creating randomly initialized DINOv2 model")
        config = AutoConfig.from_pretrained(MODEL_NAME)
        backbone = AutoModel.from_config(config)

    model = DINOv2WithClassifier(backbone, num_classes)

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)   # use_fast is default now
    transform = transforms.Compose([
        transforms.Resize(processor.size["shortest_edge"]),
        transforms.CenterCrop(processor.size["shortest_edge"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    return model, transform

# -----------------------------
# TRAIN / TEST
# -----------------------------
def test_model(model, test_loader, device):
    model.eval()
    model.to(device)
    correct, total, loss_sum = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
            # Clear batch from GPU
            del inputs, labels, outputs, loss, pred
            
    # Clear cache after testing
    torch.cuda.empty_cache()
    return correct * 100 / total, loss_sum / len(test_loader)

def train_model(
    model, 
    train_loader, 
    test_loader, 
    epochs, 
    lr, 
    path, 
    device, 
    accumulate_steps=4,
    patience=15,           # <-- NEW: early stopping patience
    min_delta=0.01         # <-- NEW: minimum improvement to count as progress
):
    best_acc = 0.0
    patience_counter = 0
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Initial evaluation
    test_acc, _ = test_model(model, test_loader, device)
    print(f"Initial accuracy: {test_acc:.2f}%")
    best_acc = test_acc

    # Save initial model (optional)
    torch.save(model.state_dict(), f"{path}/model_{best_acc:.2f}.pt")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y) / accumulate_steps
            loss.backward()
            
            running_loss += loss.item() * accumulate_steps
            
            if (i + 1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Clear batch
            del x, y, outputs, loss
            
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        # Step on remaining gradients
        if len(train_loader) % accumulate_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()
        torch.cuda.empty_cache()
        
        # Evaluate
        acc, test_loss = test_model(model, test_loader, device)
        print(f"Epoch {epoch+1}: Train Loss={running_loss/len(train_loader):.4f}, "
              f"Test Loss={test_loss:.4f}, Acc={acc:.2f}%")
        
        # === EARLY STOPPING LOGIC ===
        if acc > best_acc + min_delta:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), f"{path}/model_{best_acc:.2f}.pt")
            print(f"New best model saved: {best_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
        # ==============================

    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Training finished. Best accuracy: {best_acc:.2f}%")
    return best_acc

def load_best_model_from_folder(folder, model, device):
    model_files = [f for f in os.listdir(folder) if f.startswith("model_") and f.endswith(".pt")]
    if not model_files: 
        print("No saved models found")
        return model
    best = max(model_files, key=lambda x: float(x.split("_")[1][:-3]))
    model.load_state_dict(torch.load(os.path.join(folder, best), map_location=device))
    print(f"Loaded {best}")
    return model

# -----------------------------
# CONFIDENCE HELPERS
# -----------------------------
def get_low_confidence_images(model, dataset, k, device, batch_size=128):
    """Select low confidence images with proper memory management"""
    model.eval()
    model.to(device)
    scores, ids = [], []
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Calculating low confidence for {len(dataset)} samples...")
    with torch.no_grad():
        for bi, (x, _) in enumerate(loader):
            x = x.to(device)
            outputs = model(x)
            conf = torch.nn.functional.softmax(outputs, dim=1).max(dim=1)[0]
            scores.extend(conf.cpu().tolist())
            
            # Store indices
            start_idx = bi * batch_size
            end_idx = min(start_idx + len(x), len(dataset))
            ids.extend(range(start_idx, end_idx))
            
            # Clear GPU memory
            del x, outputs, conf
            
            if (bi + 1) % 50 == 0:
                torch.cuda.empty_cache()
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Select lowest confidence samples
    sorted_indices = np.argsort(scores)
    selected_idx = sorted_indices[:k].tolist()
    selected_global = [ids[i] for i in selected_idx]
    
    # Create remainder indices
    selected_set = set(selected_global)
    remainder_global = [i for i in range(len(dataset)) if i not in selected_set]
    
    return Subset(dataset, selected_global), Subset(dataset, remainder_global)

def get_high_confidence_images(model, dataset, k, device, batch_size=128):
    """Select high confidence images with proper memory management"""
    model.eval()
    model.to(device)
    scores, ids = [], []
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Calculating high confidence for {len(dataset)} samples...")
    with torch.no_grad():
        for bi, (x, _) in enumerate(loader):
            x = x.to(device)
            outputs = model(x)
            conf = torch.nn.functional.softmax(outputs, dim=1).max(dim=1)[0]
            scores.extend(conf.cpu().tolist())
            
            # Store indices
            start_idx = bi * batch_size
            end_idx = min(start_idx + len(x), len(dataset))
            ids.extend(range(start_idx, end_idx))
            
            # Clear GPU memory
            del x, outputs, conf
            
            if (bi + 1) % 50 == 0:
                torch.cuda.empty_cache()
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Select highest confidence samples
    sorted_indices = np.argsort(scores)[::-1]  # Reverse for high confidence
    selected_idx = sorted_indices[:k].tolist()
    selected_global = [ids[i] for i in selected_idx]
    
    # Create remainder indices
    selected_set = set(selected_global)
    remainder_global = [i for i in range(len(dataset)) if i not in selected_set]
    
    return Subset(dataset, selected_global), Subset(dataset, remainder_global)

# -----------------------------
# TRAIN UNTIL EMPTY
# -----------------------------
def train_until_empty_custom(model, init_set, remain_set, test_dataset, path, device, add_percent=0.05, mode="low+low", max_iterations=50):
    """Train with iterative sample addition"""
    original_dataset = remain_set.dataset if isinstance(remain_set, Subset) else remain_set
    total = len(original_dataset)
    add_size = max(1, int(total * add_percent))
    
    accs = []
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    # Initial accuracy
    best, _ = test_model(model, test_loader, device)
    accs.append(best)
    print(f"Initial Acc: {best:.2f}%")

    current_train = init_set
    remaining = remain_set
    
    iteration = 0
    while iteration < max_iterations and len(remaining) > 0:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration} ({mode}) - Remaining: {len(remaining)}")
        print(f"{'='*60}")
        
        # Determine how many to add
        k = min(add_size, len(remaining))
        if k == 0:
            print("No more samples to add")
            break
        
        # Select new samples based on mode
        if mode == "low+low":
            new_samples, remaining = get_low_confidence_images(model, remaining, k, device, batch_size=128)
        elif mode == "high+high":
            new_samples, remaining = get_high_confidence_images(model, remaining, k, device, batch_size=128)
        elif mode == "high+low":
            half = k // 2
            high_samples, temp_remaining = get_high_confidence_images(model, remaining, half, device, batch_size=128)
            low_samples, remaining = get_low_confidence_images(model, temp_remaining, k - half, device, batch_size=128)
            new_samples = ConcatDataset([high_samples, low_samples])
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Combine training sets
        current_train = ConcatDataset([current_train, new_samples])
        print(f"Training set size: {len(current_train)}")
        
        # Create dataloaders with reduced batch size
        train_loader = DataLoader(current_train, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        
        # Train for 1 epoch with the expanded dataset
        acc = train_model(model, train_loader, test_loader, epochs=50, lr=0.0001, path=path, device=device, accumulate_steps=2)
        accs.append(acc)
        
        print(f"Iteration {iteration} Acc: {acc:.2f}%")
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
    return accs

# -----------------------------
# RUN ALL METHODS
# -----------------------------
def run_three_methods(model, full_train, test_data, model_path, device):
    """Run all three active learning methods"""
    results = {}
    total = len(full_train)
    init_size = int(total * 0.04)
    
    print("\n" + "="*60)
    print("Preparing initial sets for all methods...")
    print("="*60)
    
    # Get initial sets (do this once to save time)
    init_low, rem_low = get_low_confidence_images(model, full_train, init_size, device, batch_size=128)
    print(f"Low confidence initial set: {len(init_low)} samples")
    
    init_high, rem_high = get_high_confidence_images(model, full_train, init_size, device, batch_size=128)
    print(f"High confidence initial set: {len(init_high)} samples")
    
    # Save initial model state
    initial_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    methods = {
        "Method 2 (Low+Low)": ("low+low", init_low, rem_low)
        # "Method 3 (High+High)": ("high+high", init_high, rem_high)
        # "Method 4 (High+Low)": ("high+low", init_high, rem_low)
    }
    
    for method_name, (mode, init, rem) in methods.items():
        print("\n" + "="*60)
        print(f"Running {method_name}")
        print("="*60)
        
        # Reset model to initial state
        model.load_state_dict(initial_state)
        model.to(device)
        torch.cuda.empty_cache()
        
        # Train initial set first
        print(f"Training initial set for {method_name} (50 epochs)...")
        train_loader = DataLoader(init, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
        initial_acc = train_model(model, train_loader, test_loader, epochs=50, lr=0.001, path=model_path, device=device, accumulate_steps=2)
        
        # Load best model
        model = load_best_model_from_folder(model_path, model, device)
        
        # Run iterative training
        accs = train_until_empty_custom(model, init, rem, test_data, model_path, device, add_percent=0.05, mode=mode, max_iterations=50)
        results[method_name] = {"initial": initial_acc, "iterations": accs}
        
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save results
    torch.save(results, os.path.join(model_path, "three_methods_results.pt"))
    return results

# -----------------------------
# EXPERIMENT FUNCTION
# -----------------------------
def run_experiment(exp_name, pretrained=False):
    print(f"\n{'='*60}")
    print(f"Running {exp_name}")
    print(f"{'='*60}")
    
    results_dir = f"./results_low_scratch/{exp_name}"
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, "models")
    os.makedirs(model_path, exist_ok=True)

    # Load model
    model, transform = load_dinov2_model(pretrained, num_classes=13)
    model = model.to(device)
    torch.cuda.empty_cache()
    
    # Load datasets
    print("Loading datasets...")
    train_ds = LabelMeDataset(dataset_path, True, transform)
    test_ds = LabelMeDataset(dataset_path, False, transform)

    # Initial 4% selection
    total = len(train_ds)
    init_size = int(total * 0.04)
    print(f"Selecting initial {init_size} samples ({0.04*100:.1f}%)...")
    
    init_low, rem = get_low_confidence_images(model, train_ds, init_size, device, batch_size=128)
    print(f"Initial training set: {len(init_low)} samples")
    
    # Train initial model
    train_loader = DataLoader(init_low, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    print("Training initial model (50 epochs)...")
    init_acc = train_model(model, train_loader, test_loader, epochs=50, lr=0.001, path=model_path, device=device, accumulate_steps=2)

    # Load best model
    model = load_best_model_from_folder(model_path, model, device)

    # Run three methods
    print("\nRunning active-learning methods...")
    results = run_three_methods(model, train_ds, test_ds, model_path, device)
    
    # Get best final accuracy
    best_final = max(v["iterations"][-1] for v in results.values())

    # Save results
    with open(os.path.join(results_dir, "final_results.txt"), "w") as f:
        f.write(f"Experiment: {exp_name}\nPretrained: {pretrained}\n")
        f.write(f"Initial Acc: {init_acc:.2f}%\nBest Final: {best_final:.2f}%\n")
        f.write(f"Improvement: {best_final - init_acc:.2f}%\n")
        f.write("\nMethod Details:\n")
        for method, data in results.items():
            f.write(f"{method}: {data['iterations'][-1]:.2f}%\n")
    
    print(f"\n{'='*60}")
    print(f"Experiment Complete: {exp_name}")
    print(f"Best Final Accuracy: {best_final:.2f}%")
    print(f"{'='*60}")
    
    return best_final

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    import time
    # print("="*60)
    # print("EXPERIMENT 1: PRETRAINED DINOv2")
    # print("="*60)
    # acc1 = run_experiment("exp1_pretrained_dinov2", pretrained=True)
    start_time = time.time()
    print("\n" + "="*60)
    print("EXPERIMENT 2: NON-PRETRAINED DINOv2")
    print("="*60)
    acc2 = run_experiment("exp2_nonpretrained_dinov2", pretrained=False)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    # print(f"Pretrained: {acc1:.2f}%")
    print(f"Non-pretrained: {acc2:.2f}%")
    # print(f"Difference: {abs(acc1-acc2):.2f}%")
    print("="*60)
    end_time = time.time()

    elapsed_seconds = end_time - start_time
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)

    print(f"Time taken: {hours} hour(s) and {minutes} minute(s)")