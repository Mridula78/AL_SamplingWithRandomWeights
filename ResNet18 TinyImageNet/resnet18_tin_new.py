import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, datasets
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
from torchvision.datasets import ImageFolder
import numpy as np
import json
import time
import warnings
import argparse
from PIL import Image
import glob
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
print(f"Using device: {device}")

# Define ResNet18 model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2]).to(device)

# Custom TinyImageNet Dataset Class
class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.classes = []
        self.class_to_idx = {}
        self.images = []
        self.labels = []
        
        if split == 'train':
            self._load_train_data()
        elif split == 'val':
            self._load_val_data()
        elif split == 'test':
            self._load_test_data()
    
    def _load_train_data(self):
        train_dir = os.path.join(self.root_dir, 'train')
        class_dirs = sorted(os.listdir(train_dir))
        
        for class_idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir] = class_idx
            self.classes.append(class_dir)
            
            class_path = os.path.join(train_dir, class_dir, 'images')
            image_files = glob.glob(os.path.join(class_path, '*.JPEG'))
            
            for img_file in image_files:
                self.images.append(img_file)
                self.labels.append(class_idx)
    
    def _load_val_data(self):
        val_dir = os.path.join(self.root_dir, 'val')
        
        # Read validation annotations
        with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
            lines = f.readlines()
        
        # Build class mapping from train directory
        train_dir = os.path.join(self.root_dir, 'train')
        class_dirs = sorted(os.listdir(train_dir))
        for class_idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir] = class_idx
            self.classes.append(class_dir)
        
        # Process validation images
        for line in lines:
            parts = line.strip().split('\t')
            img_name = parts[0]
            class_name = parts[1]
            
            img_path = os.path.join(val_dir, 'images', img_name)
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
    
    def _load_test_data(self):
        test_dir = os.path.join(self.root_dir, 'test', 'images')
        image_files = glob.glob(os.path.join(test_dir, '*.JPEG'))
        
        for img_file in image_files:
            self.images.append(img_file)
            self.labels.append(-1)  # No labels for test
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data preprocessing for TinyImageNet
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize(72),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load TinyImageNet dataset
def load_tinyimagenet_data(data_dir='./tiny-imagenet-200'):
    """
    Load TinyImageNet dataset
    Args:
        data_dir: Directory containing TinyImageNet data
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    print(f"Loading TinyImageNet data from {data_dir}")
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist.")
        print("Please download TinyImageNet from: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
        print("And extract it to the current directory.")
        exit(1)
    
    try:
        train_dataset = TinyImageNetDataset(data_dir, transform=train_transform, split='train')
        val_dataset = TinyImageNetDataset(data_dir, transform=test_transform, split='val')
        test_dataset = TinyImageNetDataset(data_dir, transform=test_transform, split='test')
        
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        print(f"Number of classes: {len(train_dataset.classes)}")
        
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

# Function to test the model
def test_model(model, test_loader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing", leave=False):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    return accuracy, avg_loss

# Function to train the model
def train_model(model, train_loader, test_loader, epochs, learning_rate, path, 
                accumulate_steps=1, phase="train"):
    best_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                         momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    total_train_time = 0.0
    epoch_accuracies = []
    epoch_times = []
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{phase}]", leave=False)
        for i, data in enumerate(train_bar):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if (i + 1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': running_loss / (i + 1),
                'acc': 100 * correct / total
            })
        
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time
        epoch_times.append(epoch_time)
        
        scheduler.step()
        
        # Test model
        new_accuracy, test_loss = test_model(model, test_loader)
        epoch_accuracies.append(new_accuracy)
        epoch_losses.append(running_loss / len(train_loader))
        
        print(f"Epoch {epoch+1}/{epochs} [{phase}]: "
              f"Train Loss: {epoch_losses[-1]:.4f}, "
              f"Test Acc: {new_accuracy:.2f}%, "
              f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if new_accuracy > best_accuracy:
            model_path = f"{path}/model_{new_accuracy:.2f}.pt"
            if os.path.exists(model_path):
                os.remove(model_path)
            best_accuracy = new_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'loss': epoch_losses[-1],
            }, model_path)
    
    # Save epoch-wise metrics
    metrics = {
        'epoch_accuracies': epoch_accuracies,
        'epoch_losses': epoch_losses,
        'epoch_times': epoch_times,
        'total_training_time': total_train_time,
        'best_accuracy': best_accuracy
    }
    
    with open(f"{path}/epoch_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    with open(f"{path}/epoch_metrics.txt", "w") as f:
        f.write("Epoch,Accuracy,Loss,Time(s)\n")
        for i, (acc, loss, t) in enumerate(zip(epoch_accuracies, epoch_losses, epoch_times), start=1):
            f.write(f"{i},{acc:.2f},{loss:.4f},{t:.2f}\n")
        f.write(f"\nTotal training time: {total_train_time:.2f}s\n")
        f.write(f"Best accuracy: {best_accuracy:.2f}%\n")
    
    print(f"Total training time for this phase: {total_train_time:.2f}s")
    return best_accuracy, total_train_time, epoch_accuracies, epoch_times

# Function for initial training
def train_test_save(train_dataset, test_loader, n, epochs, path, lr=0.01):
    best_accuracy = 0.0
    total_time = 0.0
    all_epoch_accuracies = []
    all_epoch_times = []
    
    for iteration in range(n):
        print(f"\nStarting training iteration {iteration + 1}/{n}")
        model = ResNet18().to(device)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, 
                                  num_workers=4, pin_memory=True)
        new_accuracy, iter_time, epoch_accs, epoch_times = train_model(
            model, train_loader, test_loader, epochs, lr, path, phase=f"iter_{iteration+1}"
        )
        total_time += iter_time
        all_epoch_accuracies.extend(epoch_accs)
        all_epoch_times.extend(epoch_times)

        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
    
    print(f"\nTotal training time for train_test_save: {total_time:.2f}s")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    return best_accuracy, total_time, all_epoch_accuracies, all_epoch_times

# Function to get samples by confidence
def get_samples_by_confidence(model, dataset, k, selection_type='low'):
    """
    Get samples based on confidence level
    
    Args:
        model: PyTorch model
        dataset: PyTorch dataset
        k: number of samples to select
        selection_type: 'low' for low confidence, 'high' for high confidence
    """
    model.eval()
    model = model.to(device)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, 
                            num_workers=4, pin_memory=True)
    confidence_scores = []
    all_indices = list(range(len(dataset)))

    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc="Calculating confidence", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            confidences = torch.nn.functional.softmax(outputs, dim=1).max(dim=1)[0]
            confidence_scores.extend(confidences.tolist())

    if selection_type == 'low':
        # Select k samples with LOWEST confidence
        selected_indices = sorted(range(len(confidence_scores)), 
                                  key=lambda i: confidence_scores[i], 
                                  reverse=False)[:k]
    else:  # 'high'
        # Select k samples with HIGHEST confidence
        selected_indices = sorted(range(len(confidence_scores)), 
                                  key=lambda i: confidence_scores[i], 
                                  reverse=True)[:k]
    
    selected_samples = Subset(dataset, selected_indices)
    remaining_indices = [i for i in range(len(dataset)) if i not in selected_indices]
    remainder_dataset = Subset(dataset, remaining_indices)
    
    print(f"Selected {k} {selection_type}-confidence samples from {len(dataset)} total samples")
    if confidence_scores:
        print(f"Confidence range: min={min(confidence_scores):.4f}, max={max(confidence_scores):.4f}")
    
    return selected_samples, remainder_dataset

# Function to load the best model from folder
def load_best_model_from_folder(folder_path):
    def get_accuracy_from_filename(filename):
        try:
            return float(filename.split("_")[1][:-3])
        except:
            return 0.0

    model_files = [file for file in os.listdir(folder_path) 
                  if file.startswith("model_") and file.endswith(".pt")]

    if not model_files:
        print("No model files found in the folder.")
        return None
    else:
        try:
            best_model_filename = max(model_files, key=get_accuracy_from_filename)
            best_model_path = os.path.join(folder_path, best_model_filename)
            
            checkpoint = torch.load(best_model_path, map_location=device)
            best_model = ResNet18().to(device)
            best_model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"Loaded model from: {best_model_path}")
            print(f"Model accuracy: {checkpoint['accuracy']:.2f}%")
            return best_model
        except Exception as e:
            print(f"Error loading the model: {e}")
            return None

# Function to train until degradation
def train_until_degradation(model, train_dataset, remaining_set, test_set, path, 
                           batch_size=5000, selection_type='low', initial_model_training=False):
    """
    Train model until all data is consumed
    
    Args:
        batch_size: number of samples to add each iteration
        selection_type: 'low' for low confidence, 'high' for high confidence
        initial_model_training: if True, this is the initial burn-in phase
    """
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    if initial_model_training:
        print(f"\nStarting BURN-IN phase with {len(train_dataset)} samples")
        # For burn-in phase, just train on initial dataset
        current_accuracy, total_time, epoch_accuracies, epoch_times = train_model(
            model, 
            DataLoader(train_dataset, batch_size=128, shuffle=True, 
                      num_workers=4, pin_memory=True), 
            test_loader, 100, 0.01, path, phase="burnin"
        )
        
        # Save burn-in metrics
        burnin_metrics = {
            "burnin_time_seconds": total_time,
            "burnin_time_minutes": total_time / 60,
            "burnin_time_hours": total_time / 3600,
            "final_accuracy": current_accuracy,
            "epoch_accuracies": epoch_accuracies,
            "epoch_times": epoch_times
        }
        
        with open(f"{path}/burnin_metrics.json", "w") as f:
            json.dump(burnin_metrics, f, indent=4)
        
        print(f"\nBURN-IN PHASE COMPLETE:")
        print(f"  Final accuracy: {current_accuracy:.2f}%")
        print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} minutes, {total_time/3600:.2f} hours)")
        
        return model, current_accuracy, total_time, epoch_accuracies, epoch_times
    
    else:
        # Original degradation training logic
        best_accuracy = test_model(model, test_loader)[0]
        current_accuracy = 0.0
        accuracies = []
        iteration_times = []
        iteration = 1
        total_iteration_time = 0.0

        print(f"\nStarting degradation training with {len(remaining_set)} remaining samples")
        print(f"Selection type: {selection_type}-confidence, Batch size: {batch_size}")
        
        while len(remaining_set) > 0:
            print(f"\n--- Degradation Iteration {iteration} ---")
            iteration_start_time = time.time()
            
            # Get samples based on confidence level
            if len(remaining_set) < batch_size:
                batch_size = len(remaining_set)  # Use remaining samples if less than batch_size
            
            new_images, remaining_set = get_samples_by_confidence(model, remaining_set, 
                                                                 k=batch_size, 
                                                                 selection_type=selection_type)
            
            # Add new images to training dataset
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, new_images])   
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, 
                                     num_workers=4, pin_memory=True)      
            
            # Train for 100 epochs
            current_accuracy, iter_time, epoch_accs, epoch_times = train_model(
                model, train_loader, test_loader, 100, 0.01, path, phase=f"degradation_{iteration}"
            )
            
            iteration_time = time.time() - iteration_start_time
            iteration_times.append(iteration_time)
            total_iteration_time += iteration_time
            
            accuracies.append(current_accuracy)
            
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
            
            print(f"Iteration {iteration} complete:")
            print(f"  - Current accuracy: {current_accuracy:.2f}%")
            print(f"  - Best accuracy: {best_accuracy:.2f}%")
            print(f"  - Iteration time: {iteration_time:.2f}s ({iteration_time/60:.2f} minutes)")
            print(f"  - Remaining samples: {len(remaining_set)}")
            print(f"  - Training set size: {len(train_dataset)}")
            
            iteration += 1
            
        print(f"\nTraining dataset exhausted. Stopping degradation training.")
        print(f"Total degradation training time: {total_iteration_time:.2f}s")
        print(f"Final best accuracy: {best_accuracy:.2f}%")

        # Save iteration-wise metrics
        with open(f"{path}/iteration_metrics.txt", "w") as file:
            file.write("Iteration,Accuracy(%),Time(s),Time(min),TrainingSetSize\n")
            for i, (acc, t) in enumerate(zip(accuracies, iteration_times), start=1):
                train_size = len(train_dataset) - len(remaining_set)
                file.write(f"{i},{acc:.2f},{t:.2f},{t/60:.2f},{train_size}\n")
            file.write(f"\nTotal degradation time: {total_iteration_time:.2f}s ({total_iteration_time/60:.2f} minutes)\n")
            file.write(f"Final best accuracy: {best_accuracy:.2f}%\n")

        # Save detailed metrics JSON
        degradation_metrics = {
            "total_degradation_time_seconds": total_iteration_time,
            "total_degradation_time_minutes": total_iteration_time / 60,
            "total_degradation_time_hours": total_iteration_time / 3600,
            "final_accuracy": best_accuracy,
            "iteration_accuracies": accuracies,
            "iteration_times_seconds": iteration_times,
            "iteration_times_minutes": [t/60 for t in iteration_times],
            "iteration_times_hours": [t/3600 for t in iteration_times]
        }
        
        with open(f"{path}/degradation_metrics.json", "w") as f:
            json.dump(degradation_metrics, f, indent=4)

        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'accuracy': best_accuracy,
            'degradation_metrics': degradation_metrics
        }, f"{path}/finalmodel_{best_accuracy:.2f}.pt")
        
        return model, best_accuracy, total_iteration_time, accuracies, iteration_times

# Main experiment function
def run_experiment(seed_idx, seed, experiment_config, data_dir='./tiny-imagenet-200'):
    """
    Run a complete experiment with a given seed and configuration
    
    Args:
        experiment_config: dictionary with keys:
            - name: experiment name
            - initial_selection: 'low' or 'high' for initial 4%
            - degradation_selection: 'low' or 'high' for 5% batches
    """
    print(f"\n{'='*80}")
    print(f"Running {experiment_config['name']}")
    print(f"Experiment {seed_idx+1} with seed: {seed}")
    print(f"Initial 4%: {experiment_config['initial_selection']}-confidence")
    print(f"Degradation 5%: {experiment_config['degradation_selection']}-confidence")
    print(f"{'='*80}")
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create directory for this experiment
    exp_name = experiment_config['name'].replace(" ", "_").replace("%", "").lower()
    path = f"./results/{exp_name}/seed_{seed}"
    os.makedirs(path, exist_ok=True)
    
    # Save experiment configuration
    with open(f"{path}/config.json", "w") as f:
        json.dump(experiment_config, f, indent=4)
    
    # Load data
    train_set, val_set, test_set = load_tinyimagenet_data(data_dir)
    total_train_samples = len(train_set)
    
    # Calculate sample sizes based on percentages
    initial_samples = int(total_train_samples * 0.04)  # 4% of total
    degradation_batch_size = int(total_train_samples * 0.05)  # 5% of total
    
    print(f"\nDataset Statistics:")
    print(f"  Total training samples: {total_train_samples}")
    print(f"  Initial samples (4%): {initial_samples}")
    print(f"  Degradation batch size (5%): {degradation_batch_size}")
    
    # Track total experiment time
    experiment_start_time = time.time()
    
    # Initialize model
    model = ResNet18().to(device)
    
    # Step 1: Get initial dataset (4% samples)
    print(f"\nStep 1: Getting initial {experiment_config['initial_selection']}-confidence dataset...")
    initial_trainset, remainder = get_samples_by_confidence(model, train_set, 
                                                           k=initial_samples, 
                                                           selection_type=experiment_config['initial_selection'])
    print(f"Initial training set size: {len(initial_trainset)} ({len(initial_trainset)/total_train_samples*100:.1f}%)")
    print(f"Remaining set size: {len(remainder)} ({len(remainder)/total_train_samples*100:.1f}%)")
    
    # Step 2: Train initial model (BURN-IN PHASE)
    print(f"\nStep 2: BURN-IN PHASE - Training initial model on {len(initial_trainset)} samples...")
    test_loader = DataLoader(val_set, batch_size=128, shuffle=False, 
                            num_workers=4, pin_memory=True)
    burnin_accuracy, burnin_time, burnin_epoch_accuracies, burnin_epoch_times = train_test_save(
        initial_trainset, test_loader, 1, 100, path, lr=0.01
    )
    
    # Save burn-in metrics
    burnin_metrics = {
        "burnin_samples": len(initial_trainset),
        "burnin_percentage": 4.0,
        "burnin_time_seconds": burnin_time,
        "burnin_time_minutes": burnin_time / 60,
        "burnin_time_hours": burnin_time / 3600,
        "burnin_accuracy": burnin_accuracy,
        "epoch_accuracies": burnin_epoch_accuracies,
        "epoch_times": burnin_epoch_times
    }
    
    with open(f"{path}/burnin_metrics.json", "w") as f:
        json.dump(burnin_metrics, f, indent=4)
    
    # Step 3: Load best model
    print("\nStep 3: Loading best model after burn-in...")
    model = load_best_model_from_folder(path)
    if model is None:
        print("Failed to load model. Skipping this run.")
        return 0.0, 0.0, 0.0, [], [], [], []
    
    # Step 4: Train until degradation
    print(f"\nStep 4: DEGRADATION PHASE - Adding {degradation_batch_size} samples per iteration ({experiment_config['degradation_selection']}-confidence)...")
    final_model, final_accuracy, degradation_time, iteration_accuracies, iteration_times = train_until_degradation(
        model, initial_trainset, remainder, val_set, path, 
        batch_size=degradation_batch_size, selection_type=experiment_config['degradation_selection']
    )
    
    # Calculate total experiment time
    total_experiment_time = time.time() - experiment_start_time
    
    # Calculate number of degradation iterations
    num_degradation_iterations = len(iteration_accuracies)
    
    # Prepare comprehensive summary
    summary = {
        "experiment_name": experiment_config['name'],
        "seed": seed,
        "total_experiment_time_seconds": total_experiment_time,
        "total_experiment_time_minutes": total_experiment_time / 60,
        "total_experiment_time_hours": total_experiment_time / 3600,
        "burnin_phase": {
            "samples": len(initial_trainset),
            "percentage": 4.0,
            "time_seconds": burnin_time,
            "time_minutes": burnin_time / 60,
            "time_hours": burnin_time / 3600,
            "accuracy": burnin_accuracy
        },
        "degradation_phase": {
            "batch_size": degradation_batch_size,
            "batch_percentage": 5.0,
            "num_iterations": num_degradation_iterations,
            "total_time_seconds": degradation_time,
            "total_time_minutes": degradation_time / 60,
            "total_time_hours": degradation_time / 3600,
            "final_accuracy": final_accuracy,
            "iteration_accuracies": iteration_accuracies,
            "iteration_times_seconds": iteration_times,
            "iteration_times_minutes": [t/60 for t in iteration_times],
            "iteration_times_hours": [t/3600 for t in iteration_times]
        },
        "accuracy_improvement": final_accuracy - burnin_accuracy
    }
    
    # Save comprehensive summary
    with open(f"{path}/experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    # Print detailed summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY: {experiment_config['name']} (Seed: {seed})")
    print(f"{'='*80}")
    print(f"\nBURN-IN PHASE (4% data):")
    print(f"  Samples: {len(initial_trainset)}")
    print(f"  Accuracy: {burnin_accuracy:.2f}%")
    print(f"  Time: {burnin_time:.2f}s ({burnin_time/60:.2f} minutes, {burnin_time/3600:.2f} hours)")
    
    print(f"\nDEGRADATION PHASE (5% per iteration):")
    print(f"  Batch size: {degradation_batch_size} samples")
    print(f"  Number of iterations: {num_degradation_iterations}")
    print(f"  Final accuracy: {final_accuracy:.2f}%")
    print(f"  Total degradation time: {degradation_time:.2f}s ({degradation_time/60:.2f} minutes, {degradation_time/3600:.2f} hours)")
    
    print(f"\nITERATION-WISE RESULTS:")
    for i, (acc, t) in enumerate(zip(iteration_accuracies, iteration_times), start=1):
        print(f"  Iteration {i}: Accuracy = {acc:.2f}%, Time = {t:.2f}s ({t/60:.2f} minutes)")
    
    print(f"\nOVERALL EXPERIMENT:")
    print(f"  Total time: {total_experiment_time:.2f}s ({total_experiment_time/60:.2f} minutes, {total_experiment_time/3600:.2f} hours)")
    print(f"  Accuracy improvement: {final_accuracy - burnin_accuracy:.2f}%")
    print(f"{'='*80}")
    
    return final_accuracy, total_experiment_time, burnin_time, degradation_time, iteration_accuracies, iteration_times

# Function to run all experiments
def run_all_experiments(data_dir='./tiny-imagenet-200'):
    """Run all 4 experiments with multiple seeds"""
    # Create results directory
    os.makedirs("./results", exist_ok=True)
    
    # Define seeds
    seeds = [42, 789, 101112]
    
    # Define all 4 experiments
    experiments = [
        {
            'name': 'Exp 1 - Low Confidence 4% + Low Confidence 5%',
            'initial_selection': 'low',
            'degradation_selection': 'low'
        },
        {
            'name': 'Exp 2 - High Confidence 4% + Low Confidence 5%',
            'initial_selection': 'high',
            'degradation_selection': 'low'
        },
        {
            'name': 'Exp 3 - High Confidence 4% + High Confidence 5%',
            'initial_selection': 'high',
            'degradation_selection': 'high'
        },
        {
            'name': 'Exp 4 - Low Confidence 4% + High Confidence 5%',
            'initial_selection': 'low',
            'degradation_selection': 'high'
        }
    ]
    
    # Store results from all runs
    all_results = {}
    
    # Run each experiment
    for exp_idx, experiment in enumerate(experiments):
        print(f"\n{'*'*100}")
        print(f"STARTING: {experiment['name']}")
        print(f"{'*'*100}")
        
        exp_accuracies = []
        exp_total_times = []
        exp_burnin_times = []
        exp_degradation_times = []
        all_iteration_accuracies = []
        all_iteration_times = []
        exp_summary = {}
        
        # Run with different seeds
        for seed_idx, seed in enumerate(seeds):
            accuracy, total_time, burnin_time, degradation_time, iteration_accs, iteration_times = run_experiment(
                seed_idx, seed, experiment, data_dir
            )
            
            exp_accuracies.append(accuracy)
            exp_total_times.append(total_time)
            exp_burnin_times.append(burnin_time)
            exp_degradation_times.append(degradation_time)
            all_iteration_accuracies.append(iteration_accs)
            all_iteration_times.append(iteration_times)
            
            exp_summary[f"seed_{seed}"] = {
                "accuracy": accuracy,
                "total_time_hours": total_time / 3600,
                "burnin_time_hours": burnin_time / 3600,
                "degradation_time_hours": degradation_time / 3600,
                "iteration_accuracies": iteration_accs,
                "iteration_times_hours": [t/3600 for t in iteration_times]
            }
            
            # Save intermediate results for this experiment
            exp_dir = f"./results/{experiment['name'].replace(' ', '_').replace('%', '').lower()}"
            os.makedirs(exp_dir, exist_ok=True)
            with open(f"{exp_dir}/results_summary.json", "w") as f:
                json.dump(exp_summary, f, indent=4)
        
        # Calculate statistics for this experiment
        if exp_accuracies and exp_total_times:
            mean_accuracy = np.mean(exp_accuracies)
            std_accuracy = np.std(exp_accuracies)
            mean_total_time = np.mean(exp_total_times)
            std_total_time = np.std(exp_total_times)
            mean_burnin_time = np.mean(exp_burnin_times)
            mean_degradation_time = np.mean(exp_degradation_times)
            
            exp_results = {
                'name': experiment['name'],
                'seeds': seeds,
                'accuracy_mean_std': f"{mean_accuracy:.2f} ± {std_accuracy:.2f}%",
                'total_time_hours': f"{mean_total_time/3600:.2f} ± {std_total_time/3600:.2f} hours",
                'burnin_time_hours': f"{mean_burnin_time/3600:.2f} hours",
                'degradation_time_hours': f"{mean_degradation_time/3600:.2f} hours",
                'detailed_results': exp_summary
            }
            
            all_results[experiment['name']] = exp_results
            
            # Save final results for this experiment
            with open(f"{exp_dir}/final_results.json", "w") as f:
                json.dump(exp_results, f, indent=4)
            
            # Save a simple text summary for this experiment
            with open(f"{exp_dir}/summary.txt", "w") as f:
                f.write(f"{experiment['name']} - FINAL RESULTS SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Configuration:\n")
                f.write(f"  Initial 4%: {experiment['initial_selection']}-confidence\n")
                f.write(f"  Degradation batches (5%): {experiment['degradation_selection']}-confidence\n\n")
                f.write(f"Statistics across {len(seeds)} seeds:\n")
                f.write(f"  Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%\n")
                f.write(f"  Total time: {mean_total_time/3600:.2f} ± {std_total_time/3600:.2f} hours\n")
                f.write(f"  Burn-in time: {mean_burnin_time/3600:.2f} hours\n")
                f.write(f"  Degradation time: {mean_degradation_time/3600:.2f} hours\n\n")
                
                f.write("Detailed Results per Seed:\n")
                for i, (seed, acc, total_t, burnin_t, deg_t) in enumerate(zip(seeds, exp_accuracies, 
                                                                              exp_total_times, exp_burnin_times, 
                                                                              exp_degradation_times)):
                    f.write(f"  Seed {seed}:\n")
                    f.write(f"    Final Accuracy: {acc:.2f}%\n")
                    f.write(f"    Total Time: {total_t/3600:.2f} hours\n")
                    f.write(f"    Burn-in Time: {burnin_t/3600:.2f} hours\n")
                    f.write(f"    Degradation Time: {deg_t/3600:.2f} hours\n")
                    
                    # Write iteration-wise results
                    if i < len(all_iteration_accuracies):
                        f.write(f"    Iteration-wise Accuracies: {[f'{x:.2f}' for x in all_iteration_accuracies[i]]}\n")
                        f.write(f"    Iteration-wise Times (hours): {[f'{x/3600:.3f}' for x in all_iteration_times[i]]}\n")
                    f.write("\n")
        
        print(f"\n{'*'*100}")
        print(f"COMPLETED: {experiment['name']}")
        print(f"Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%")
        print(f"Total Time: {mean_total_time/60:.2f} ± {std_total_time/60:.2f} minutes")
        print(f"Burn-in Time: {mean_burnin_time/60:.2f} minutes")
        print(f"Degradation Time: {mean_degradation_time/60:.2f} minutes")
        print(f"{'*'*100}\n")
    
    # Save comprehensive results across all experiments
    with open("./results/all_experiments_summary.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Print final comparison
    print(f"\n{'='*100}")
    print("ALL EXPERIMENTS COMPLETE - FINAL COMPARISON")
    print(f"{'='*100}\n")
    
    for exp_name, results in all_results.items():
        print(f"{exp_name}:")
        print(f"  Accuracy: {results['accuracy_mean_std']}")
        print(f"  Total Time: {results['total_time_hours']}")
        print(f"  Burn-in Time: {results['burnin_time_hours']}")
        print(f"  Degradation Time: {results['degradation_time_hours']}")
        print()

# Function to run a specific experiment
def run_specific_experiment(exp_number, data_dir='./tiny-imagenet-200'):
    """Run a specific experiment (1, 2, 3, or 4)"""
    experiments = [
        {
            'name': 'Exp 1 - Low Confidence 4% + Low Confidence 5%',
            'initial_selection': 'low',
            'degradation_selection': 'low'
        },
        {
            'name': 'Exp 2 - High Confidence 4% + Low Confidence 5%',
            'initial_selection': 'high',
            'degradation_selection': 'low'
        },
        {
            'name': 'Exp 3 - High Confidence 4% + High Confidence 5%',
            'initial_selection': 'high',
            'degradation_selection': 'high'
        },
        {
            'name': 'Exp 4 - Low Confidence 4% + High Confidence 5%',
            'initial_selection': 'low',
            'degradation_selection': 'high'
        }
    ]
    
    if exp_number < 1 or exp_number > 4:
        print("Invalid experiment number. Choose 1, 2, 3, or 4.")
        return
    
    experiment = experiments[exp_number - 1]
    seeds = [42, 789, 101112]
    
    print(f"\n{'='*100}")
    print(f"RUNNING SPECIFIC EXPERIMENT: {experiment['name']}")
    print(f"{'='*100}")
    
    exp_accuracies = []
    exp_total_times = []
    exp_burnin_times = []
    exp_degradation_times = []
    
    for seed_idx, seed in enumerate(seeds):
        accuracy, total_time, burnin_time, degradation_time, _, _ = run_experiment(
            seed_idx, seed, experiment, data_dir
        )
        exp_accuracies.append(accuracy)
        exp_total_times.append(total_time)
        exp_burnin_times.append(burnin_time)
        exp_degradation_times.append(degradation_time)
    
    # Calculate statistics
    if exp_accuracies and exp_total_times:
        mean_accuracy = np.mean(exp_accuracies)
        std_accuracy = np.std(exp_accuracies)
        mean_total_time = np.mean(exp_total_times)
        mean_burnin_time = np.mean(exp_burnin_times)
        mean_degradation_time = np.mean(exp_degradation_times)
        
        print(f"\n{'='*100}")
        print(f"{experiment['name']} - FINAL RESULTS")
        print(f"{'='*100}")
        print(f"\nConfiguration:")
        print(f"  Initial 4%: {experiment['initial_selection']}-confidence")
        print(f"  Degradation batches (5%): {experiment['degradation_selection']}-confidence")
        print(f"\nResults across {len(seeds)} seeds:")
        print(f"  Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%")
        print(f"  Total Time: {mean_total_time/3600:.2f} hours ({mean_total_time/60:.2f} minutes)")
        print(f"  Burn-in Time: {mean_burnin_time/3600:.2f} hours ({mean_burnin_time/60:.2f} minutes)")
        print(f"  Degradation Time: {mean_degradation_time/3600:.2f} hours ({mean_degradation_time/60:.2f} minutes)")
        print(f"\nDetailed Results:")
        for i, (seed, acc, total_t, burnin_t, deg_t) in enumerate(zip(seeds, exp_accuracies, 
                                                                      exp_total_times, exp_burnin_times, 
                                                                      exp_degradation_times)):
            print(f"  Seed {seed}:")
            print(f"    Accuracy: {acc:.2f}%")
            print(f"    Total Time: {total_t/3600:.2f} hours")
            print(f"    Burn-in Time: {burnin_t/3600:.2f} hours")
            print(f"    Degradation Time: {deg_t/3600:.2f} hours")

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ResNet18 experiments on TinyImageNet')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3, 4], 
                       help='Run specific experiment (1, 2, 3, or 4)')
    parser.add_argument('--all', action='store_true', 
                       help='Run all 4 experiments (default)')
    parser.add_argument('--data-dir', type=str, default='./tiny-imagenet-200',
                       help='Path to TinyImageNet dataset directory')
    
    args = parser.parse_args()
    
    if args.exp:
        run_specific_experiment(args.exp, args.data_dir)
    else:
        # Default: run all experiments
        run_all_experiments(args.data_dir)