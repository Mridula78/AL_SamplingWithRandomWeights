import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, datasets
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import scipy.io
import numpy as np
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
print(f"Using device: {device}")

# Define VGG16 model for SVHN
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        # Features (conv layers)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def VGG16_SVHN():
    return VGG16(num_classes=10).to(device)

# Data preprocessing for SVHN
def get_svhn_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

# Load SVHN dataset
def load_data():
    # SVHN dataset - Street View House Numbers
    train_set = datasets.SVHN(root='./data', split='train', download=True, transform=get_svhn_transforms())
    test_set = datasets.SVHN(root='./data', split='test', download=True, transform=get_svhn_transforms())
    print(f"Size of train_set: {len(train_set)}")
    print(f"Size of test_set: {len(test_set)}")
    return train_set, test_set

# Function to test the model
def test_model(model, test_loader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct * 100 / total
    return accuracy

# Function to train the model with timing and iteration tracking
def train_model(model, train_loader, test_loader, epochs, learning_rate, path, accumulate_steps=1):
    best_accuracy = test_model(model, test_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    total_train_time = 0.0
    epoch_accuracies = []
    epoch_times = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        for i, data in enumerate(train_loader, 0):
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
        
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time
        epoch_times.append(epoch_time)
        
        scheduler.step()
        new_accuracy = test_model(model, test_loader)
        epoch_accuracies.append(new_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}: Accuracy: {new_accuracy:.2f}%, Loss: {loss.item():.4f}, Time: {epoch_time:.2f}s")
        
        # Check if the current accuracy is higher than the best
        if new_accuracy > best_accuracy:
            model_path = f"{path}/model_{best_accuracy:.2f}.pt"
            if os.path.exists(model_path):
                os.remove(model_path)
            best_accuracy = new_accuracy
            torch.save(model.state_dict(), f"{path}/model_{best_accuracy:.2f}.pt")
    
    # Save epoch-wise metrics
    with open(f"{path}/epoch_metrics.txt", "w") as f:
        f.write("Epoch,Accuracy,Time(s)\n")
        for i, (acc, t) in enumerate(zip(epoch_accuracies, epoch_times), start=1):
            f.write(f"{i},{acc:.2f},{t:.2f}\n")
        f.write(f"\nTotal training time: {total_train_time:.2f}s\n")
        f.write(f"Best accuracy: {best_accuracy:.2f}%\n")
    
    print(f"Total training time for this phase: {total_train_time:.2f}s")
    return best_accuracy, total_train_time, epoch_accuracies, epoch_times

def train_test_save(train_dataset, test_set, n, epochs, path, lr=0.01):
    best_accuracy = 0.0
    total_time = 0.0
    all_epoch_accuracies = []
    all_epoch_times = []
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    
    for iteration in range(n):
        print(f"\nStarting training iteration {iteration + 1}/{n}")
        model = VGG16_SVHN().to(device)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        new_accuracy, iter_time, epoch_accs, epoch_times = train_model(model, train_loader, test_loader, epochs, lr, path)
        total_time += iter_time
        all_epoch_accuracies.extend(epoch_accs)
        all_epoch_times.extend(epoch_times)

        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
    
    print(f"\nTotal training time for train_test_save: {total_time:.2f}s")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    return best_accuracy, total_time, all_epoch_accuracies, all_epoch_times

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
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    confidence_scores = []

    with torch.no_grad():
        for inputs, _ in data_loader:
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
    
    selected_samples = torch.utils.data.Subset(dataset, selected_indices)
    remaining_indices = [i for i in range(len(dataset)) if i not in selected_indices]
    remainder_dataset = torch.utils.data.Subset(dataset, remaining_indices)
    
    print(f"Selected {k} {selection_type}-confidence samples from {len(dataset)} total samples")
    print(f"Confidence range: min={min(confidence_scores):.4f}, max={max(confidence_scores):.4f}")
    
    return selected_samples, remainder_dataset

def load_best_model_from_folder(folder_path):
    def get_accuracy_from_filename(filename):
        try:
            return float(filename.split("_")[1][:-3])
        except:
            return 0.0

    model_files = [file for file in os.listdir(folder_path) if file.startswith("model_") and file.endswith(".pt")]

    if not model_files:
        print("No model files found in the folder.")
        return None
    else:
        try:
            best_model_filename = max(model_files, key=get_accuracy_from_filename)
            best_model_path = os.path.join(folder_path, best_model_filename)
            
            best_model = VGG16_SVHN().to(device)
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))
            
            print(f"Loaded the model with the highest accuracy: {best_model_path}")
            return best_model
        except Exception as e:
            print(f"Error loading the model: {e}")
            return None

def train_until_degradation(model, train_dataset, remaining_set, test_set, path, 
                           batch_size=5000, selection_type='low', initial_model_training=False):
    """
    Train model until all data is consumed
    
    Args:
        batch_size: number of samples to add each iteration
        selection_type: 'low' for low confidence, 'high' for high confidence
        initial_model_training: if True, this is the initial burn-in phase
    """
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    
    if initial_model_training:
        print(f"\nStarting BURN-IN phase with {len(train_dataset)} samples")
        # For burn-in phase, just train on initial dataset
        current_accuracy, total_time, epoch_accuracies, epoch_times = train_model(
            model, DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2), 
            test_loader, 100, 0.01, path
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
        best_accuracy = test_model(model, test_loader)
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
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)      
            
            # Train for 100 epochs
            current_accuracy, iter_time, epoch_accs, epoch_times = train_model(model, train_loader, test_loader, 100, 0.01, path)
            
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
            for i, (acc, t, train_size) in enumerate(zip(accuracies, iteration_times, 
                                                         range(len(train_dataset) - len(remaining_set), 
                                                               len(train_dataset) + 1, 
                                                               batch_size)), start=1):
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
            "iteration_times_hours": [t/3600 for t in iteration_times],
            "iteration_training_set_sizes": list(range(len(train_dataset) - len(remaining_set), 
                                                       len(train_dataset) + 1, 
                                                       batch_size))
        }
        
        with open(f"{path}/degradation_metrics.json", "w") as f:
            json.dump(degradation_metrics, f, indent=4)

        torch.save(model.state_dict(), f"{path}/finalmodel_{best_accuracy:.2f}.pt")
        return model, best_accuracy, total_iteration_time, accuracies, iteration_times

def run_experiment(seed_idx, seed, experiment_config):
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
    
    # Create directory for this experiment with model_data folder
    exp_name = experiment_config['name'].replace(" ", "_").lower()
    path = f"./model_data/{exp_name}/seed_{seed}"
    os.makedirs(path, exist_ok=True)
    
    # Save experiment configuration
    with open(f"{path}/config.json", "w") as f:
        json.dump(experiment_config, f, indent=4)
    
    # Load data
    train_set, test_set = load_data()
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
    model = VGG16_SVHN().to(device)
    
    # Step 1: Get initial dataset (4% samples)
    print(f"\nStep 1: Getting initial {experiment_config['initial_selection']}-confidence dataset...")
    initial_trainset, remainder = get_samples_by_confidence(model, train_set, 
                                                           k=initial_samples, 
                                                           selection_type=experiment_config['initial_selection'])
    print(f"Initial training set size: {len(initial_trainset)} ({len(initial_trainset)/total_train_samples*100:.1f}%)")
    print(f"Remaining set size: {len(remainder)} ({len(remainder)/total_train_samples*100:.1f}%)")
    
    # Step 2: Train initial model (BURN-IN PHASE)
    print(f"\nStep 2: BURN-IN PHASE - Training initial model on {len(initial_trainset)} samples...")
    burnin_accuracy, burnin_time, burnin_epoch_accuracies, burnin_epoch_times = train_test_save(
        initial_trainset, test_set, 1, 100, path, lr=0.01
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
        model, initial_trainset, remainder, test_set, path, 
        batch_size=degradation_batch_size, selection_type=experiment_config['degradation_selection']
    )
    
    # Calculate total experiment time
    total_experiment_time = time.time() - experiment_start_time
    
    # Calculate number of degradation iterations
    num_degradation_iterations = len(iteration_accuracies)
    
    # Prepare comprehensive summary for vgg16-svhn.txt
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(f"EXPERIMENT SUMMARY: {experiment_config['name']}")
    summary_lines.append(f"Model: VGG16")
    summary_lines.append(f"Dataset: SVHN")
    summary_lines.append(f"Seed: {seed}")
    summary_lines.append("=" * 80)
    
    summary_lines.append(f"\nDATASET STATISTICS:")
    summary_lines.append(f"  Total training samples: {total_train_samples}")
    summary_lines.append(f"  Initial samples (4%): {initial_samples}")
    summary_lines.append(f"  Degradation batch size (5%): {degradation_batch_size}")
    
    summary_lines.append(f"\nBURN-IN PHASE (4% data):")
    summary_lines.append(f"  Samples: {len(initial_trainset)}")
    summary_lines.append(f"  Accuracy: {burnin_accuracy:.2f}%")
    summary_lines.append(f"  Time: {burnin_time:.2f}s ({burnin_time/60:.2f} minutes, {burnin_time/3600:.2f} hours)")
    
    summary_lines.append(f"\nDEGRADATION PHASE (5% per iteration):")
    summary_lines.append(f"  Batch size: {degradation_batch_size} samples")
    summary_lines.append(f"  Number of iterations: {num_degradation_iterations}")
    summary_lines.append(f"  Final accuracy: {final_accuracy:.2f}%")
    summary_lines.append(f"  Total degradation time: {degradation_time:.2f}s ({degradation_time/60:.2f} minutes, {degradation_time/3600:.2f} hours)")
    
    summary_lines.append(f"\nITERATION-WISE RESULTS:")
    for i, (acc, t) in enumerate(zip(iteration_accuracies, iteration_times), start=1):
        summary_lines.append(f"  Iteration {i}: Accuracy = {acc:.2f}%, Time = {t:.2f}s ({t/60:.2f} minutes)")
    
    summary_lines.append(f"\nOVERALL EXPERIMENT:")
    summary_lines.append(f"  Total time: {total_experiment_time:.2f}s ({total_experiment_time/60:.2f} minutes, {total_experiment_time/3600:.2f} hours)")
    summary_lines.append(f"  Accuracy improvement: {final_accuracy - burnin_accuracy:.2f}%")
    summary_lines.append("=" * 80)
    
    # Save to vgg16-svhn.txt
    with open(f"{path}/vgg16-svhn.txt", "w") as f:
        f.write("\n".join(summary_lines))
    
    # Also save JSON for machine readability
    summary_json = {
        "model": "VGG16",
        "dataset": "SVHN",
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
    
    with open(f"{path}/experiment_summary.json", "w") as f:
        json.dump(summary_json, f, indent=4)
    
    # Print summary to console
    print("\n".join(summary_lines))
    
    return final_accuracy, total_experiment_time, burnin_time, degradation_time, iteration_accuracies, iteration_times

def run_all_experiments():
    """Run all 4 experiments with multiple seeds"""
    # Create model_data directory
    os.makedirs("./model_data", exist_ok=True)
    
    # Define seeds
    seeds = [42, 789, 101112]
    
    # Define all 4 experiments with updated percentages
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
                seed_idx, seed, experiment
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
            exp_dir = f"./model_data/{experiment['name'].replace(' ', '_').lower()}"
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
                'model': 'VGG16',
                'dataset': 'SVHN',
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
            
            # Save vgg16-svhn.txt summary
            summary_lines = []
            summary_lines.append("=" * 100)
            summary_lines.append(f"FINAL RESULTS SUMMARY: {experiment['name']}")
            summary_lines.append(f"Model: VGG16")
            summary_lines.append(f"Dataset: SVHN")
            summary_lines.append("=" * 100)
            summary_lines.append(f"\nConfiguration:")
            summary_lines.append(f"  Initial 4%: {experiment['initial_selection']}-confidence")
            summary_lines.append(f"  Degradation batches (5%): {experiment['degradation_selection']}-confidence")
            summary_lines.append(f"\nStatistics across {len(seeds)} seeds:")
            summary_lines.append(f"  Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%")
            summary_lines.append(f"  Total time: {mean_total_time/3600:.2f} ± {std_total_time/3600:.2f} hours")
            summary_lines.append(f"  Burn-in time: {mean_burnin_time/3600:.2f} hours")
            summary_lines.append(f"  Degradation time: {mean_degradation_time/3600:.2f} hours")
            
            summary_lines.append(f"\nDetailed Results per Seed:")
            for i, (seed, acc, total_t, burnin_t, deg_t) in enumerate(zip(seeds, exp_accuracies, 
                                                                          exp_total_times, exp_burnin_times, 
                                                                          exp_degradation_times)):
                summary_lines.append(f"  Seed {seed}:")
                summary_lines.append(f"    Final Accuracy: {acc:.2f}%")
                summary_lines.append(f"    Total Time: {total_t/3600:.2f} hours")
                summary_lines.append(f"    Burn-in Time: {burnin_t/3600:.2f} hours")
                summary_lines.append(f"    Degradation Time: {deg_t/3600:.2f} hours")
                
                # Write iteration-wise results
                if i < len(all_iteration_accuracies):
                    summary_lines.append(f"    Iteration-wise Accuracies: {[f'{x:.2f}' for x in all_iteration_accuracies[i]]}")
                    summary_lines.append(f"    Iteration-wise Times (hours): {[f'{x/3600:.3f}' for x in all_iteration_times[i]]}")
                summary_lines.append("")
            
            with open(f"{exp_dir}/vgg16-svhn.txt", "w") as f:
                f.write("\n".join(summary_lines))
            
            print(f"\n{'*'*100}")
            print(f"COMPLETED: {experiment['name']}")
            print(f"Model: VGG16, Dataset: SVHN")
            print(f"Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%")
            print(f"Total Time: {mean_total_time/60:.2f} ± {std_total_time/60:.2f} minutes")
            print(f"Burn-in Time: {mean_burnin_time/60:.2f} minutes")
            print(f"Degradation Time: {mean_degradation_time/60:.2f} minutes")
            print(f"{'*'*100}\n")
    
    # Save comprehensive results across all experiments
    with open("./model_data/all_experiments_summary.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Create a master vgg16-svhn.txt file
    master_lines = []
    master_lines.append("=" * 120)
    master_lines.append("ALL EXPERIMENTS COMPLETE - FINAL COMPARISON")
    master_lines.append(f"Model: VGG16")
    master_lines.append(f"Dataset: SVHN")
    master_lines.append("=" * 120)
    
    for exp_name, results in all_results.items():
        master_lines.append(f"\n{exp_name}:")
        master_lines.append(f"  Accuracy: {results['accuracy_mean_std']}")
        master_lines.append(f"  Total Time: {results['total_time_hours']}")
        master_lines.append(f"  Burn-in Time: {results['burnin_time_hours']}")
        master_lines.append(f"  Degradation Time: {results['degradation_time_hours']}")
    
    master_lines.append("\n" + "=" * 120)
    
    with open("./model_data/vgg16-svhn.txt", "w") as f:
        f.write("\n".join(master_lines))
    
    # Print final comparison
    print("\n".join(master_lines))

def run_specific_experiment(exp_number):
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
    print(f"Model: VGG16")
    print(f"Dataset: SVHN")
    print(f"{'='*100}")
    
    exp_accuracies = []
    exp_total_times = []
    exp_burnin_times = []
    exp_degradation_times = []
    
    for seed_idx, seed in enumerate(seeds):
        accuracy, total_time, burnin_time, degradation_time, _, _ = run_experiment(
            seed_idx, seed, experiment
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
        print(f"Model: VGG16")
        print(f"Dataset: SVHN")
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run VGG16 experiments on SVHN')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3, 4], 
                       help='Run specific experiment (1, 2, 3, or 4)')
    parser.add_argument('--all', action='store_true', 
                       help='Run all 4 experiments (default)')
    
    args = parser.parse_args()
    
    if args.exp:
        run_specific_experiment(args.exp)
    else:
        # Default: run all experiments
        run_all_experiments()