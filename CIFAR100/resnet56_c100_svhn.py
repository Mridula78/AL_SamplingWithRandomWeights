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
from torch.utils.tensorboard import SummaryWriter
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
print(device)

# Define Resnet model

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes).to(device)


def resnet110(num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)


def resnet1202(num_classes=10):
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes)

# CIFAR-100 Data preprocessing
cifar100_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

cifar100_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# SVHN Data preprocessing
svhn_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
])

svhn_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
])

def load_cifar100_data():
    train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=cifar100_transform_train)
    test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=cifar100_transform_test)
    print(f"CIFAR-100 - Size of train_set: {len(train_set)}")
    print(f"CIFAR-100 - Size of test_set: {len(test_set)}")
    return train_set, test_set

def load_svhn_data():
    train_set = datasets.SVHN(root='./data', split='train', download=True, transform=svhn_transform_train)
    test_set = datasets.SVHN(root='./data', split='test', download=True, transform=svhn_transform_test)
    print(f"SVHN - Size of train_set: {len(train_set)}")
    print(f"SVHN - Size of test_set: {len(test_set)}")
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

# Function to train the model with timing
def train_model(model, train_loader, test_loader, epochs, learning_rate, path, accumulate_steps=1):
    best_accuracy = test_model(model, test_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    total_train_time = 0.0

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
        
        scheduler.step()
        new_accuracy = test_model(model, test_loader)
        print(f"epoch: {epoch}, new accuracy: {new_accuracy:.2f}, loss: {loss.item():.4f}, epoch time: {epoch_time:.2f}s")
        
        # Check if the current accuracy is higher than the best
        if new_accuracy > best_accuracy:
            model_path = f"{path}/model_{best_accuracy:.2f}.pt"
            if os.path.exists(model_path):
                os.remove(model_path)
            best_accuracy = new_accuracy
            torch.save(model.state_dict(), f"{path}/model_{best_accuracy:.2f}.pt")
    
    print(f"Total training time for this phase: {total_train_time:.2f}s")
    return best_accuracy, total_train_time

def train_test_save(train_dataset, test_set, n, epochs, path, lr=0.01, num_classes=100):
    best_accuracy = 0.0
    total_time = 0.0
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    
    for iteration in range(n):
        print(f"\nStarting training iteration {iteration + 1}/{n}")
        model = resnet56(num_classes=num_classes).to(device)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        new_accuracy, iter_time = train_model(model, train_loader, test_loader, epochs, lr, path)
        total_time += iter_time

        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
    
    print(f"\nTotal training time for train_test_save: {total_time:.2f}s")
    print(f"Best accuracy: {best_accuracy:.2f}")
    return best_accuracy, total_time

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
    # print(f"Confidence range: min={min(confidence_scores):.4f}, max={max(confidence_scores):.4f}")
    
    return selected_samples, remainder_dataset

def load_best_model_from_folder(folder_path, num_classes=100):
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
            
            best_model = resnet56(num_classes=num_classes).to(device)
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))
            
            print(f"Loaded the model with the highest accuracy: {best_model_path}")
            return best_model
        except Exception as e:
            print(f"Error loading the model: {e}")
            return None

def train_until_degradation(model, train_dataset, remaining_set, test_set, path, 
                           batch_size=5000, selection_type='low', num_classes=100):
    """
    Train model until all data is consumed
    
    Args:
        batch_size: number of samples to add each iteration (k=5000)
        selection_type: 'low' for low confidence, 'high' for high confidence
    """
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
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
        
        # Train for 200 epochs
        current_accuracy, iter_time = train_model(model, train_loader, test_loader, 100, 0.01, path)
        
        iteration_time = time.time() - iteration_start_time
        iteration_times.append(iteration_time)
        total_iteration_time += iteration_time
        
        accuracies.append(current_accuracy)
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
        
        print(f"Iteration {iteration} complete:")
        print(f"  - Current accuracy: {current_accuracy:.2f}")
        print(f"  - Best accuracy: {best_accuracy:.2f}")
        print(f"  - Iteration time: {iteration_time:.2f}s")
        print(f"  - Remaining samples: {len(remaining_set)}")
        print(f"  - Training set size: {len(train_dataset)}")
        
        iteration += 1
        
    print(f"\nTraining dataset exhausted. Stopping degradation training.")
    print(f"Total degradation training time: {total_iteration_time:.2f}s")
    print(f"Final best accuracy: {best_accuracy:.2f}")

    # Save accuracies and times
    with open(f"{path}/accuracies.txt", "w") as file:
        file.write("Iteration,Accuracy,Time(s)\n")
        for i, (acc, t) in enumerate(zip(accuracies, iteration_times), start=1):
            file.write(f"{i},{acc:.2f},{t:.2f}\n")
        file.write(f"\nTotal degradation time: {total_iteration_time:.2f}s\n")
        file.write(f"Final best accuracy: {best_accuracy:.2f}\n")

    torch.save(model.state_dict(), f"{path}/finalmodel_{best_accuracy:.2f}.pt")
    return model, best_accuracy, total_iteration_time

def run_experiment(dataset_name, seed_idx, seed, experiment_config):
    """
    Run a complete experiment with a given seed and configuration
    
    Args:
        dataset_name: 'cifar100' or 'svhn'
        experiment_config: dictionary with keys:
            - name: experiment name
            - initial_selection: 'low' or 'high' for initial 10k
            - degradation_selection: 'low' or 'high' for 5k batches
    """
    print(f"\n{'='*80}")
    print(f"Running {experiment_config['name']} on {dataset_name.upper()}")
    print(f"Experiment {seed_idx+1} with seed: {seed}")
    print(f"Initial 10k: {experiment_config['initial_selection']}-confidence")
    print(f"Degradation 5k: {experiment_config['degradation_selection']}-confidence")
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
    exp_name = experiment_config['name'].replace(" ", "_").lower()
    path = f"./results/{dataset_name}/{exp_name}/seed_{seed}"
    os.makedirs(path, exist_ok=True)
    
    # Save experiment configuration
    with open(f"{path}/config.json", "w") as f:
        json.dump(experiment_config, f, indent=4)
    
    # Load data based on dataset
    if dataset_name == 'cifar100':
        train_set, test_set = load_cifar100_data()
        num_classes = 100
        initial_k = 10000  # 10k samples
        degradation_batch = 5000  # 5k batches
    elif dataset_name == 'svhn':
        train_set, test_set = load_svhn_data()
        num_classes = 10
        # For SVHN, adjust batch sizes since it has different dataset size
        initial_k = min(10000, len(train_set))
        degradation_batch = min(5000, len(train_set) - initial_k)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Initialize model with appropriate number of classes
    model = resnet56(num_classes=num_classes).to(device)
    
    # Track total experiment time
    experiment_start_time = time.time()
    
    # Step 1: Get initial dataset (10k samples or adjusted for dataset size)
    print(f"\nStep 1: Getting initial {experiment_config['initial_selection']}-confidence dataset...")
    initial_trainset, remainder = get_samples_by_confidence(model, train_set, 
                                                           k=initial_k, 
                                                           selection_type=experiment_config['initial_selection'])
    print(f"Initial training set size: {len(initial_trainset)}")
    print(f"Remaining set size: {len(remainder)}")
    
    # Step 2: Train initial model
    print("\nStep 2: Training initial model...")
    initial_accuracy, initial_time = train_test_save(initial_trainset, test_set, 1, 100, path, 
                                                     lr=0.01, num_classes=num_classes)
    
    # Step 3: Load best model
    print("\nStep 3: Loading best model...")
    model = load_best_model_from_folder(path, num_classes=num_classes)
    if model is None:
        print("Failed to load model. Skipping this run.")
        return 0.0, 0.0
    
    # Step 4: Train until degradation
    print(f"\nStep 4: Training until degradation ({experiment_config['degradation_selection']}-confidence)...")
    final_model, final_accuracy, degradation_time = train_until_degradation(
        model, initial_trainset, remainder, test_set, path, 
        batch_size=degradation_batch, selection_type=experiment_config['degradation_selection'],
        num_classes=num_classes
    )
    
    # Calculate total experiment time
    total_experiment_time = time.time() - experiment_start_time
    
    print(f"\n{experiment_config['name']} on {dataset_name.upper()} (Seed: {seed}) Summary:")
    print(f"  - Initial training time: {initial_time:.2f}s")
    print(f"  - Degradation training time: {degradation_time:.2f}s")
    print(f"  - Total experiment time: {total_experiment_time:.2f}s")
    print(f"  - Initial accuracy: {initial_accuracy:.2f}%")
    print(f"  - Final accuracy: {final_accuracy:.2f}%")
    
    return final_accuracy, total_experiment_time

def run_dataset_experiments(dataset_name, exp_number=None):
    """Run experiments for a specific dataset (CIFAR-100 or SVHN)"""
    # Create results directory for the dataset
    os.makedirs(f"./results/{dataset_name}", exist_ok=True)
    
    # Define seeds
    seeds = [42, 789, 101112]
    
    # Define experiment configuration (only one experiment)
    experiment = {
        'name': 'Low Confidence 10k + Low Confidence 5k',
        'initial_selection': 'low',
        'degradation_selection': 'low'
    }
    
    print(f"\n{'*'*100}")
    print(f"RUNNING EXPERIMENT ON {dataset_name.upper()}")
    print(f"Configuration: {experiment['name']}")
    print(f"{'*'*100}")
    
    exp_accuracies = []
    exp_times = []
    exp_summary = {}
    
    # Run with different seeds
    for seed_idx, seed in enumerate(seeds):
        accuracy, exp_time = run_experiment(dataset_name, seed_idx, seed, experiment)
        exp_accuracies.append(accuracy)
        exp_times.append(exp_time)
        exp_summary[f"seed_{seed}"] = {
            "accuracy": accuracy,
            "time_hours": exp_time / 3600
        }
        
        # Save intermediate results for this experiment
        exp_dir = f"./results/{dataset_name}/{experiment['name'].replace(' ', '_').lower()}"
        os.makedirs(exp_dir, exist_ok=True)
        with open(f"{exp_dir}/results_summary.json", "w") as f:
            json.dump(exp_summary, f, indent=4)
    
    # Calculate statistics for this experiment
    if exp_accuracies and exp_times:
        mean_accuracy = np.mean(exp_accuracies)
        std_accuracy = np.std(exp_accuracies)
        mean_time = np.mean(exp_times)
        std_time = np.std(exp_times)
        
        exp_results = {
            'dataset': dataset_name,
            'name': experiment['name'],
            'seeds': seeds,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_time': mean_time,
            'std_time': std_time,
            'accuracy_format': f"{mean_accuracy:.2f} ± {std_accuracy:.2f}%",
            'time_format_hours': f"{mean_time/3600:.2f} ± {std_time/3600:.2f} hours"
        }
        
        # Save results for this experiment
        with open(f"{exp_dir}/final_results.json", "w") as f:
            json.dump(exp_results, f, indent=4)
        
        # Save a simple text summary for this experiment
        with open(f"{exp_dir}/summary.txt", "w") as f:
            f.write(f"{experiment['name']} on {dataset_name.upper()} - FINAL RESULTS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Initial 10k: {experiment['initial_selection']}-confidence\n")
            f.write(f"  Degradation batches (5k): {experiment['degradation_selection']}-confidence\n\n")
            f.write("Accuracy Results:\n")
            f.write(f"  Mean ± Std: {mean_accuracy:.2f} ± {std_accuracy:.2f}%\n\n")
            f.write("Training Time Results:\n")
            f.write(f"  Mean ± Std: {mean_time/3600:.2f} ± {std_time/3600:.2f} hours\n\n")
            
            f.write("Detailed Results:\n")
            for i, (seed, acc, t) in enumerate(zip(seeds, exp_accuracies, exp_times)):
                f.write(f"  Seed {seed}:\n")
                f.write(f"    Accuracy: {acc:.2f}%\n")
                f.write(f"    Time: {t:.2f}s ({t/60:.2f} minutes)\n")
    
    print(f"\n{'*'*100}")
    print(f"COMPLETED: {experiment['name']} on {dataset_name.upper()}")
    print(f"Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%")
    print(f"Time: {mean_time/60:.2f} ± {std_time/60:.2f} minutes")
    print(f"{'*'*100}\n")
    
    return mean_accuracy, std_accuracy, mean_time, std_time

def run_all_datasets():
    """Run the experiment on both CIFAR-100 and SVHN"""
    print(f"\n{'='*100}")
    print("RUNNING EXPERIMENTS ON CIFAR-100 AND SVHN")
    print(f"{'='*100}\n")
    
    datasets = ['cifar100', 'svhn']
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"STARTING {dataset.upper()}")
        print(f"{'='*80}")
        
        mean_acc, std_acc, mean_time, std_time = run_dataset_experiments(dataset)
        
        all_results[dataset] = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_time_hours': mean_time / 3600,
            'std_time_hours': std_time / 3600,
            'accuracy_format': f"{mean_acc:.2f} ± {std_acc:.2f}%",
            'time_format': f"{mean_time/60:.2f} ± {std_time/60:.2f} minutes"
        }
    
    # Save comprehensive results across all datasets
    with open("./results/all_datasets_summary.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Print final comparison
    print(f"\n{'='*100}")
    print("ALL DATASETS COMPLETE - FINAL COMPARISON")
    print(f"{'='*100}\n")
    
    for dataset, results in all_results.items():
        print(f"{dataset.upper()}:")
        print(f"  Accuracy: {results['accuracy_format']}")
        print(f"  Time: {results['time_format']}")
        print()

def run_specific_dataset(dataset_name):
    """Run experiment on a specific dataset"""
    if dataset_name not in ['cifar100', 'svhn']:
        print("Invalid dataset name. Choose 'cifar100' or 'svhn'.")
        return
    
    print(f"\n{'='*100}")
    print(f"RUNNING EXPERIMENT ON {dataset_name.upper()}")
    print(f"{'='*100}")
    
    mean_acc, std_acc, mean_time, std_time = run_dataset_experiments(dataset_name)
    
    print(f"\n{'='*100}")
    print(f"{dataset_name.upper()} - FINAL RESULTS")
    print(f"{'='*100}")
    print(f"\nResults:")
    print(f"  Accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
    print(f"  Time: {mean_time/60:.2f} ± {std_time/60:.2f} minutes")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run resnet56 experiments on CIFAR-100 and SVHN')
    parser.add_argument('--dataset', type=str, choices=['cifar100', 'svhn'], 
                       help='Run on specific dataset (cifar100 or svhn)')
    parser.add_argument('--all', action='store_true', 
                       help='Run on both CIFAR-100 and SVHN (default)')
    
    args = parser.parse_args()
    
    if args.dataset:
        run_specific_dataset(args.dataset)
    else:
        # Default: run on both datasets
        run_all_datasets()