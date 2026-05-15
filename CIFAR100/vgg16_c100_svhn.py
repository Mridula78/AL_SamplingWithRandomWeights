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

# Define VGG model
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# Function to get dataset-specific transforms and data loader
def get_dataset_config(dataset_name):
    """Return dataset-specific configuration"""
    if dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        def load_data():
            train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            print(f"Size of train_set: {len(train_set)}")
            print(f"Size of test_set: {len(test_set)}")
            return train_set, test_set
        
        num_classes = 10
        model_name = 'VGG16'
        
    elif dataset_name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        def load_data():
            train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
            test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
            print(f"Size of train_set: {len(train_set)}")
            print(f"Size of test_set: {len(test_set)}")
            return train_set, test_set
        
        num_classes = 100
        model_name = 'VGG16'
        
    elif dataset_name == 'svhn':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        
        def load_data():
            train_set = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
            test_set = datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
            print(f"Size of train_set: {len(train_set)}")
            print(f"Size of test_set: {len(test_set)}")
            return train_set, test_set
        
        num_classes = 10
        model_name = 'VGG16'
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return {
        'load_data': load_data,
        'num_classes': num_classes,
        'model_name': model_name
    }

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

def train_test_save(train_dataset, test_set, n, epochs, path, lr=0.01, num_classes=10, model_name='VGG16'):
    best_accuracy = 0.0
    total_time = 0.0
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    
    for iteration in range(n):
        print(f"\nStarting training iteration {iteration + 1}/{n}")
        model = VGG(model_name, num_classes=num_classes).to(device)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        new_accuracy, iter_time = train_model(model, train_loader, test_loader, epochs, lr, path)
        total_time += iter_time

        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
    
    print(f"\nTotal training time for train_test_save: {total_time:.2f}s")
    print(f"Best accuracy: {best_accuracy:.2f}")
    return best_accuracy, total_time

def get_lowconf_and_remainder_datasets(model, dataset, k_low):
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

    top_low_indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i], reverse=False)[:k_low]
    lowconf_samples = torch.utils.data.Subset(dataset, top_low_indices)
    remaining_indices = [i for i in range(len(dataset)) if i not in top_low_indices]
    remainder_dataset = torch.utils.data.Subset(dataset, remaining_indices)
    return lowconf_samples, remainder_dataset

def load_best_model_from_folder(folder_path, num_classes=10, model_name='VGG16'):
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
            
            best_model = VGG(model_name, num_classes=num_classes).to(device)
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))
            
            print(f"Loaded the model with the highest accuracy: {best_model_path}")
            return best_model
        except Exception as e:
            print(f"Error loading the model: {e}")
            return None

def train_until_degradation(model, train_dataset, remaining_set, test_set, path, num_classes=10, model_name='VGG16'):
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    best_accuracy = test_model(model, test_loader)
    current_accuracy = 0.0
    accuracies = []
    iteration_times = []
    iteration = 1
    total_iteration_time = 0.0

    print(f"\nStarting degradation training with {len(remaining_set)} remaining samples")
    
    while len(remaining_set) != 0:
        print(f"\n--- Degradation Iteration {iteration} ---")
        iteration_start_time = time.time()
        
        # Get low confidence samples
        least_conf_images, remaining_set = get_lowconf_and_remainder_datasets(model, remaining_set, k_low=5000)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, least_conf_images])   
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)      
        
        # Train for 200 epochs
        current_accuracy, iter_time = train_model(model, train_loader, test_loader, 200, 0.01, path)
        
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

def run_experiment(seed_idx, seed, dataset_name, dataset_config):
    """Run a complete experiment with a given seed and dataset"""
    print(f"\n{'='*60}")
    print(f"Running experiment {seed_idx+1} with seed: {seed} on {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create directory for this run
    path = f"./results_{dataset_name}/seed_{seed}"
    os.makedirs(path, exist_ok=True)
    
    # Load data
    load_data_func = dataset_config['load_data']
    num_classes = dataset_config['num_classes']
    model_name = dataset_config['model_name']
    
    train_set, test_set = load_data_func()
    
    # Initialize model
    model = VGG(model_name, num_classes=num_classes).to(device)
    
    # Track total experiment time
    experiment_start_time = time.time()
    
    # Step 1: Get initial low-confidence dataset
    print("\nStep 1: Getting initial low-confidence dataset...")
    initial_trainset, remainder = get_lowconf_and_remainder_datasets(model, train_set, k_low=10000)
    print(f"Initial training set size: {len(initial_trainset)}")
    print(f"Remaining set size: {len(remainder)}")
    
    # Step 2: Train initial model
    print("\nStep 2: Training initial model...")
    initial_accuracy, initial_time = train_test_save(initial_trainset, test_set, 1, 200, path, 
                                                    lr=0.01, num_classes=num_classes, model_name=model_name)
    
    # Step 3: Load best model
    print("\nStep 3: Loading best model...")
    model = load_best_model_from_folder(path, num_classes=num_classes, model_name=model_name)
    if model is None:
        print("Failed to load model. Skipping this run.")
        return 0.0, 0.0
    
    # Step 4: Train until degradation
    print("\nStep 4: Training until degradation...")
    final_model, final_accuracy, degradation_time = train_until_degradation(
        model, initial_trainset, remainder, test_set, path, 
        num_classes=num_classes, model_name=model_name
    )
    
    # Calculate total experiment time
    total_experiment_time = time.time() - experiment_start_time
    
    print(f"\nExperiment {seed_idx+1} (Seed: {seed}) on {dataset_name.upper()} Summary:")
    print(f"  - Initial training time: {initial_time:.2f}s")
    print(f"  - Degradation training time: {degradation_time:.2f}s")
    print(f"  - Total experiment time: {total_experiment_time:.2f}s")
    print(f"  - Initial accuracy: {initial_accuracy:.2f}%")
    print(f"  - Final accuracy: {final_accuracy:.2f}%")
    
    return final_accuracy, total_experiment_time

def run_dataset_experiments(dataset_name):
    """Run experiments for a specific dataset"""
    print(f"\n{'#'*100}")
    print(f"STARTING EXPERIMENTS FOR {dataset_name.upper()}")
    print(f"{'#'*100}")
    
    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    
    # Create results directory
    os.makedirs(f"./results_{dataset_name}", exist_ok=True)
    
    # Define seeds
    seeds = [42, 123, 456, 789, 101112]
    
    # Store results from all runs
    all_accuracies = []
    all_times = []
    results_summary = {}
    
    # Run experiments
    for i, seed in enumerate(seeds):
        accuracy, exp_time = run_experiment(i, seed, dataset_name, dataset_config)
        all_accuracies.append(accuracy)
        all_times.append(exp_time)
        results_summary[f"seed_{seed}"] = {
            "accuracy": accuracy,
            "time_seconds": exp_time,
            "time_minutes": exp_time / 60,
            "time_hours": exp_time / 3600
        }
        
        # Save intermediate results
        with open(f"./results_{dataset_name}/results_summary.json", "w") as f:
            json.dump(results_summary, f, indent=4)
    
    # Calculate statistics
    if all_accuracies and all_times:
        mean_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        mean_time = np.mean(all_times)
        std_time = np.std(all_times)
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS SUMMARY FOR {dataset_name.upper()} (5 runs)")
        print(f"{'='*80}")
        print("\nACCURACY RESULTS:")
        print(f"{'='*40}")
        for i, (seed, acc, t) in enumerate(zip(seeds, all_accuracies, all_times)):
            print(f"Experiment {i+1} (Seed {seed}):")
            print(f"  Accuracy: {acc:.2f}%")
            print(f"  Time: {t:.2f}s ({t/60:.2f} minutes, {t/3600:.2f} hours)")
            print()
        
        print(f"\nSTATISTICAL SUMMARY:")
        print(f"{'='*40}")
        print(f"Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%")
        print(f"Training Time: {mean_time:.2f} ± {std_time:.2f}s")
        print(f"Training Time: {mean_time/60:.2f} ± {std_time/60:.2f} minutes")
        print(f"Training Time: {mean_time/3600:.2f} ± {std_time/3600:.2f} hours")
        print(f"{'='*80}")
        
        # Save final results
        final_results = {
            "dataset": dataset_name,
            "seeds": seeds,
            "accuracies": all_accuracies,
            "times_seconds": all_times,
            "accuracy_mean": mean_accuracy,
            "accuracy_std": std_accuracy,
            "time_mean_seconds": mean_time,
            "time_std_seconds": std_time,
            "accuracy_format": f"{mean_accuracy:.2f} ± {std_accuracy:.2f}%",
            "time_format_seconds": f"{mean_time:.2f} ± {std_time:.2f}s",
            "time_format_minutes": f"{mean_time/60:.2f} ± {std_time/60:.2f} minutes",
            "time_format_hours": f"{mean_time/3600:.2f} ± {std_time/3600:.2f} hours"
        }
        
        with open(f"./results_{dataset_name}/final_results.json", "w") as f:
            json.dump(final_results, f, indent=4)
        
        # Also save a simple text summary
        with open(f"./results_{dataset_name}/summary.txt", "w") as f:
            f.write(f"FINAL RESULTS SUMMARY FOR {dataset_name.upper()}\n")
            f.write("=" * 60 + "\n\n")
            f.write("Accuracy Results:\n")
            f.write(f"Mean ± Std: {mean_accuracy:.2f} ± {std_accuracy:.2f}%\n\n")
            f.write("Training Time Results:\n")
            f.write(f"Mean ± Std: {mean_time:.2f} ± {std_time:.2f} seconds\n")
            f.write(f"Mean ± Std: {mean_time/60:.2f} ± {std_time/60:.2f} minutes\n")
            f.write(f"Mean ± Std: {mean_time/3600:.2f} ± {std_time/3600:.2f} hours\n\n")
            
            f.write("Detailed Results:\n")
            for i, (seed, acc, t) in enumerate(zip(seeds, all_accuracies, all_times)):
                f.write(f"Experiment {i+1} (Seed {seed}):\n")
                f.write(f"  Accuracy: {acc:.2f}%\n")
                f.write(f"  Time: {t:.2f}s ({t/60:.2f} minutes)\n\n")
    else:
        print(f"No valid results obtained from {dataset_name} experiments.")

def main():
    """Main function to run experiments on both CIFAR-100 and SVHN"""
    print(f"\n{'*'*120}")
    print("STARTING EXPERIMENTS FOR CIFAR-100 AND SVHN DATASETS")
    print(f"{'*'*120}")
    
    # Run experiments for both datasets
    datasets_to_run = ['cifar100', 'svhn']
    
    for dataset in datasets_to_run:
        run_dataset_experiments(dataset)
        print(f"\n{'#'*100}")
        print(f"COMPLETED EXPERIMENTS FOR {dataset.upper()}")
        print(f"{'#'*100}\n")
    
    print(f"\n{'*'*120}")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"{'*'*120}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run VGG16 experiments on CIFAR-100 and SVHN')
    parser.add_argument('--dataset', type=str, choices=['cifar100', 'svhn', 'both'], default='both',
                       help='Dataset to run experiments on (cifar100, svhn, or both)')
    
    args = parser.parse_args()
    
    if args.dataset == 'both':
        main()
    elif args.dataset in ['cifar100', 'svhn']:
        run_dataset_experiments(args.dataset)
    else:
        print(f"Invalid dataset: {args.dataset}. Choose 'cifar100', 'svhn', or 'both'")