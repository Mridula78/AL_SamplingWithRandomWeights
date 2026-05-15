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

# Define DenseNet121 model
class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=16).to(device)

# Data preprocessing for SVHN
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
])

# Load SVHN dataset
def load_data():
    train_set = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    test_set = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
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

def train_test_save(train_dataset, test_set, n, epochs, path, lr=0.01):
    best_accuracy = 0.0
    total_time = 0.0
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    
    for iteration in range(n):
        print(f"\nStarting training iteration {iteration + 1}/{n}")
        model = DenseNet121().to(device)
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
            
            best_model = DenseNet121().to(device)
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))
            
            print(f"Loaded the model with the highest accuracy: {best_model_path}")
            return best_model
        except Exception as e:
            print(f"Error loading the model: {e}")
            return None

def train_until_degradation(model, train_dataset, remaining_set, test_set, path, 
                           batch_size=5000, selection_type='low'):
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

def run_experiment(seed_idx, seed, experiment_config):
    """
    Run a complete experiment with a given seed and configuration
    
    Args:
        experiment_config: dictionary with keys:
            - name: experiment name
            - initial_selection: 'low' or 'high' for initial 10k
            - degradation_selection: 'low' or 'high' for 5k batches
    """
    print(f"\n{'='*80}")
    print(f"Running {experiment_config['name']}")
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
    path = f"./results_svhn/{exp_name}/seed_{seed}"
    os.makedirs(path, exist_ok=True)
    
    # Save experiment configuration
    with open(f"{path}/config.json", "w") as f:
        json.dump(experiment_config, f, indent=4)
    
    # Load data
    train_set, test_set = load_data()
    
    # Initialize model
    model = DenseNet121().to(device)
    
    # Track total experiment time
    experiment_start_time = time.time()
    
    # Step 1: Get initial dataset (10k samples)
    print(f"\nStep 1: Getting initial {experiment_config['initial_selection']}-confidence dataset...")
    initial_trainset, remainder = get_samples_by_confidence(model, train_set, 
                                                           k=10000, 
                                                           selection_type=experiment_config['initial_selection'])
    print(f"Initial training set size: {len(initial_trainset)}")
    print(f"Remaining set size: {len(remainder)}")
    
    # Step 2: Train initial model
    print("\nStep 2: Training initial model...")
    initial_accuracy, initial_time = train_test_save(initial_trainset, test_set, 1, 100, path, lr=0.01)
    
    # Step 3: Load best model
    print("\nStep 3: Loading best model...")
    model = load_best_model_from_folder(path)
    if model is None:
        print("Failed to load model. Skipping this run.")
        return 0.0, 0.0
    
    # Step 4: Train until degradation
    print(f"\nStep 4: Training until degradation ({experiment_config['degradation_selection']}-confidence)...")
    final_model, final_accuracy, degradation_time = train_until_degradation(
        model, initial_trainset, remainder, test_set, path, 
        batch_size=5000, selection_type=experiment_config['degradation_selection']
    )
    
    # Calculate total experiment time
    total_experiment_time = time.time() - experiment_start_time
    
    print(f"\n{experiment_config['name']} (Seed: {seed}) Summary:")
    print(f"  - Initial training time: {initial_time:.2f}s")
    print(f"  - Degradation training time: {degradation_time:.2f}s")
    print(f"  - Total experiment time: {total_experiment_time:.2f}s")
    print(f"  - Initial accuracy: {initial_accuracy:.2f}%")
    print(f"  - Final accuracy: {final_accuracy:.2f}%")
    
    return final_accuracy, total_experiment_time

def run_all_experiments():
    """Run all 4 experiments with"""
    # Create results directory
    os.makedirs("./results_svhn", exist_ok=True)
    
    # Define 5 different seeds
    seeds = [42, 789, 101112]
    
    # Define all 4 experiments
    experiments = [
        {
            'name': 'Exp 1 - Low Confidence 10k + Low Confidence 5k',
            'initial_selection': 'low',
            'degradation_selection': 'low'
        },
        {
            'name': 'Exp 2 - High Confidence 10k + Low Confidence 5k',
            'initial_selection': 'high',
            'degradation_selection': 'low'
        },
        {
            'name': 'Exp 3 - High Confidence 10k + High Confidence 5k',
            'initial_selection': 'high',
            'degradation_selection': 'high'
        },
        {
            'name': 'Exp 4 - Low Confidence 10k + High Confidence 5k',
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
        exp_times = []
        exp_summary = {}
        
        # Run with 5 different seeds
        for seed_idx, seed in enumerate(seeds):
            accuracy, exp_time = run_experiment(seed_idx, seed, experiment)
            exp_accuracies.append(accuracy)
            exp_times.append(exp_time)
            exp_summary[f"seed_{seed}"] = {
                "accuracy": accuracy,
                "time_hours": exp_time / 3600
            }
            
            # Save intermediate results for this experiment
            exp_dir = f"./results_svhn/{experiment['name'].replace(' ', '_').lower()}"
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
                'name': experiment['name'],
                'seeds': seeds,
                'accuracy_format': f"{mean_accuracy:.2f} ± {std_accuracy:.2f}%",
                 'time_format_hours': f"{mean_time/3600:.2f} ± {std_time/3600:.2f} hours"
            }
            
            all_results[experiment['name']] = exp_results
            
            # Save results for this experiment
            with open(f"{exp_dir}/final_results.json", "w") as f:
                json.dump(exp_results, f, indent=4)
            
            # Save a simple text summary for this experiment
            with open(f"{exp_dir}/summary.txt", "w") as f:
                f.write(f"{experiment['name']} - FINAL RESULTS SUMMARY\n")
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
        print(f"COMPLETED: {experiment['name']}")
        print(f"Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%")
        print(f"Time: {mean_time/60:.2f} ± {std_time/60:.2f} minutes")
        print(f"{'*'*100}\n")
    
    # Save comprehensive results across all experiments
    with open("./results_svhn/all_experiments_summary.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Print final comparison
    print(f"\n{'='*100}")
    print("ALL EXPERIMENTS COMPLETE - FINAL COMPARISON")
    print(f"{'='*100}\n")
    
    for exp_name, results in all_results.items():
        print(f"{exp_name}:")
        print(f"  Accuracy: {results['accuracy_format']}")
        print(f"  Time: {results['time_format_hours']}")
        print()

def run_specific_experiment(exp_number):
    """Run a specific experiment (1, 2, 3, or 4)"""
    experiments = [
        {
            'name': 'Exp 1 - Low Confidence 10k + Low Confidence 5k',
            'initial_selection': 'low',
            'degradation_selection': 'low'
        },
        {
            'name': 'Exp 2 - High Confidence 10k + Low Confidence 5k',
            'initial_selection': 'high',
            'degradation_selection': 'low'
        },
        {
            'name': 'Exp 3 - High Confidence 10k + High Confidence 5k',
            'initial_selection': 'high',
            'degradation_selection': 'high'
        },
        {
            'name': 'Exp 4 - Low Confidence 10k + High Confidence 5k',
            'initial_selection': 'low',
            'degradation_selection': 'high'
        }
    ]
    
    if exp_number < 1 or exp_number > 4:
        print("Invalid experiment number. Choose 1, 2, 3, or 4.")
        return
    
    experiment = experiments[exp_number - 1]
    seeds = [42, 456, 101112]
    
    print(f"\n{'='*100}")
    print(f"RUNNING SPECIFIC EXPERIMENT: {experiment['name']}")
    print(f"{'='*100}")
    
    exp_accuracies = []
    exp_times = []
    
    for seed_idx, seed in enumerate(seeds):
        accuracy, exp_time = run_experiment(seed_idx, seed, experiment)
        exp_accuracies.append(accuracy)
        exp_times.append(exp_time)
    
    # Calculate statistics
    if exp_accuracies and exp_times:
        mean_accuracy = np.mean(exp_accuracies)
        std_accuracy = np.std(exp_accuracies)
        mean_time = np.mean(exp_times)
        std_time = np.std(exp_times)
        
        print(f"\n{'='*100}")
        print(f"{experiment['name']} - FINAL RESULTS")
        print(f"{'='*100}")
        print(f"\nConfiguration:")
        print(f"  Initial 10k: {experiment['initial_selection']}-confidence")
        print(f"  Degradation batches (5k): {experiment['degradation_selection']}-confidence")
        print(f"\nResults:")
        print(f"  Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%")
        print(f"  Time: {mean_time/60:.2f} ± {std_time/60:.2f} minutes")
        print(f"\nDetailed Results:")
        for i, (seed, acc, t) in enumerate(zip(seeds, exp_accuracies, exp_times)):
            print(f"  Seed {seed}: Accuracy = {acc:.2f}%, Time = {t/60:.2f} minutes")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run DenseNet121 experiments on SVHN')
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