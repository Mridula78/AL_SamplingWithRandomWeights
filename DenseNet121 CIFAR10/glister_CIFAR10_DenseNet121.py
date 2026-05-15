"""
GLISTER Active Learning for CIFAR-10 + DenseNet121 - Multiple Runs
Runs experiments 3 times and reports mean ± stddev for time and accuracy.

Dataset: CIFAR-10 (downloads automatically via torchvision)

Active Learning Setup:
- Start with small labeled pool (initial_budget)
- Each round: select budget_per_round new samples using GLISTER, add to labeled pool, retrain
- Continue until all training data is labeled
- Compare against random selection baseline
- Run 3 times and report statistics

Usage:
    python glister_cifar10_densenet121.py \
        --datadir ./data/cifar10 \
        --initial_budget 1000 \
        --budget_per_round 1000 \
        --epochs_per_round 20 \
        --max_pool_size 10000 \
        --num_runs 3
"""

# ──────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────
import argparse
import copy
import numpy as np
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import (
    DataLoader, Dataset, Subset, random_split
)

# ──────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir',           default='./data/cifar10',
                        help='Directory to download/store CIFAR-10 dataset')
    parser.add_argument('--initial_budget',    type=int,   default=1000,
                        help='Initial labeled pool size')
    parser.add_argument('--budget_per_round',  type=int,   default=1000,
                        help='Number of samples to select per round')
    parser.add_argument('--epochs_per_round',  type=int,   default=50,
                        help='Training epochs per round')
    parser.add_argument('--max_pool_size',     type=int,   default=10000,
                        help='Maximum unlabeled pool size to consider per round')
    parser.add_argument('--feature',           default='dss',
                        choices=['dss', 'noise', 'classimb'])
    parser.add_argument('--lr',                type=float, default=0.1)
    parser.add_argument('--results_dir',       default='./results')
    parser.add_argument('--num_runs',          type=int,   default=3,
                        help='Number of times to run the experiment')
    parser.add_argument('--seed',              type=int,   default=42)
    args, unknown = parser.parse_known_args()  
    return args

# ──────────────────────────────────────────────────────────────
# Seeds
# ──────────────────────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ──────────────────────────────────────────────────────────────
# DenseNet121 with embedding output
# ──────────────────────────────────────────────────────────────
class DenseNet121Emb(nn.Module):
    """DenseNet-121 that returns (logits, penultimate_embedding)."""
    def __init__(self, num_classes=10):
        super().__init__()
        # Load pretrained DenseNet121 and modify for CIFAR-10
        self.densenet = torchvision.models.densenet121(pretrained=False)
        
        # Modify first conv layer for CIFAR-10 (32x32 images)
        # Original DenseNet uses 7x7 conv with stride 2, we use 3x3 with stride 1
        self.densenet.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, 
                                                  padding=1, bias=False)
        # Remove the first pooling layer (not needed for 32x32 images)
        self.densenet.features.pool0 = nn.Identity()
        
        # Get the number of features before the classifier
        self.embDim = self.densenet.classifier.in_features
        
        # Replace classifier with our own
        self.densenet.classifier = nn.Identity()
        self.fc = nn.Linear(self.embDim, num_classes)
    
    def forward(self, x):
        # Get features from DenseNet backbone
        features = self.densenet.features(x)
        # Global average pooling
        out = F.adaptive_avg_pool2d(features, (1, 1))
        emb = torch.flatten(out, 1)
        # Classification
        logits = self.fc(emb)
        return logits, emb
    
    def get_embedding_dim(self):
        return self.embDim

# ──────────────────────────────────────────────────────────────
# CIFAR-10 DataLoader
# ──────────────────────────────────────────────────────────────
def load_cifar10_data(datadir):
    """
    Load CIFAR-10 dataset and return train/val/test sets.
    CIFAR-10 has 50,000 training images and 10,000 test images.
    We'll use 45,000 for training and 5,000 for validation.
    """
    print('[CIFAR-10] Loading dataset...')
    
    # Define transforms
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
    
    # Download and load training data
    full_trainset = torchvision.datasets.CIFAR10(
        root=datadir, train=True, download=True, transform=transform_train
    )
    
    # Split training data into train and validation (45k/5k split)
    train_size = 45000
    val_size = 5000
    train_dataset, val_dataset = random_split(
        full_trainset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root=datadir, train=False, download=True, transform=transform_test
    )
    
    print(f'[CIFAR-10] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')
    return train_dataset, val_dataset, test_dataset

# ──────────────────────────────────────────────────────────────
# Train/eval helpers
# ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, 100.0 * correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

def train_model(model, train_loader, val_loader, num_epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if epoch % 10 == 0 or epoch == num_epochs:
            print(f'  Epoch [{epoch}/{num_epochs}] Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)')
    
    return best_val_acc

# ──────────────────────────────────────────────────────────────
# GLISTER-specific
# ──────────────────────────────────────────────────────────────
def get_grad_embedding(model, loader, device):
    """
    Compute per-sample gradient embeddings for subset selection.
    Returns: [N x embedding_dim] numpy array
    """
    model.eval()
    embDim = model.get_embedding_dim()
    embeddings = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            _, emb = model(inputs)
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def select_batch_glister(model, unlabeled_loader, labeled_loader, budget, device):
    """
    GLISTER selection: gradient matching between labeled and unlabeled.
    Returns: list of selected indices from the unlabeled_loader.
    """
    # Get embeddings
    unlabeled_embs = get_grad_embedding(model, unlabeled_loader, device)
    labeled_embs = get_grad_embedding(model, labeled_loader, device)
    
    # Compute similarity matrix
    # We want unlabeled samples whose embeddings are most similar to labeled embeddings
    # Use cosine similarity
    unlabeled_norm = unlabeled_embs / (np.linalg.norm(unlabeled_embs, axis=1, keepdims=True) + 1e-8)
    labeled_norm = labeled_embs / (np.linalg.norm(labeled_embs, axis=1, keepdims=True) + 1e-8)
    
    # Similarity matrix: [unlabeled x labeled]
    sim_matrix = unlabeled_norm @ labeled_norm.T
    
    # GLISTER heuristic: select samples with highest average similarity to labeled set
    scores = sim_matrix.mean(axis=1)
    
    # Select top-k
    selected_indices = np.argsort(-scores)[:budget]
    return selected_indices.tolist()

# ──────────────────────────────────────────────────────────────
# Active Learning round
# ──────────────────────────────────────────────────────────────
def active_learning_round(
    model, labeled_indices, unlabeled_indices, train_dataset, val_dataset, test_dataset,
    budget, max_pool_size, num_epochs, lr, device, method='glister'
):
    """
    One active learning round:
    1) Train on current labeled set
    2) Select new samples using method (glister or random)
    3) Add them to labeled set
    4) Return updated labeled/unlabeled indices and performance
    """
    # Create labeled loader
    labeled_subset = Subset(train_dataset, labeled_indices)
    labeled_loader = DataLoader(labeled_subset, batch_size=128, shuffle=True, num_workers=2)
    
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Train model on labeled data
    print(f'  Training on {len(labeled_indices)} labeled samples...')
    best_val_acc = train_model(model, labeled_loader, val_loader, num_epochs, lr, device)
    test_acc = evaluate(model, test_loader, device)
    print(f'  After training: Val Acc = {best_val_acc:.2f}%, Test Acc = {test_acc:.2f}%')
    
    # If no unlabeled data left, we're done
    if len(unlabeled_indices) == 0:
        return labeled_indices, unlabeled_indices, best_val_acc, test_acc
    
    # Sample from unlabeled pool if it's too large
    if len(unlabeled_indices) > max_pool_size:
        pool_indices = random.sample(unlabeled_indices, max_pool_size)
    else:
        pool_indices = unlabeled_indices
    
    # Create unlabeled loader
    unlabeled_subset = Subset(train_dataset, pool_indices)
    unlabeled_loader = DataLoader(unlabeled_subset, batch_size=128, shuffle=False, num_workers=2)
    
    # Select samples
    actual_budget = min(budget, len(pool_indices))
    print(f'  Selecting {actual_budget} samples using {method.upper()}...')
    
    if method == 'glister':
        selected_local_indices = select_batch_glister(
            model, unlabeled_loader, labeled_loader, actual_budget, device
        )
    else:  # random
        selected_local_indices = random.sample(range(len(pool_indices)), actual_budget)
    
    # Map back to original dataset indices
    selected_indices = [pool_indices[i] for i in selected_local_indices]
    
    # Update labeled/unlabeled sets
    new_labeled = labeled_indices + selected_indices
    new_unlabeled = [idx for idx in unlabeled_indices if idx not in selected_indices]
    
    return new_labeled, new_unlabeled, best_val_acc, test_acc

# ──────────────────────────────────────────────────────────────
# Single experiment run
# ──────────────────────────────────────────────────────────────
def run_single_experiment(args, run_idx, method='glister'):
    """
    Run a single active learning experiment (either GLISTER or Random).
    Returns: list of dicts with per-round results
    """
    print(f'\n{"="*80}')
    print(f'Run {run_idx + 1}/{args.num_runs} - Method: {method.upper()}')
    print(f'{"="*80}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set seed for reproducibility of this run
    set_seed(args.seed + run_idx)
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_cifar10_data(args.datadir)
    
    # Initialize labeled/unlabeled pools
    all_train_indices = list(range(len(train_dataset)))
    random.shuffle(all_train_indices)
    
    labeled_indices = all_train_indices[:args.initial_budget]
    unlabeled_indices = all_train_indices[args.initial_budget:]
    
    results = []
    round_num = 0
    
    while len(unlabeled_indices) > 0:
        round_num += 1
        print(f'\n--- Round {round_num} ({method.upper()}) ---')
        print(f'Labeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}')
        
        # Create new model for this round
        model = DenseNet121Emb(num_classes=10).to(device)
        
        # Run active learning round
        labeled_indices, unlabeled_indices, val_acc, test_acc = active_learning_round(
            model, labeled_indices, unlabeled_indices, train_dataset, val_dataset, test_dataset,
            args.budget_per_round, args.max_pool_size, args.epochs_per_round, args.lr, device, method
        )
        
        results.append({
            'round': round_num,
            'labeled_size': len(labeled_indices),
            'val_acc': val_acc,
            'test_acc': test_acc
        })
        
        # Stop if we've labeled everything
        if len(unlabeled_indices) == 0:
            print(f'\nAll training data labeled! Stopping.')
            break
    
    return results

# ──────────────────────────────────────────────────────────────
# Multiple experiment runs with statistics
# ──────────────────────────────────────────────────────────────
def run_multiple_experiments(args):
    """
    Run the active learning experiment multiple times and compute statistics.
    """
    os.makedirs(args.results_dir, exist_ok=True)
    
    all_glister_results = []
    all_random_results = []
    glister_times = []
    random_times = []
    
    # Run GLISTER experiments
    print(f'\n{"#"*80}')
    print(f'RUNNING GLISTER EXPERIMENTS ({args.num_runs} runs)')
    print(f'{"#"*80}')
    
    for run_idx in range(args.num_runs):
        start_time = time.time()
        results = run_single_experiment(args, run_idx, method='glister')
        elapsed = time.time() - start_time
        glister_times.append(elapsed)
        all_glister_results.append(results)
        print(f'\nGLISTER Run {run_idx + 1} completed in {elapsed/3600:.2f} hours')
    
    # Run Random baseline experiments
    print(f'\n{"#"*80}')
    print(f'RUNNING RANDOM BASELINE EXPERIMENTS ({args.num_runs} runs)')
    print(f'{"#"*80}')
    
    for run_idx in range(args.num_runs):
        start_time = time.time()
        results = run_single_experiment(args, run_idx, method='random')
        elapsed = time.time() - start_time
        random_times.append(elapsed)
        all_random_results.append(results)
        print(f'\nRANDOM Run {run_idx + 1} completed in {elapsed/3600:.2f} hours')
    
    # Compute statistics across runs
    num_rounds = len(all_glister_results[0])
    
    stats = {
        'rounds': [],
        'labeled_sizes': [],
        'glister_val_mean': [],
        'glister_val_std': [],
        'glister_test_mean': [],
        'glister_test_std': [],
        'random_val_mean': [],
        'random_val_std': [],
        'random_test_mean': [],
        'random_test_std': []
    }
    
    for round_idx in range(num_rounds):
        glister_vals = [run[round_idx]['val_acc'] for run in all_glister_results]
        glister_tests = [run[round_idx]['test_acc'] for run in all_glister_results]
        random_vals = [run[round_idx]['val_acc'] for run in all_random_results]
        random_tests = [run[round_idx]['test_acc'] for run in all_random_results]
        
        stats['rounds'].append(round_idx + 1)
        stats['labeled_sizes'].append(all_glister_results[0][round_idx]['labeled_size'])
        
        stats['glister_val_mean'].append(np.mean(glister_vals))
        stats['glister_val_std'].append(np.std(glister_vals))
        stats['glister_test_mean'].append(np.mean(glister_tests))
        stats['glister_test_std'].append(np.std(glister_tests))
        
        stats['random_val_mean'].append(np.mean(random_vals))
        stats['random_val_std'].append(np.std(random_vals))
        stats['random_test_mean'].append(np.mean(random_tests))
        stats['random_test_std'].append(np.std(random_tests))
    
    # Time statistics
    glister_time_mean = np.mean(glister_times)
    glister_time_std = np.std(glister_times)
    random_time_mean = np.mean(random_times)
    random_time_std = np.std(random_times)
    
    # Print summary table
    print(f"\n{'='*100}")
    print(f"FINAL RESULTS (Mean ± Stddev over {args.num_runs} runs, {num_rounds} rounds)")
    print(f"{'='*100}")
    print(f"{'Round':<8} {'Labeled':<10} {'GLISTER Val':<20} {'GLISTER Test':<20} "
          f"{'Random Val':<20} {'Random Test':<20}")
    print(f"{'-'*100}")
    
    for i in range(num_rounds):
        print(f"{stats['rounds'][i]:<8} {stats['labeled_sizes'][i]:<10} "
              f"{stats['glister_val_mean'][i]:6.2f}±{stats['glister_val_std'][i]:4.2f}        "
              f"{stats['glister_test_mean'][i]:6.2f}±{stats['glister_test_std'][i]:4.2f}        "
              f"{stats['random_val_mean'][i]:6.2f}±{stats['random_val_std'][i]:4.2f}        "
              f"{stats['random_test_mean'][i]:6.2f}±{stats['random_test_std'][i]:4.2f}")
    
    print(f"{'='*100}")
    print(f"\nTime Statistics:")
    print(f"  GLISTER: {glister_time_mean/3600:.2f}±{glister_time_std/3600:.2f} hours")
    print(f"  RANDOM:  {random_time_mean/3600:.2f}±{random_time_std/3600:.2f} hours")
    print(f"{'='*100}")
    
    # Plot results with error bars
    plt.figure(figsize=(16, 6))
    
    # Plot 1: Performance by Round
    plt.subplot(1, 2, 1)
    rounds = stats['rounds']
    
    plt.errorbar(rounds, stats['glister_val_mean'], yerr=stats['glister_val_std'],
                 fmt='b-o', capsize=3, label='GLISTER Val', alpha=0.7)
    plt.errorbar(rounds, stats['glister_test_mean'], yerr=stats['glister_test_std'],
                 fmt='b--s', capsize=3, label='GLISTER Test', alpha=0.7)
    plt.errorbar(rounds, stats['random_val_mean'], yerr=stats['random_val_std'],
                 fmt='g-o', capsize=3, label='Random Val', alpha=0.7)
    plt.errorbar(rounds, stats['random_test_mean'], yerr=stats['random_test_std'],
                 fmt='g--s', capsize=3, label='Random Test', alpha=0.7)
    
    plt.xlabel('Active Learning Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Active Learning Performance by Round\n(Mean ± Std over {args.num_runs} runs)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Performance by Labeled Pool Size
    plt.subplot(1, 2, 2)
    labeled_sizes = stats['labeled_sizes']
    
    plt.errorbar(labeled_sizes, stats['glister_val_mean'], yerr=stats['glister_val_std'],
                 fmt='b-o', capsize=3, label='GLISTER Val', alpha=0.7)
    plt.errorbar(labeled_sizes, stats['glister_test_mean'], yerr=stats['glister_test_std'],
                 fmt='b--s', capsize=3, label='GLISTER Test', alpha=0.7)
    plt.errorbar(labeled_sizes, stats['random_val_mean'], yerr=stats['random_val_std'],
                 fmt='g-o', capsize=3, label='Random Val', alpha=0.7)
    plt.errorbar(labeled_sizes, stats['random_test_mean'], yerr=stats['random_test_std'],
                 fmt='g--s', capsize=3, label='Random Test', alpha=0.7)
    
    plt.xlabel('Labeled Pool Size', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Active Learning Performance by Pool Size\n(Mean ± Std over {args.num_runs} runs)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(args.results_dir, 
                           f'active_learning_{args.feature}_init{args.initial_budget}_'
                           f'budget{args.budget_per_round}_runs{args.num_runs}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved: {fig_path}')
    
    # Save detailed results to file
    log_path = os.path.join(args.results_dir,
                           f'active_learning_{args.feature}_init{args.initial_budget}_'
                           f'budget{args.budget_per_round}_runs{args.num_runs}.txt')
    with open(log_path, 'w') as f:
        f.write(f"Active Learning Results - Multiple Runs\n")
        f.write(f"{'='*70}\n")
        f.write(f"Number of runs: {args.num_runs}\n")
        f.write(f"Number of rounds: {num_rounds}\n")
        f.write(f"Initial budget: {args.initial_budget}\n")
        f.write(f"Budget per round: {args.budget_per_round}\n")
        f.write(f"Epochs per round: {args.epochs_per_round}\n")
        f.write(f"Max pool size: {args.max_pool_size}\n")
        f.write(f"Feature: {args.feature}\n")
        f.write(f"{'='*70}\n\n")
        
        f.write(f"Time Statistics:\n")
        f.write(f"  GLISTER: {glister_time_mean/3600:.2f}±{glister_time_std/3600:.2f} hours\n")
        f.write(f"  RANDOM:  {random_time_mean/3600:.2f}±{random_time_std/3600:.2f} hours\n\n")
        
        f.write(f"{'='*100}\n")
        f.write(f"{'Round':<8} {'Labeled':<10} {'GLISTER Val':<20} {'GLISTER Test':<20} "
                f"{'Random Val':<20} {'Random Test':<20}\n")
        f.write(f"{'-'*100}\n")
        
        for i in range(num_rounds):
            f.write(f"{stats['rounds'][i]:<8} {stats['labeled_sizes'][i]:<10} "
                   f"{stats['glister_val_mean'][i]:6.2f}±{stats['glister_val_std'][i]:4.2f}        "
                   f"{stats['glister_test_mean'][i]:6.2f}±{stats['glister_test_std'][i]:4.2f}        "
                   f"{stats['random_val_mean'][i]:6.2f}±{stats['random_val_std'][i]:4.2f}        "
                   f"{stats['random_test_mean'][i]:6.2f}±{stats['random_test_std'][i]:4.2f}\n")
        
        f.write(f"{'='*100}\n\n")
        
        # Write individual run details
        f.write(f"\n{'='*70}\n")
        f.write(f"INDIVIDUAL RUN DETAILS\n")
        f.write(f"{'='*70}\n\n")
        
        for run_idx in range(args.num_runs):
            f.write(f"\n--- Run {run_idx + 1} ---\n")
            f.write(f"GLISTER time: {glister_times[run_idx]/3600:.2f} hours\n")
            f.write(f"RANDOM time: {random_times[run_idx]/3600:.2f} hours\n\n")
            
            f.write(f"{'Round':<8} {'Labeled':<10} {'GLISTER Val':<15} {'GLISTER Test':<15} "
                   f"{'Random Val':<15} {'Random Test':<15}\n")
            f.write(f"{'-'*70}\n")
            
            for round_idx in range(num_rounds):
                g = all_glister_results[run_idx][round_idx]
                r = all_random_results[run_idx][round_idx]
                f.write(f"{g['round']:<8} {g['labeled_size']:<10} {g['val_acc']:<15.2f} "
                       f"{g['test_acc']:<15.2f} {r['val_acc']:<15.2f} {r['test_acc']:<15.2f}\n")
    
    print(f'Detailed results saved: {log_path}')
    
    return stats, glister_times, random_times


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = get_args()
    print(f'Active Learning Experiment: CIFAR-10 + DenseNet121 - Multiple Runs')
    print(f'Number of runs: {args.num_runs}')
    print(f'Initial budget: {args.initial_budget}')
    print(f'Budget per round: {args.budget_per_round}')
    print(f'Epochs per round: {args.epochs_per_round}')
    print(f'Max pool size: {args.max_pool_size}')
    print(f'Feature: {args.feature}')
    print(f'Will run until entire dataset is labeled')
    
    stats, glister_times, random_times = run_multiple_experiments(args)
    
    print('\n=== EXPERIMENT COMPLETED ===')
    print(f"Total runs: {args.num_runs}")
    print(f"Rounds per run: {len(stats['rounds'])}")
    print(f"Final GLISTER Test Accuracy: {stats['glister_test_mean'][-1]:.2f}±{stats['glister_test_std'][-1]:.2f}%")
    print(f"Final Random Test Accuracy: {stats['random_test_mean'][-1]:.2f}±{stats['random_test_std'][-1]:.2f}%")
