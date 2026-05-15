"""
GLISTER Active Learning for TinyImageNet + ResNet18 - Multiple Runs
Runs experiments 3 times and reports mean ± stddev for time and accuracy.

Dataset: Downloads automatically from http://cs231n.stanford.edu/tiny-imagenet-200.zip

Active Learning Setup:
- Start with small labeled pool (initial_budget)
- Each round: select budget_per_round new samples using GLISTER, add to labeled pool, retrain
- Continue until all training data is labeled
- Compare against random selection baseline
- Run 3 times and report statistics

Usage:
    python glister_active_learning_multiple_runs.py \
        --datadir ./data/tiny-imagenet-200 \
        --initial_budget 5000 \
        --budget_per_round 5000 \
        --epochs_per_round 20 \
        --max_pool_size 15000 \
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
import urllib.request
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import (
    DataLoader, Dataset, Subset, random_split
)

# ──────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir',           default='./data/tiny-imagenet-200',
                        help='Directory to download/store TinyImageNet dataset')
    parser.add_argument('--initial_budget',    type=int,   default=1000,
                        help='Initial labeled pool size')
    parser.add_argument('--budget_per_round',  type=int,   default=1000,
                        help='Number of samples to select per round')
    parser.add_argument('--epochs_per_round',  type=int,   default=100,
                        help='Training epochs per round')
    parser.add_argument('--max_pool_size',     type=int,   default=15000,
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
# ResNet18
# ──────────────────────────────────────────────────────────────
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18Emb(nn.Module):
    """ResNet-18 that returns (logits, penultimate_embedding)."""
    def __init__(self, num_classes=200):
        super().__init__()
        self.in_planes = 64
        self.embDim = 512
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        emb = out.view(out.size(0), -1)
        return self.linear(emb), emb

    def get_embedding_dim(self):
        return self.embDim

# ──────────────────────────────────────────────────────────────
# TinyImageNet dataset downloader
# ──────────────────────────────────────────────────────────────
def download_tinyimagenet(data_root):
    """
    Download and extract TinyImageNet dataset if not already present.
    Downloads from Stanford's CS231n course server.
    """
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    
    # Check if dataset already exists
    if os.path.isdir(os.path.join(data_root, 'train')) and \
       os.path.isdir(os.path.join(data_root, 'val')):
        print(f"[TinyImageNet] Dataset already exists at {data_root}")
        return data_root
    
    # Create parent directory
    parent_dir = os.path.dirname(data_root) if os.path.dirname(data_root) else '.'
    os.makedirs(parent_dir, exist_ok=True)
    
    zip_path = os.path.join(parent_dir, 'tiny-imagenet-200.zip')
    
    # Download if zip doesn't exist
    if not os.path.exists(zip_path):
        print(f"[TinyImageNet] Downloading dataset from {url}")
        print(f"[TinyImageNet] This may take several minutes (~237 MB)...")
        
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rDownloading: {percent}%", end='', flush=True)
        
        try:
            urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
            print("\n[TinyImageNet] Download complete!")
        except Exception as e:
            print(f"\n[TinyImageNet] Download failed: {e}")
            raise
    else:
        print(f"[TinyImageNet] Found existing zip file at {zip_path}")
    
    # Extract the zip file
    if not os.path.isdir(data_root):
        print(f"[TinyImageNet] Extracting dataset to {parent_dir}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(parent_dir)
            print(f"[TinyImageNet] Extraction complete!")
        except Exception as e:
            print(f"[TinyImageNet] Extraction failed: {e}")
            raise
    
    # Verify extraction
    if not os.path.isdir(os.path.join(data_root, 'train')) or \
       not os.path.isdir(os.path.join(data_root, 'val')):
        raise RuntimeError(f"Dataset extraction failed. Expected train/ and val/ dirs in {data_root}")
    
    print(f"[TinyImageNet] Dataset ready at {data_root}")
    return data_root


# ──────────────────────────────────────────────────────────────
# TinyImageNet dataset loader
# ──────────────────────────────────────────────────────────────
_IMG_EXTS = {'.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG'}

def _is_image(fname):
    return os.path.splitext(fname)[1] in _IMG_EXTS

def _find_datadir(root):
    if os.path.isdir(os.path.join(root, 'train')) and os.path.isdir(os.path.join(root, 'val')):
        return root
    for name in os.listdir(root):
        sub = os.path.join(root, name)
        if os.path.isdir(sub):
            if os.path.isdir(os.path.join(sub, 'train')) and os.path.isdir(os.path.join(sub, 'val')):
                print(f"[TinyImageNet] Auto-detected dataset root: {sub}")
                return sub
    raise RuntimeError(f"Cannot find 'train/' and 'val/' under '{root}'.")

class TinyImageNetDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.transform = transform
        self.samples   = []
        self.targets   = []

        # Download dataset if not present
        root = download_tinyimagenet(root)
        root = _find_datadir(root)
        train_dir = os.path.join(root, 'train')

        wnids = sorted(
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        )
        self.class_to_idx = {w: i for i, w in enumerate(wnids)}

        if split == 'train':
            for wnid in wnids:
                cls_dir = os.path.join(train_dir, wnid)
                img_dir = os.path.join(cls_dir, 'images')
                if not os.path.isdir(img_dir):
                    img_dir = cls_dir
                idx = self.class_to_idx[wnid]
                for fname in os.listdir(img_dir):
                    if _is_image(fname):
                        self.samples.append(os.path.join(img_dir, fname))
                        self.targets.append(idx)

        elif split == 'val':
            val_dir  = os.path.join(root, 'val')
            ann_file = os.path.join(val_dir, 'val_annotations.txt')
            img_dir  = os.path.join(val_dir, 'images')

            if os.path.exists(ann_file) and os.path.isdir(img_dir):
                fname_to_wnid = {}
                with open(ann_file) as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            fname_to_wnid[parts[0]] = parts[1]
                for fname in sorted(os.listdir(img_dir)):
                    if not _is_image(fname):
                        continue
                    wnid = fname_to_wnid.get(fname)
                    if wnid and wnid in self.class_to_idx:
                        self.samples.append(os.path.join(img_dir, fname))
                        self.targets.append(self.class_to_idx[wnid])
            else:
                for wnid in wnids:
                    cls_dir = os.path.join(val_dir, wnid)
                    if not os.path.isdir(cls_dir):
                        continue
                    img_subdir = os.path.join(cls_dir, 'images')
                    search = img_subdir if os.path.isdir(img_subdir) else cls_dir
                    idx = self.class_to_idx[wnid]
                    for fname in os.listdir(search):
                        if _is_image(fname):
                            self.samples.append(os.path.join(search, fname))
                            self.targets.append(idx)

        if not self.samples:
            raise RuntimeError(f"No images loaded for split='{split}' from '{root}'.")

        self.targets = np.array(self.targets, dtype=np.int64)
        print(f"[TinyImageNet] split={split}  images={len(self.samples)}  "
              f"classes={len(np.unique(self.targets))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])


def get_tinyimagenet_loaders(datadir, feature='dss', val_fraction=0.1,
                              batch_size=64, num_workers=2):
    mean = (0.480, 0.448, 0.398)
    std  = (0.277, 0.269, 0.282)

    train_tf = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_dataset = TinyImageNetDataset(datadir, split='train', transform=train_tf)
    test_dataset = TinyImageNetDataset(datadir, split='val',   transform=eval_tf)
    num_cls = 200

    if feature == 'classimb':
        full_dataset = _apply_classimb(full_dataset, num_cls)
    elif feature == 'noise':
        full_dataset = _apply_noise(full_dataset, num_cls, noise_rate=0.2)

    n_full = len(full_dataset)
    n_val  = int(n_full * val_fraction)
    n_trn  = n_full - n_val
    trainset, valset = random_split(
        full_dataset, [n_trn, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    class _EvalSubset(Dataset):
        def __init__(self, subset, root, tf):
            self.indices = subset.indices
            self.base = TinyImageNetDataset(root, split='train', transform=tf)
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.base[self.indices[i]]

    valset_eval = _EvalSubset(valset, datadir, eval_tf)

    valloader   = DataLoader(valset_eval,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)
    testloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)

    print(f"[Loaders] train={len(trainset)}  val={len(valset_eval)}  "
          f"test={len(test_dataset)}  classes={num_cls}")

    return trainset, valset_eval, test_dataset, valloader, testloader, num_cls


def _apply_classimb(dataset, num_cls):
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    counts  = np.bincount(targets, minlength=num_cls)
    min_cnt = int(counts.min() * 0.1)
    sel_cls = np.random.choice(num_cls, size=int(0.3 * num_cls), replace=False)
    keep    = []
    for c in range(num_cls):
        idxs = np.where(targets == c)[0]
        if c in sel_cls:
            idxs = np.random.choice(idxs, size=min(min_cnt, len(idxs)), replace=False)
        keep.extend(idxs.tolist())
    return Subset(dataset, keep)


def _apply_noise(dataset, num_cls, noise_rate=0.2):
    class NoisyDataset(Dataset):
        def __init__(self, ds, noisy_targets):
            self.ds = ds
            self.noisy_targets = noisy_targets
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, _ = self.ds[idx]
            return img, int(self.noisy_targets[idx])

    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    noisy   = targets.copy()
    n_noisy = int(len(targets) * noise_rate)
    idx_noisy = np.random.choice(len(targets), size=n_noisy, replace=False)
    noisy[idx_noisy] = np.random.randint(0, num_cls, size=n_noisy)
    return NoisyDataset(dataset, noisy)


# ──────────────────────────────────────────────────────────────
# Dataset with id
# ──────────────────────────────────────────────────────────────
class DatasetWithId(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx


# ──────────────────────────────────────────────────────────────
# GLISTER (Taylor one-step) set function
# ──────────────────────────────────────────────────────────────
class SetFunctionTaylor:
    """
    Greedy one-step Taylor approximation subset selection (GLISTER).
    """
    def __init__(self, trainset, valloader, model, criterion, criterion_nored,
                 lr, device, num_cls, max_pool_size=15000):
        self.trainset       = trainset
        self.valloader      = valloader
        self.model          = model
        self.criterion      = criterion
        self.criterion_nored = criterion_nored
        self.lr             = lr
        self.device         = device
        self.num_cls        = num_cls
        self.max_pool_size  = max_pool_size

    def _val_gradient(self, clone_dict):
        """Compute ∇L_val w.r.t. model parameters (flat vector)."""
        self.model.load_state_dict(clone_dict)
        self.model.eval()
        
        # Accumulate gradients across validation batches
        self.model.zero_grad()
        total_loss = 0.0
        n_batches = 0
        
        for x, y in self.valloader:
            x, y = x.to(self.device), y.to(self.device)
            out, _ = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            total_loss += loss.item()
            n_batches += 1
        
        # Collect gradients
        grads = [p.grad.detach().clone() for p in self.model.parameters() if p.grad is not None]
        
        # Clear memory
        self.model.zero_grad()
        torch.cuda.empty_cache()
        
        return torch.cat([g.view(-1) for g in grads])

    def select_samples(self, budget, clone_dict, unlabeled_indices):
        """
        Select 'budget' samples from unlabeled_indices using GLISTER.
        Memory-efficient version that processes unlabeled pool in chunks.
        Returns list of selected indices.
        """
        # If unlabeled pool is very large, sample a subset for efficiency
        if len(unlabeled_indices) > self.max_pool_size:
            sampled_pool = np.random.choice(
                unlabeled_indices, size=self.max_pool_size, replace=False
            ).tolist()
            print(f"  Sampled {self.max_pool_size} from {len(unlabeled_indices)} unlabeled samples")
        else:
            sampled_pool = unlabeled_indices
        
        val_grad = self._val_gradient(clone_dict)

        self.model.load_state_dict(clone_dict)
        self.model.train()

        scores = []
        bsz    = 128
        loader = DataLoader(
            DatasetWithId(Subset(self.trainset, sampled_pool)),
            batch_size=bsz, shuffle=False, num_workers=2, pin_memory=False)

        for x, y, idx in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            out, _ = self.model(x)
            loss   = self.criterion(out, y)
            loss.backward()
            trn_grad = torch.cat([
                p.grad.detach().view(-1)
                for p in self.model.parameters() if p.grad is not None
            ])
            score  = float(torch.dot(val_grad, -trn_grad).item())
            scores.extend([score / x.size(0)] * x.size(0))
            self.model.zero_grad()
            
            # Clear cache periodically
            if len(scores) % 1000 == 0:
                torch.cuda.empty_cache()

        scores = np.array(scores)
        top_k  = np.argsort(-scores)[:budget]
        return [sampled_pool[i] for i in top_k]


# ──────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────
def model_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            out, _ = model(x)
            pred   = out.argmax(1)
            correct += pred.eq(y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def weight_reset(m):
    set_seed(42)
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def train_model(model, optimizer, scheduler, criterion, trainset, 
                labeled_indices, num_epochs, valloader, testloader, 
                batch_size, device):
    """Train model for num_epochs on labeled_indices."""
    val_accs = []
    test_accs = []
    
    for ep in range(num_epochs):
        model.train()
        subset = Subset(trainset, labeled_indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=False)
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = model(x)
            loss   = criterion(out, y)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.clamp_(-0.1, 0.1)
            optimizer.step()
        
        scheduler.step()
        
        # Clear cache after each epoch
        torch.cuda.empty_cache()
        
        # Evaluate every 5 epochs
        if (ep + 1) % 5 == 0 or ep == num_epochs - 1:
            va = model_accuracy(model, valloader, device)
            ta = model_accuracy(model, testloader, device)
            val_accs.append(va)
            test_accs.append(ta)
    
    return val_accs[-1], test_accs[-1]


# ──────────────────────────────────────────────────────────────
# Active Learning Loop (single run)
# ──────────────────────────────────────────────────────────────
def active_learning_loop(args, trainset, valloader, testloader, num_cls, 
                        device, selection_method='glister'):
    """
    Run active learning rounds until the dataset is exhausted.
    selection_method: 'glister' or 'random'
    Returns: (round_results, total_time)
    """
    start_time = time.time()
    
    N = len(trainset)
    
    # Initialize labeled and unlabeled pools
    all_indices = list(range(N))
    np.random.shuffle(all_indices)
    
    labeled_indices = all_indices[:args.initial_budget]
    unlabeled_indices = all_indices[args.initial_budget:]
    
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    
    # Track results per round
    round_results = []
    
    print(f"\n{'='*60}")
    print(f"Active Learning with {selection_method.upper()}")
    print(f"Initial labeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}")
    print(f"Budget per round: {args.budget_per_round}")
    print(f"Will run until dataset is exhausted")
    print(f"{'='*60}")
    
    round_num = 0
    
    # Continue until no more unlabeled samples
    while True:
        round_num += 1
        print(f"\n--- Round {round_num} ---")
        print(f"Labeled pool size: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}")
        
        # Create fresh model for this round
        model = ResNet18Emb(num_classes=num_cls).to(device)
        model.apply(weight_reset)
        
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                             momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_per_round)
        
        # Train on current labeled pool
        val_acc, test_acc = train_model(
            model, optimizer, scheduler, criterion, trainset,
            labeled_indices, args.epochs_per_round, valloader, testloader,
            64, device
        )
        
        print(f"After training: Val={val_acc:.2f}%, Test={test_acc:.2f}%")
        
        round_results.append({
            'round': round_num,
            'labeled_size': len(labeled_indices),
            'val_acc': val_acc,
            'test_acc': test_acc
        })
        
        # Check if we have more unlabeled samples to select
        if len(unlabeled_indices) == 0:
            print(f"\nDataset exhausted! All {N} samples have been labeled.")
            break
        
        # Select new samples
        budget = min(args.budget_per_round, len(unlabeled_indices))
        
        if selection_method == 'glister':
            # Use GLISTER to select samples
            clone_dict = copy.deepcopy(model.state_dict())
            setf = SetFunctionTaylor(
                trainset, valloader, model, criterion, criterion_nored,
                args.lr, device, num_cls, max_pool_size=args.max_pool_size
            )
            selected = setf.select_samples(budget, clone_dict, unlabeled_indices)
            
            # Clean up
            del setf, clone_dict
            torch.cuda.empty_cache()
        else:  # random
            selected = np.random.choice(
                unlabeled_indices, size=budget, replace=False
            ).tolist()
        
        # Move selected samples to labeled pool
        labeled_indices.extend(selected)
        unlabeled_indices = [idx for idx in unlabeled_indices if idx not in selected]
        
        print(f"Selected {len(selected)} new samples")
        
        # Clean up model
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    
    return round_results, total_time


# ──────────────────────────────────────────────────────────────
# Main experiment with multiple runs
# ──────────────────────────────────────────────────────────────
def run_multiple_experiments(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    os.makedirs(args.results_dir, exist_ok=True)

    # Load data once
    print('Loading TinyImageNet...')
    (trainset, valset, testset, valloader, testloader, num_cls) = \
        get_tinyimagenet_loaders(args.datadir, feature=args.feature, 
                                val_fraction=0.1, batch_size=64, num_workers=2)

    # Storage for all runs
    all_glister_results = []
    all_random_results = []
    glister_times = []
    random_times = []
    
    # Run experiments multiple times
    for run_idx in range(args.num_runs):
        print(f"\n{'#'*70}")
        print(f"# RUN {run_idx + 1}/{args.num_runs}")
        print(f"{'#'*70}")
        
        # Set different seed for each run
        run_seed = args.seed + run_idx * 100
        set_seed(run_seed)
        
        # Run GLISTER active learning
        print(f"\n>>> Running GLISTER (Run {run_idx + 1})...")
        glister_results, glister_time = active_learning_loop(
            args, trainset, valloader, testloader, num_cls, device, 
            selection_method='glister'
        )
        all_glister_results.append(glister_results)
        glister_times.append(glister_time)
        print(f"GLISTER Run {run_idx + 1} completed in {glister_time/3600:.2f} hours")
        
        # Reset seed for fair comparison
        set_seed(run_seed)
        
        # Run Random active learning
        print(f"\n>>> Running RANDOM (Run {run_idx + 1})...")
        random_results, random_time = active_learning_loop(
            args, trainset, valloader, testloader, num_cls, device,
            selection_method='random'
        )
        all_random_results.append(random_results)
        random_times.append(random_time)
        print(f"RANDOM Run {run_idx + 1} completed in {random_time/3600:.2f} hours")
    
    # Compute statistics
    print(f"\n{'='*70}")
    print("COMPUTING STATISTICS ACROSS ALL RUNS")
    print(f"{'='*70}")
    
    num_rounds = len(all_glister_results[0])
    
    # Initialize storage for statistics
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
    
    # Compute statistics for each round
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
    print(f'Active Learning Experiment: TinyImageNet - Multiple Runs')
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