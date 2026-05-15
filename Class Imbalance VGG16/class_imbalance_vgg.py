
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
import scipy.io
import numpy as np
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Define VGG model
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

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


def create_imbalanced_dataset(dataset, target_counts):
    """
    Create an imbalanced dataset by randomly removing samples from each class.
    
    Args:
        dataset: Original dataset
        target_counts: Dict with class indices as keys and target counts as values
                      {0: 5000, 1: 4500, ...}
    """
    # Group indices by class
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)
    
    # Randomly sample from each class
    selected_indices = []
    for class_id, count in target_counts.items():
        indices = class_indices[class_id]
        if len(indices) < count:
            print(f"Warning: Class {class_id} has only {len(indices)} samples, but {count} requested.")
            selected_indices.extend(indices)
        else:
            selected = np.random.choice(indices, count, replace=False)
            selected_indices.extend(selected.tolist())
    
    return Subset(dataset, selected_indices)


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


# Function to train the model
def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    learning_rate,
    path,
    accumulate_steps=1
):
    model.to(device)
    model.train()

    best_accuracy = test_model(model, test_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # IMPORTANT: reduce CUDA memory fragmentation
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if (i + 1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

        scheduler.step()

        # ---- evaluation (no gradients) ----
        new_accuracy = test_model(model, test_loader)

        print(
            f"epoch: {epoch}, "
            f"accuracy: {new_accuracy:.2f}, "
            f"loss: {running_loss / len(train_loader):.4f}"
        )

        # ---- save only ONE best model (overwrite) ----
        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
            torch.save(
                model.state_dict(),
                os.path.join(path, "best_model.pt")
            )

    return best_accuracy



def train_test_save(train_dataset, test_loader, n, epochs, path, lr=0.01):
    best_accuracy = 0.0
    for _ in range(n):
        model = VGG('VGG16').to(device)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        new_accuracy = train_model(model, train_loader, test_loader, epochs, lr, path)

        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
    print(f"best accuracy: {best_accuracy:.2f}")


def get_highconf_and_remainder_datasets(model, dataset, k):
    model.eval()
    model = model.to(device)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    confidence_scores = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            confidences = F.softmax(outputs, dim=1).max(dim=1)[0]
            confidence_scores.extend(confidences.tolist())

    model.train()

    top_indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i], reverse=True)[:k]
    
    # Get actual dataset indices
    if isinstance(dataset, Subset):
        actual_top_indices = [dataset.indices[i] for i in top_indices]
        all_indices = set(dataset.indices)
        remaining_indices = [idx for idx in dataset.indices if idx not in actual_top_indices]
        highconf_samples = Subset(dataset.dataset, actual_top_indices)
        remainder_dataset = Subset(dataset.dataset, remaining_indices)
    else:
        highconf_samples = Subset(dataset, top_indices)
        remaining_indices = [i for i in range(len(dataset)) if i not in top_indices]
        remainder_dataset = Subset(dataset, remaining_indices)
    
    return highconf_samples, remainder_dataset


def get_lowconf_and_remainder_datasets(model, dataset, k_low):
    model.eval()
    model = model.to(device)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    confidence_scores = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            confidences = F.softmax(outputs, dim=1).max(dim=1)[0]
            confidence_scores.extend(confidences.tolist())

    top_low_indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i], reverse=False)[:k_low]
    
    # Get actual dataset indices
    if isinstance(dataset, Subset):
        actual_low_indices = [dataset.indices[i] for i in top_low_indices]
        remaining_indices = [idx for idx in dataset.indices if idx not in actual_low_indices]
        lowconf_samples = Subset(dataset.dataset, actual_low_indices)
        remainder_dataset = Subset(dataset.dataset, remaining_indices)
    else:
        lowconf_samples = Subset(dataset, top_low_indices)
        remaining_indices = [i for i in range(len(dataset)) if i not in top_low_indices]
        remainder_dataset = Subset(dataset, remaining_indices)
    
    return lowconf_samples, remainder_dataset


def train_until_degradation(model, train_dataset, remaining_set, test_loader, path, use_high_conf=False):
    best_accuracy = test_model(model, test_loader)
    current_accuracy = 0.0
    accuracies = []
    iteration = 0

    while len(remaining_set) != 0:
        iteration += 1
        
        if use_high_conf:
            new_samples, remaining_set = get_highconf_and_remainder_datasets(model, remaining_set, k=2750)
            print(f"Iteration {iteration}: Adding HIGH confidence samples")
        else:
            new_samples, remaining_set = get_lowconf_and_remainder_datasets(model, remaining_set, k_low=2750)
            print(f"Iteration {iteration}: Adding LOW confidence samples")
        
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, new_samples])   
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)      
        current_accuracy = train_model(model, train_loader, test_loader, 100, 0.01, path)
        accuracies.append(current_accuracy)
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
        print(f"Current accuracy: {current_accuracy:.2f}, Best accuracy: {best_accuracy:.2f}")
        import gc
        del train_loader
        torch.cuda.empty_cache()
        gc.collect()
        
    print("Training dataset exhausted. Stopping training.")
    print("Final accuracy on the test set:", best_accuracy)

    with open(f"{path}/accuracies.txt", "w") as file:
        for i, acc in enumerate(accuracies, start=1):
            file.write(f"Iteration {i}: Accuracy = {acc:.2f}\n")

    torch.save(model.state_dict(), f"{path}/finalmodel_{best_accuracy:.2f}.pt")
    return model, best_accuracy


def run_experiment(method_name, train_set, test_loader, base_path, 
                   use_high_conf_initial, use_high_conf_training):
    """
    Run a single experiment with specified method.
    
    Args:
        method_name: Name of the method for logging
        train_set: Imbalanced training dataset
        test_loader: Test data loader
        base_path: Base path for saving models
        use_high_conf_initial: If True, use high confidence for initial selection
        use_high_conf_training: If True, use high confidence in train_until_degradation
    """
    print(f"\n{'='*80}")
    print(f"Starting Experiment: {method_name}")
    print(f"{'='*80}\n")
    
    # Create directory for this method
    method_path = os.path.join(base_path, method_name.replace(" ", "_"))
    os.makedirs(method_path, exist_ok=True)
    
    # Initialize model
    initial_model = VGG('VGG16').to(device)
    
    # Select initial training set
    if use_high_conf_initial:
        print("Selecting HIGH confidence samples for initial training set...")
        initial_trainset, remainder = get_highconf_and_remainder_datasets(
            initial_model, train_set, k=5500)
    else:
        print("Selecting LOW confidence samples for initial training set...")
        initial_trainset, remainder = get_lowconf_and_remainder_datasets(
            initial_model, train_set, k_low=5500)
    
    print(f"Initial training set size: {len(initial_trainset)}")
    print(f"Remaining set size: {len(remainder)}")
    
    # Train on initial set
    print("\nTraining on initial set...")
    train_test_save(initial_trainset, test_loader, 1, 100, method_path, lr=0.01)
    # Re-create model and load best weights from initial training
    model = VGG('VGG16').to(device)
    model.load_state_dict(torch.load(os.path.join(method_path, "best_model.pt"),map_location=device))

    
    # Continue training until degradation
    print(f"\nContinuing training with {'HIGH' if use_high_conf_training else 'LOW'} confidence selection...")
    final_model, final_accuracy = train_until_degradation(
        model, initial_trainset, remainder, test_loader, method_path, 
        use_high_conf=use_high_conf_training)
    
    print(f"\n{method_name} - Final Accuracy: {final_accuracy:.2f}%")
    
    return final_accuracy


def run_all_experiments(train_set, test_loader, base_path):
    """
    Wrapper function to run all 4 method combinations.
    """
    results = {}
    
    methods = [
        ("Method1_LowConf_Initial_LowConf_Training", False, False),
        ("Method2_LowConf_Initial_HighConf_Training", False, True),
        ("Method3_HighConf_Initial_LowConf_Training", True, False),
        ("Method4_HighConf_Initial_HighConf_Training", True, True),
    ]
    
    for method_name, use_high_initial, use_high_training in methods:
        accuracy = run_experiment(
            method_name, train_set, test_loader, base_path,
            use_high_initial, use_high_training
        )
        results[method_name] = accuracy
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL EXPERIMENTS")
    print(f"{'='*80}")
    for method_name, accuracy in results.items():
        print(f"{method_name}: {accuracy:.2f}%")
    
    # Save summary
    with open(os.path.join(base_path, "experiment_summary.txt"), "w") as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n")
        for method_name, accuracy in results.items():
            f.write(f"{method_name}: {accuracy:.2f}%\n")
    
    return results


# Main execution
if __name__ == "__main__":
    # Kaggle CIFAR10 setup
    KAGGLE_CIFAR_PATH = "/kaggle/input/cifar10/CIFAR10"
    BATCH_SIZE = 64
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    
    train_set_full = datasets.ImageFolder(
        root=os.path.join(KAGGLE_CIFAR_PATH, "train"),
        transform=transform_train
    )
    
    test_set = datasets.ImageFolder(
        root=os.path.join(KAGGLE_CIFAR_PATH, "test"),
        transform=transform_test
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    print("Full train size:", len(train_set_full))
    print("Test size:", len(test_set))
    
    # Create imbalanced dataset
    target_counts = {
        0: 5000,  # class 1
        1: 4500,  # class 2
        2: 4000,  # class 3
        3: 3500,  # class 4
        4: 3000,  # class 5
        5: 2500,  # class 6
        6: 2000,  # class 7
        7: 1500,  # class 8
        8: 1000,  # class 9
        9: 500    # class 10
    }
    
    print("\nCreating imbalanced dataset...")
    train_set_imbalanced = create_imbalanced_dataset(train_set_full, target_counts)
    print(f"Imbalanced train size: {len(train_set_imbalanced)}")
    
    # Base path for saving results
    base_path = "/kaggle/working/VGG_CLC_Experiments"
    os.makedirs(base_path, exist_ok=True)
    
    # Run all experiments
    results = run_all_experiments(train_set_imbalanced, test_loader, base_path)
