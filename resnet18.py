# Fixed version of Tiny ImageNet ResNet training code with correct test label handling
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Model Components
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class CustomResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def get_model():
    return CustomResNet(BasicBlock, [2, 2, 2, 2]).to(device)

# Transforms
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_test = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Correct TinyImageNet val loader
class TinyImageNetValDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        with open(annotations_file, 'r') as f:
            lines = f.readlines()
        self.img_labels = [(line.split('\t')[0], line.split('\t')[1]) for line in lines]
        self.transform = transform
        self.img_dir = img_dir
        self.classes = sorted(set(label for _, label in self.img_labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_name, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label_idx = self.class_to_idx[label]
        if self.transform:
            image = self.transform(image)
        return image, label_idx

# Load data
def load_tiny_imagenet(root='./tiny-imagenet-200'):
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')
    val_img_dir = os.path.join(val_dir, 'images')
    val_annotations = os.path.join(val_dir, 'val_annotations.txt')
    train_set = datasets.ImageFolder(train_dir, transform=transform_train)
    val_set = TinyImageNetValDataset(val_img_dir, val_annotations, transform=transform_test)
    return train_set, val_set

train_set, test_set = load_tiny_imagenet()
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)


try:
    train_set, test_set = load_tiny_imagenet('./tiny-imagenet-200')
except RuntimeError as e:
    print(e)
    possible_paths = ['./tiny-imagenet-200', '../tiny-imagenet-200',
                      './data/tiny-imagenet-200', '../data/tiny-imagenet-200']
    for path in possible_paths:
        if os.path.exists(path):
            train_set, test_set = load_tiny_imagenet(path)
            break
    else:
        raise RuntimeError("Could not find Tiny ImageNet dataset.")

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

# Training/Testing Functions
def test_model(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100. * correct / total

def train_model(model, train_loader, epochs, learning_rate, accumulate_steps=1):
    best_accuracy = test_model(model, test_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
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
        scheduler.step()
        new_accuracy = test_model(model, test_loader)
        print(f"Epoch: {epoch}, Accuracy: {new_accuracy:.2f}%, Loss: {loss:.4f}")
        if new_accuracy > best_accuracy:
            os.makedirs("./Results", exist_ok=True)
            if os.path.exists(f"./Results/model_{best_accuracy:.2f}.pt"):
                os.remove(f"./Results/model_{best_accuracy:.2f}.pt")
            best_accuracy = new_accuracy
            torch.save(model.state_dict(), f"./Results/model_{best_accuracy:.2f}.pt")
    return best_accuracy

def train_test_save(train_dataset, n, epochs, lr=0.001):
    best_accuracy = 0.0
    for _ in range(n):
        model = get_model()
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        new_accuracy = train_model(model, train_loader, epochs, lr)
        best_accuracy = max(best_accuracy, new_accuracy)
    print(f"Best accuracy: {best_accuracy:.2f}%")

def get_highconf_and_remainder_datasets(model, dataset, k):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    confidence_scores = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            confidences = F.softmax(outputs, dim=1).max(dim=1)[0]
            confidence_scores.extend(confidences.cpu().tolist())
    top_indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i], reverse=True)[:k]
    highconf_samples = torch.utils.data.Subset(dataset, top_indices)
    remainder_indices = list(set(range(len(dataset))) - set(top_indices))
    remainder_dataset = torch.utils.data.Subset(dataset, remainder_indices)
    return highconf_samples, remainder_dataset

def get_lowconf_and_remainder_datasets(model, dataset, k_low):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    confidence_scores = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            confidences = F.softmax(outputs, dim=1).max(dim=1)[0]
            confidence_scores.extend(confidences.cpu().tolist())
    low_indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i])[:k_low]
    lowconf_samples = torch.utils.data.Subset(dataset, low_indices)
    remainder_indices = list(set(range(len(dataset))) - set(low_indices))
    remainder_dataset = torch.utils.data.Subset(dataset, remainder_indices)
    return lowconf_samples, remainder_dataset

def load_best_model_from_folder(folder_path):
    def extract_acc(name):
        return float(name.split("_")[1][:-3])
    files = [f for f in os.listdir(folder_path) if f.startswith("model_")]
    if not files:
        return None
    best_model_path = os.path.join(folder_path, max(files, key=extract_acc))
    model = get_model()
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model

def train_until_degradation(model, train_dataset, remaining_set, degradation_thres=0.03):
    best_accuracy = test_model(model, test_loader)
    accuracies = []
    iteration = 1
    while len(remaining_set) > 0:
        print(f"\nIteration {iteration}")
        lowconf_samples, remaining_set = get_lowconf_and_remainder_datasets(model, remaining_set, k_low=5000)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, lowconf_samples])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        acc = train_model(model, train_loader, epochs=100, learning_rate=0.01)
        accuracies.append(acc)
        if acc - best_accuracy < -degradation_thres * best_accuracy:
            print("Performance degraded. Stopping training.")
            break
        best_accuracy = max(best_accuracy, acc)
        iteration += 1
    with open("./Results/accuracies.txt", "w") as f:
        for i, acc in enumerate(accuracies):
            f.write(f"Iteration {i+1}: Accuracy = {acc:.2f}%\n")
    torch.save(model, f"./Results/finalmodel_{best_accuracy:.2f}.pt")
    return model

# ===== RUN =====
initial_trainset, remainder = get_lowconf_and_remainder_datasets(get_model(), train_set, k_low=10000)
print(f"Initial high-confidence samples: {len(initial_trainset)}")
train_test_save(initial_trainset, n=1, epochs=100, lr=0.001)
model = load_best_model_from_folder("./Results")
final_model = train_until_degradation(model, initial_trainset, remainder, degradation_thres=0.3)
