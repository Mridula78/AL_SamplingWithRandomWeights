import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.init as init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Define Resnet model

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
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


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9]).to(device)


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

# Data preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


model = resnet56()
model = model.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)

if torch.cuda.is_available():

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Print information about each GPU
    for gpu_id in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"GPU {gpu_id}: {gpu_name}")

    # Set the default device to the first GPU (you can choose a different GPU if needed)
    torch.cuda.set_device(0)
    print("Using GPU:", torch.cuda.current_device())
else:
    print("No GPU available. Using CPU.")
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
def train_model(model, train_loader, epochs, learning_rate, accumulate_steps=1):
    best_accuracy = test_model(model, test_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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
        print(f"epoch: {epoch}, new accuracy: {new_accuracy}, loss: {loss}")
        # Check if the current accuracy is higher than the best
        if new_accuracy > best_accuracy:
            if os.path.exists(f"/home/user5/Documents/Models/HC_ResNet56/model_{best_accuracy:.2f}.pt"):
                os.remove(f"/home/user5/Documents/Models/HC_ResNet56/model_{best_accuracy:.2f}.pt")  # Delete the previous model
            best_accuracy = new_accuracy
            # Save the new model with the highest accuracy
            torch.save(model.state_dict(), f"/home/user5/Documents/Models/HC_ResNet56/model_{best_accuracy:.2f}.pt")
    return best_accuracy

def train_test_save(train_dataset, n, epochs, lr=0.01):
    best_accuracy = 0.0
    for _ in range(n):
        model = resnet56()
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        new_accuracy = train_model(model, train_loader, epochs, lr)

        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
    print(f"best accuracy: {best_accuracy}")

# ... (previous code)

def get_highconf_and_remainder_datasets(model, dataset, k):
    model.eval()
    model = model.to(device)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    confidence_scores = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs, _ = inputs.to(device), _.to(device)
            outputs = model(inputs)
            logits = outputs
            confidences = F.softmax(logits, dim=1).max(dim=1)[0]
            confidence_scores.extend(confidences.tolist())

    model.train()

    top_indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i], reverse=True)[:k]
    highconf_samples = torch.utils.data.Subset(dataset, top_indices)
    remaining_indices = [i for i in range(len(dataset)) if i not in highconf_samples.indices]
    remainder_dataset = torch.utils.data.Subset(dataset, remaining_indices)
    return highconf_samples, remainder_dataset

def get_lowconf_and_remainder_datasets(model, dataset, k_low):
    model.eval()
    model = model.to(device)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    confidence_scores = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs, _ = inputs.to(device), _.to(device)
            outputs = model(inputs)
            confidences = torch.nn.functional.softmax(outputs, dim=1).max(dim=1)[0]
            confidence_scores.extend(confidences.tolist())

    top_low_indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i], reverse=False)[:k_low]
    lowconf_samples = torch.utils.data.Subset(dataset, top_low_indices)
    remaining_indices = [i for i in range(len(dataset)) if i not in top_low_indices]
    remainder_dataset = torch.utils.data.Subset(dataset, remaining_indices)
    return lowconf_samples, remainder_dataset

def load_best_model_from_folder(folder_path):
    def get_accuracy_from_filename(filename):
        return float(filename.split("_")[1][:-3])

    model_files = os.listdir(folder_path)
    model_files = [file for file in model_files if file.startswith("model_") and file.endswith(".pt")]

    if not model_files:
        print("No model files found in the folder.")
        return None
    else:
        try:
            best_model_filename = max(model_files, key=get_accuracy_from_filename)
            best_model_path = os.path.join(folder_path, best_model_filename)
            
            
            best_model = resnet56()
            
            # Load the state_dict into the model
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))
            
            print(f"Loaded the model with the highest accuracy: {best_model_path}")
            return best_model
        except Exception as e:
            print(f"Error loading the model: {e}")
            return None

def train_until_degradation(model, train_dataset, remaining_set, degradation_thres=0.03):
    best_accuracy = test_model(model, test_loader)
    current_accuracy = 0.0
    accuracies = []
    iteration = 1

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    while len(remaining_set)!=0:
        iteration += 1
        least_conf_images, remaining_set = get_lowconf_and_remainder_datasets(model, remaining_set, k_low=5000)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, least_conf_images]) 
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)        
        current_accuracy = train_model(model,train_loader,100,0.01)
        accuracies.append(current_accuracy) #(2)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
        print(f"Current accuracy: {current_accuracy:.2f}, Best accuracy: {best_accuracy:.2f}")
        
    print("Training dataset exhausted. Stopping training.")
    print("Final accuracy on the validation set:", best_accuracy)

    print("Final accuracy on the validation set:", best_accuracy)
    with open("/home/user5/Documents/Models/HC_ResNet56/Resnet56_HC_accuracies.txt", "w") as file:
        for i, acc in enumerate(accuracies, start=1):
            file.write(f"Iteration {i}: Accuracy = {acc:.2f}\n")

    torch.save(model, f"/home/user5/Documents/Models/HC_ResNet56/finalmodel_{best_accuracy:.2f}.pt")
    return model
                                                                                                                                                         
initial_trainset, remainder = get_lowconf_and_remainder_datasets(model,train_set,k_low=10000)
print(len(initial_trainset))
train_test_save(initial_trainset, n=1, epochs=100, lr=0.01)
model = load_best_model_from_folder("/home/user5/Documents/Models/HC_ResNet56/")
final_model = train_until_degradation(model,initial_trainset,remainder,degradation_thres=0.3)

