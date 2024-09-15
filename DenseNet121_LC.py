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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(2)
# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

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

# Initialize DenseNet-121 model
model = DenseNet121()
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
    torch.cuda.set_device(2)
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
            if os.path.exists(f"/home/user5/Documents/Models/LC_Densenet/model_{best_accuracy:.2f}.pt"):
                os.remove(f"/home/user5/Documents/Models/LC_Densenet/model_{best_accuracy:.2f}.pt")  # Delete the previous model
            best_accuracy = new_accuracy
            # Save the new model with the highest accuracy
            torch.save(model.state_dict(), f"/home/user5/Documents/Models/LC_Densenet/model_{best_accuracy:.2f}.pt")
    return best_accuracy

def train_test_save(train_dataset, n, epochs, lr=0.01):
    best_accuracy = 0.0
    for _ in range(n):
        model = DenseNet121().to(device)
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
            
            # Instantiate the DenseNet121 model
            best_model = DenseNet121()
            
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
        #print("length of dataset:",len(train_dataset))       
        current_accuracy = train_model(model,train_loader,100,0.01)
        accuracies.append(current_accuracy) #(2)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
        print(f"Current accuracy: {current_accuracy:.2f}, Best accuracy: {best_accuracy:.2f}")
        
    print("Training dataset exhausted. Stopping training.")
    print("Final accuracy on the validation set:", best_accuracy)

    print("Final accuracy on the validation set:", best_accuracy)
    with open("/home/user5/Documents/Models/LC_Densenet/Densenet_LC_accuracies.txt", "w") as file:
        for i, acc in enumerate(accuracies, start=1):
            file.write(f"Iteration {i}: Accuracy = {acc:.2f}\n")

    torch.save(model, f"/home/user5/Documents/Models/LC_Densenet/finalmodel_{best_accuracy:.2f}.pt")
    return model
                                                                                                                                                         
initial_trainset, remainder = get_lowconf_and_remainder_datasets(model,train_set,k_low=10000)
print(len(initial_trainset))
train_test_save(initial_trainset, n=1, epochs=100, lr=0.01)
model = load_best_model_from_folder("/home/user5/Documents/Models/LC_Densenet/")
final_model = train_until_degradation(model,initial_trainset,remainder,degradation_thres=0.3)



