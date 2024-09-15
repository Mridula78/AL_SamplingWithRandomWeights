import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, datasets
from torchvision import transforms
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import scipy.io
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0) 
# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Define MobileNet Model
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)




# Data preprocessing
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

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

print(f"Size of train_set: {len(train_set)}")
print(f"Size of test_set: {len(test_set)}")

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


model = model = ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 256,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dim_head = 32,
            dropout = 0.1,
            emb_dropout = 0.1
        ).to(device)

model = model.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)

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
def train_model(model, train_loader, epochs, learning_rate,path, accumulate_steps=1):
    best_accuracy = test_model(model, test_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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
            if os.path.exists(f"{path}/model_{best_accuracy:.2f}.pt"):
                os.remove(f"{path}/model_{best_accuracy:.2f}.pt")  # Delete the previous model
            best_accuracy = new_accuracy
            # Save the new model with the highest accuracy
            torch.save(model.state_dict(), f"{path}/model_{best_accuracy:.2f}.pt")
    return best_accuracy

def train_test_save(train_dataset, n, epochs,path,  lr=0.01):
    best_accuracy = 0.0
    for _ in range(n):
        model = ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 256,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dim_head = 32,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        new_accuracy = train_model(model, train_loader, epochs, lr,path)

        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
    print(f"best accuracy: {best_accuracy}")


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
            
            
            best_model = ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 256,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dim_head = 32,
            dropout = 0.1,
            emb_dropout = 0.1
        )
            
            # Load the state_dict into the model
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))
            
            print(f"Loaded the model with the highest accuracy: {best_model_path}")
            return best_model
        except Exception as e:
            print(f"Error loading the model: {e}")
            return None

def train_until_degradation(model, train_dataset, remaining_set, path):
    best_accuracy = test_model(model, test_loader)
    current_accuracy = 0.0
    accuracies = []
    iteration = 1

    while len(remaining_set)!=0:
        iteration += 1
        least_conf_images, remaining_set = get_lowconf_and_remainder_datasets(model, remaining_set, k_low=5000)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, least_conf_images])   
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)      
        current_accuracy = train_model(model,train_loader,200,0.01,path)
        accuracies.append(current_accuracy) #(2)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
        print(f"Current accuracy: {current_accuracy:.2f}, Best accuracy: {best_accuracy:.2f}")
        
    print("Training dataset exhausted. Stopping training.")
    print("Final accuracy on the validation set:", best_accuracy)

    print("Final accuracy on the validation set:", best_accuracy)
    with open(f"{path}/accuracies.txt", "w") as file:
        for i, acc in enumerate(accuracies, start=1):
            file.write(f"Iteration {i}: Accuracy = {acc:.2f}\n")

    torch.save(model, f"{path}/finalmodel_{best_accuracy:.2f}.pt")
    return model
                                                                                                                                                         
initial_trainset, remainder = get_lowconf_and_remainder_datasets(model,train_set,k_low=10000)
path = "/home/user5/Documents/Models/CHC"
print(len(initial_trainset))
train_test_save(initial_trainset, 1, 200, path, lr=0.01)
model = load_best_model_from_folder(path)
final_model = train_until_degradation(model,initial_trainset,remainder,path)