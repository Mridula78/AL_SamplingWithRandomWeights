import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import xml.etree.ElementTree as ET
from torchmetrics.detection import MeanAveragePrecision
from PIL import Image
import random
import warnings
import json
import time
import argparse

warnings.filterwarnings('ignore')

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# VOC Classes
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
NUM_CLASSES = len(VOC_CLASSES) + 1  # +1 for background

# ==================== DATASET LOADER ====================

class PascalVOCDataset(Dataset):
    """Custom Pascal VOC Dataset loader"""
    def __init__(self, root_dir, image_set='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_set = image_set
        
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotation_dir = os.path.join(root_dir, 'Annotations')
        
        image_sets_dir = os.path.join(root_dir, 'ImageSets', 'Main')
        image_set_file = os.path.join(image_sets_dir, image_set + '.txt')
        
        with open(image_set_file, 'r') as f:
            image_names = [line.strip().split()[0] for line in f.readlines()]
        
        self.images = []
        self.annotations = []
        
        for name in image_names:
            img_path = os.path.join(self.image_dir, name + '.jpg')
            xml_path = os.path.join(self.annotation_dir, name + '.xml')
            
            if not os.path.exists(xml_path):
                continue
                
            self.images.append(img_path)
            self.annotations.append(xml_path)
        
        print(f"Loaded {len(self.images)} images for {image_set} set")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        xml_path = self.annotations[idx]
        boxes = []
        labels = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in VOC_CLASSES:
                    continue
                    
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                    
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(VOC_CLASSES.index(name))
        
        except Exception as e:
            print(f"Error parsing XML {xml_path}: {e}")
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        if self.transform:
            image = self.transform(image)
            
        return image, target

def collate_fn(batch):
    """Collate function for DataLoader"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets

def prepare_data(data_path='./data/VOCdevkit/VOC2012'):
    """Prepare Pascal VOC dataset"""
    print(f'==> Preparing Pascal VOC data from {data_path}')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = PascalVOCDataset(
        root_dir=data_path,
        image_set='trainval',
        transform=transform
    )
    
    test_dataset = PascalVOCDataset(
        root_dir=data_path,
        image_set='val',
        transform=transform
    )
    
    print(f"Training set: {len(train_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")
    
    return train_dataset, test_dataset

# ==================== MODEL FUNCTIONS ====================

def create_model():
    """Create SSD model with MobileNetV3 backbone"""
    try:
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model = ssdlite320_mobilenet_v3_large(weights=weights, num_classes=NUM_CLASSES)
        print("Loaded SSDLite320 with MobileNetV3 backbone pretrained on COCO")
    except:
        print("Using SSDLite320 with MobileNetV3 backbone (no pretraining)")
        model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=NUM_CLASSES)
    
    model = model.to(device)
    return model

def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for images, targets in train_loader:
        if len(images) == 0:
            continue
            
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
    else:
        avg_loss = 0
    
    return avg_loss

def evaluate_model(model, test_loader):
    """Evaluate model using mAP metric"""
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
    
    with torch.no_grad():
        for images, targets in test_loader:
            if len(images) == 0:
                continue
                
            images = list(image.to(device) for image in images)
            predictions = model(images)
            
            formatted_preds = []
            formatted_targets = []
            
            for i in range(len(images)):
                pred = predictions[i]
                targ = targets[i]
                
                keep = pred['scores'] > 0.01  # Filter low confidence predictions
                
                formatted_preds.append({
                    'boxes': pred['boxes'][keep].cpu(),
                    'scores': pred['scores'][keep].cpu(),
                    'labels': pred['labels'][keep].cpu()
                })
                
                formatted_targets.append({
                    'boxes': targ['boxes'].cpu(),
                    'labels': targ['labels'].cpu()
                })
            
            metric.update(formatted_preds, formatted_targets)
    
    results = metric.compute()
    map_score = results['map'].item()
    
    return map_score

# ==================== EXPERIMENT FUNCTIONS ====================

def get_random_samples(dataset, k):
    """Get k random samples from dataset"""
    indices = np.random.choice(len(dataset), k, replace=False)
    selected = Subset(dataset, indices)
    
    remaining_indices = [i for i in range(len(dataset)) if i not in indices]
    remaining = Subset(dataset, remaining_indices)
    
    return selected, remaining

def run_pascalvoc_experiment():
    """Run experiment on PascalVOC with SSD"""
    print("="*80)
    print("SSD - PASCAL VOC EXPERIMENT")
    print("="*80)
    
    # Set random seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Configuration
    TOTAL_TRAIN_SAMPLES = 5717  # Pascal VOC 2012 trainval set size
    INITIAL_SAMPLES = 10000  # Starting point
    BATCH_PERCENT = 0.05      # 5% per iteration
    EPOCHS_PER_ITER = 2       # 2 epochs per iteration
    BATCH_SIZE = 8           # Batch size for training
    
    print(f"Seed: {SEED}")
    print(f"Device: {device}")
    print(f"Initial samples: {INITIAL_SAMPLES}")
    print(f"Batch percent per iteration: {BATCH_PERCENT*100}%")
    print(f"Epochs per iteration: {EPOCHS_PER_ITER}")
    
    # Load data
    print("\nLoading Pascal VOC dataset...")
    train_set, test_set = prepare_data()
    
    if len(train_set) == 0:
        print("Error: No training data loaded!")
        return None
    
    total_samples = len(train_set)
    batch_size = int(total_samples * BATCH_PERCENT)
    
    print(f"\nDataset Statistics:")
    print(f"  Total training samples: {total_samples:,}")
    print(f"  Test samples: {len(test_set):,}")
    print(f"  Initial samples: {INITIAL_SAMPLES:,}")
    print(f"  Batch per iteration: {batch_size:,} samples ({BATCH_PERCENT*100:.0f}%)")
    
    # Create model
    print("\nCreating SSD model...")
    model = create_model()
    
    # Create test loader
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, 
                            collate_fn=collate_fn, shuffle=False, num_workers=2)
    
    # Step 1: Get initial random samples
    print("\n" + "-"*80)
    print("STEP 1: Selecting initial random samples...")
    
    initial_samples = min(INITIAL_SAMPLES, total_samples)
    train_dataset, remaining_set = get_random_samples(train_set, initial_samples)
    
    print(f"  Selected {len(train_dataset):,} initial samples")
    print(f"  Remaining samples: {len(remaining_set):,}")
    
    # Step 2: Burn-in training (2 epochs on initial samples)
    print("\n" + "-"*80)
    print("STEP 2: Burn-in training (2 epochs on initial samples)...")
    
    # Setup optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=5e-4)
    
    burnin_start = time.time()
    
    for epoch in range(EPOCHS_PER_ITER):
        epoch_start = time.time()
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                 collate_fn=collate_fn, shuffle=True, num_workers=2)
        
        avg_loss = train_epoch(model, train_loader, optimizer, None)
        epoch_time = time.time() - epoch_start
        
        print(f"  Epoch {epoch+1}/{EPOCHS_PER_ITER}: "
              f"Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    
    burnin_time = time.time() - burnin_start
    burnin_map = evaluate_model(model, test_loader)
    
    print(f"\nBurn-in completed!")
    print(f"  Time: {burnin_time:.2f} seconds ({burnin_time/60:.2f} minutes)")
    print(f"  mAP after burn-in: {burnin_map:.4f}")
    
    # Step 3: Iterative training with additional samples
    print("\n" + "-"*80)
    print("STEP 3: Iterative training with additional samples...")
    
    total_time = burnin_time
    iteration = 0
    iteration_results = []
    
    while len(remaining_set) > 0:
        iteration += 1
        iteration_start = time.time()
        
        # Get next batch (5% of total data or remaining if less)
        current_batch_size = min(batch_size, len(remaining_set))
        new_samples, remaining_set = get_random_samples(remaining_set, current_batch_size)
        
        # Add to training dataset
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, new_samples])
        
        print(f"\n  Iteration {iteration}:")
        print(f"    Adding {current_batch_size:,} new samples")
        print(f"    Total training samples: {len(train_dataset):,}")
        print(f"    Remaining samples: {len(remaining_set):,}")
        
        # Train for 2 epochs
        iteration_epoch_maps = []
        
        for epoch in range(EPOCHS_PER_ITER):
            epoch_start = time.time()
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                     collate_fn=collate_fn, shuffle=True, num_workers=2)
            
            avg_loss = train_epoch(model, train_loader, optimizer, None)
            epoch_time = time.time() - epoch_start
            
            # Evaluate every epoch
            current_map = evaluate_model(model, test_loader)
            iteration_epoch_maps.append(current_map)
            
            print(f"    Epoch {epoch+1}/{EPOCHS_PER_ITER}: "
                  f"Loss: {avg_loss:.4f}, mAP: {current_map:.4f}, Time: {epoch_time:.2f}s")
        
        iteration_time = time.time() - iteration_start
        total_time += iteration_time
        current_map = iteration_epoch_maps[-1]  # mAP after last epoch
        
        iteration_results.append({
            'iteration': iteration,
            'samples_added': current_batch_size,
            'total_samples': len(train_dataset),
            'iteration_time': iteration_time,
            'map': current_map,
            'epoch_maps': iteration_epoch_maps
        })
        
        print(f"    Iteration completed in {iteration_time:.2f}s ({iteration_time/60:.2f} min)")
        print(f"    Current mAP: {current_map:.4f}")
    
    # Final evaluation
    final_map = evaluate_model(model, test_loader)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED!")
    print("="*80)
    
    # Calculate statistics
    total_iterations = len(iteration_results)
    total_samples_used = len(train_dataset)
    
    print(f"\nFinal Results:")
    print(f"  Burn-in time: {burnin_time:.2f} seconds ({burnin_time/60:.2f} minutes, {burnin_time/3600:.2f} hours)")
    print(f"  Total annotation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes, {total_time/3600:.2f} hours)")
    print(f"  Burn-in mAP: {burnin_map:.4f}")
    print(f"  Final mAP: {final_map:.4f}")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Total samples used: {total_samples_used:,} ({total_samples_used/total_samples*100:.1f}% of dataset)")
    print(f"  mAP improvement: {final_map - burnin_map:.4f}")
    
    # Save model
    os.makedirs("./results_pascalvoc", exist_ok=True)
    model_save_path = f"./results_pascalvoc/ssd_pascalvoc_final_{final_map:.4f}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    # Prepare comprehensive results
    results = {
        'model': 'SSD with MobileNetV3',
        'dataset': 'PascalVOC',
        'seed': SEED,
        'initial_samples': INITIAL_SAMPLES,
        'batch_percent': BATCH_PERCENT,
        'epochs_per_iter': EPOCHS_PER_ITER,
        'burnin_time_seconds': burnin_time,
        'burnin_time_minutes': burnin_time / 60,
        'burnin_time_hours': burnin_time / 3600,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'total_time_hours': total_time / 3600,
        'burnin_map': burnin_map,
        'final_map': final_map,
        'map_improvement': final_map - burnin_map,
        'total_iterations': total_iterations,
        'total_samples_used': total_samples_used,
        'percentage_used': total_samples_used / total_samples * 100,
        'iteration_results': iteration_results,
        'hardware_info': {
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    }
    
    # Save results to JSON
    results_json_path = "./results_pascalvoc/ssd_pascalvoc_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    print(f"Results saved to: {results_json_path}")
    
    # Save results to CSV
    results_csv_path = "./results_pascalvoc/ssd_pascalvoc_results.csv"
    with open(results_csv_path, 'w') as f:
        f.write("Iteration,SamplesAdded,TotalSamples,IterationTime(s),IterationTime(min),mAP\n")
        for res in iteration_results:
            f.write(f"{res['iteration']},{res['samples_added']},{res['total_samples']},"
                   f"{res['iteration_time']:.2f},{res['iteration_time']/60:.2f},{res['map']:.4f}\n")
        f.write(f"\nSummary,BurninTime(s),{burnin_time:.2f}\n")
        f.write(f"Summary,BurninTime(min),{burnin_time/60:.2f}\n")
        f.write(f"Summary,BurninTime(hours),{burnin_time/3600:.2f}\n")
        f.write(f"Summary,TotalTime(s),{total_time:.2f}\n")
        f.write(f"Summary,TotalTime(min),{total_time/60:.2f}\n")
        f.write(f"Summary,TotalTime(hours),{total_time/3600:.2f}\n")
        f.write(f"Summary,BurninmAP,{burnin_map:.4f}\n")
        f.write(f"Summary,FinalmAP,{final_map:.4f}\n")
        f.write(f"Summary,mAPImprovement,{final_map - burnin_map:.4f}\n")
    print(f"CSV results saved to: {results_csv_path}")
    
    # Print final summary in requested format
    print("\n" + "="*80)
    print("FINAL SUMMARY (Requested Format)")
    print("="*80)
    print(f"Model: SSD with MobileNetV3")
    print(f"Dataset: PascalVOC")
    print(f"Final mAP: {final_map:.4f}")
    print(f"Burn-in Time: {burnin_time/60:.2f} minutes ({burnin_time/3600:.2f} hours)")
    print(f"Total Annotation Time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print("="*80)
    
    return results

def run_multiple_seeds(num_seeds=3):
    """Run experiment with multiple seeds for statistical significance"""
    seeds = [42, 123, 456]
    all_results = []
    
    print("="*80)
    print(f"RUNNING {num_seeds} SEEDS FOR PASCALVOC EXPERIMENT")
    print("="*80)
    
    for i, seed in enumerate(seeds[:num_seeds]):
        print(f"\n{'='*60}")
        print(f"SEED {i+1}/{num_seeds}: {seed}")
        print(f"{'='*60}")
        
        # Set seed
        global SEED
        SEED = seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        
        # Run experiment
        results = run_pascalvoc_experiment()
        if results:
            all_results.append(results)
    
    # Calculate statistics across seeds
    if all_results:
        print("\n" + "="*80)
        print("STATISTICS ACROSS SEEDS")
        print("="*80)
        
        burnin_times = [r['burnin_time_hours'] for r in all_results]
        total_times = [r['total_time_hours'] for r in all_results]
        final_maps = [r['final_map'] for r in all_results]
        
        mean_burnin = np.mean(burnin_times)
        std_burnin = np.std(burnin_times)
        mean_total = np.mean(total_times)
        std_total = np.std(total_times)
        mean_map = np.mean(final_maps)
        std_map = np.std(final_maps)
        
        print(f"\nBurn-in Time: {mean_burnin:.2f} ± {std_burnin:.2f} hours")
        print(f"Total Annotation Time: {mean_total:.2f} ± {std_total:.2f} hours")
        print(f"Final mAP: {mean_map:.4f} ± {std_map:.4f}")
        
        # Save combined results
        combined_results = {
            'seeds': seeds[:num_seeds],
            'mean_burnin_time_hours': mean_burnin,
            'std_burnin_time_hours': std_burnin,
            'mean_total_time_hours': mean_total,
            'std_total_time_hours': std_total,
            'mean_final_map': mean_map,
            'std_final_map': std_map,
            'detailed_results': all_results
        }
        
        combined_path = "./results_pascalvoc/ssd_pascalvoc_combined_results.json"
        with open(combined_path, 'w') as f:
            json.dump(combined_results, f, indent=4, default=str)
        
        print(f"\nCombined results saved to: {combined_path}")
        
        return combined_results
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Run SSD experiment on PascalVOC')
    parser.add_argument('--seeds', type=int, default=1,
                       help='Number of seeds to run (default: 1)')
    parser.add_argument('--data-path', type=str, default='./data/VOCdevkit/VOC2012',
                       help='Path to PascalVOC dataset')
    
    args = parser.parse_args()
    
    if args.seeds > 1:
        run_multiple_seeds(args.seeds)
    else:
        run_pascalvoc_experiment()

if __name__ == "__main__":
    main()