# Are Candidate Models Really Needed for Active Learning?

**Paper:** *Are Candidate Models Really Needed for Active Learning?*
Harshini Mridula Mohan, Maanya Manjunath, Vipul Arya, S.H. Shabbeer Basha, Nitin Cheekatla
*Preprint submitted to Computer Vision and Image Understanding, May 2026*

This repository contains the official code for our Deep Active Learning (DAL) framework that eliminates the need for pre-trained candidate models. We show that models with **randomly initialized weights** can achieve competitive or superior results compared to active learning methods that rely on candidate models, using three confidence-based sampling strategies: **High Confidence (HC)**, **Low Confidence (LC)**, and **High Confidence + Low Confidence (HCLC)**.

---

## Key Results

### CIFAR-10

| Model | Method | Accuracy (%) |
|---|---|---|
| DenseNet-121 | Ours (LC) | 92.87 ± 0.07 |
| DenseNet-121 | Ours (HCLC) | **93.08 ± 0.23** |
| DenseNet-121 | GLISTER | 92.86 ± 0.12 |
| DenseNet-121 | BADGE | 91.24 ± 0.09 |
| ResNet-18 | Ours (LC) | **93.53 ± 0.08** |
| ResNet-18 | Ensemble AL | 96.18 (uses candidate model) |
| VGG-16 | Ours (LC) | **94.21 ± 0.14** |
| Swin Transformer | Ours (LC) | **86.23 ± 0.10** |
| MobileNetV2 | Ours (LC/HCLC) | **94.16 ± 0.03** |

### CIFAR-100

| Model | Method | Accuracy (%) |
|---|---|---|
| DenseNet-121 | Ours (LC) | **71.65 ± 0.15** |
| DenseNet-121 | BADGE | 71.82 ± 0.20 |
| ResNet-18 | Ours (HCLC) | **73.42 ± 0.24** |
| MobileNetV2 | Ours (LC/HCLC) | **73.79 ± 0.13** |

### SVHN

| Model | Method | Accuracy (%) |
|---|---|---|
| DenseNet-121 | Ours (LC/HCLC) | **95.77 ± 0.01** |
| DenseNet-121 | BADGE | 95.80 ± 0.07 |
| ResNet-18 | Ours (LC) | **95.84 ± 0.09** |

### TinyImageNet (ResNet-18)

| Method | Accuracy (%) | Time |
|---|---|---|
| Ours (LC) | **55.99 ± 0.12** | ~30 hrs |
| BADGE | 56.73 ± 0.12 | ~45 hrs (1.5× slower) |
| GLISTER | 50.85 ± 0.02 | — |

### PASCAL VOC 2012 (Object Detection, SSD/VGG-16)

| Method | mAP |
|---|---|
| Ours (LC) | **81.53 ± 0.09** |
| Ours (HC) | 80.97 ± 0.21 |
| FF-SSD-300/512 | 80.80 |
| PPAL | 74.00 |

---

## Repository Structure

```
├── CIFAR10/
│   ├── densenet121_c10.py      # DenseNet-121 on CIFAR-10
│   ├── resnet18_c10.py         # ResNet-18 on CIFAR-10
│   ├── resnet56_c10.py         # ResNet-56 on CIFAR-10
│   ├── vgg16_c10.py            # VGG-16 on CIFAR-10
│   ├── mobilenet_c10.py        # MobileNetV2 on CIFAR-10
│   ├── swin_c10.py             # Swin Transformer on CIFAR-10
│   └── smallvit_c10.py         # ViT-Small on CIFAR-10
├── CIFAR100/
│   ├── resnet18_c100_svhn.py   # ResNet-18 on CIFAR-100 / SVHN
│   ├── resnet56_c100_svhn.py   # ResNet-56 on CIFAR-100 / SVHN
│   ├── vgg16_c100_svhn.py      # VGG-16 on CIFAR-100 / SVHN
│   └── mobilenet_c100_new.py   # MobileNetV2 on CIFAR-100
├── SVHN/
│   ├── densenet121_svhn.py     # DenseNet-121 on SVHN
│   ├── resnet18_c100_svhn.py   # ResNet-18 on SVHN
│   ├── resnet56_svhn.py        # ResNet-56 on SVHN
│   └── vgg16_svhn.py           # VGG-16 on SVHN
├── ResNet18 TinyImageNet/
│   ├── resnet18_tin_new.py     # ResNet-18 on TinyImageNet
│   └── glister_TinyImageNet_ResNet18.py  # GLISTER baseline
├── PascalVOC SSD/
│   └── pascal_voc_ssd.py       # SSD object detection on VOC
├── DinoV2 LabelMe1250K/
│   └── DinoV2_LabelMe1250K.py  # DinoV2 on LabelMe1250K
├── DenseNet121 CIFAR10/
│   └── glister_CIFAR10_DenseNet121.py   # GLISTER baseline
└── Class Imbalance VGG16/
    └── class_imbalance_vgg.py  # Class imbalance experiment
```

---

## Requirements

```bash
pip install torch torchvision numpy scipy matplotlib tqdm
pip install transformers  # for DinoV2 experiments
pip install torchmetrics  # for Pascal VOC mAP evaluation
pip install timm einops   # for Swin Transformer
```

We used Python 3.8+ and PyTorch 2.0+. All experiments were run on NVIDIA GeForce RTX 4080 (16 GB).

---

## Reproducing Results

All scripts share the same structure. Each experiment runs with 3 seeds (`42`, `789`, `101112`) and reports mean ± std. You can run all experiments or a specific one.

### CIFAR-10

**DenseNet-121** (LC, HC, HCLC):
```bash
cd CIFAR10
python densenet121_c10.py --all       # all 4 experiment variants
python densenet121_c10.py --exp 1     # LC only
```

**ResNet-18:**
```bash
python resnet18_c10.py --all
```

**ResNet-56:**
```bash
python resnet56_c10.py --all
```

**VGG-16:**
```bash
python vgg16_c10.py
```

**MobileNetV2:**
```bash
python mobilenet_c10.py --all
```

**Swin Transformer:**
```bash
python swin_c10.py --all
```

**ViT-Small:**
```bash
python smallvit_c10.py --all
```

The 4 experiment variants correspond to:

| `--exp` | Initial selection | Subsequent selection |
|---|---|---|
| 1 | Low Confidence | Low Confidence (LC) |
| 2 | High Confidence | Low Confidence (HCLC) |
| 3 | High Confidence | High Confidence (HC) |
| 4 | Low Confidence | High Confidence |

---

### CIFAR-100

**ResNet-18:**
```bash
cd CIFAR100
python resnet18_c100_svhn.py --dataset cifar100
```

**ResNet-56:**
```bash
python resnet56_c100_svhn.py --dataset cifar100
```

**VGG-16:**
```bash
python vgg16_c100_svhn.py --dataset cifar100
```

**MobileNetV2:**
```bash
python mobilenet_c100_new.py --all
```

---

### SVHN

**DenseNet-121:**
```bash
cd SVHN
python densenet121_svhn.py --all
```

**ResNet-18:**
```bash
python resnet18_c100_svhn.py --dataset svhn
```

**ResNet-56:**
```bash
python resnet56_svhn.py --all
```

**VGG-16:**
```bash
python vgg16_svhn.py --all
```

---

### TinyImageNet

Download TinyImageNet first (the script will attempt to auto-download from Stanford):

```bash
cd "ResNet18 TinyImageNet"
python resnet18_tin_new.py --all --data-dir ./tiny-imagenet-200
```

Or run a specific experiment:
```bash
python resnet18_tin_new.py --exp 1 --data-dir ./tiny-imagenet-200
```

---

### Pascal VOC 2012 (Object Detection)

Download Pascal VOC 2012 and place it at `./data/VOCdevkit/VOC2012`, then:

```bash
cd "PascalVOC SSD"
python pascal_voc_ssd.py
```

For multiple seeds:
```bash
python pascal_voc_ssd.py --seeds 3
```

---

### DinoV2 on LabelMe1250K (Foundation Model Ablation)

Download LabelMe1250K and set `dataset_path` in the script, then:

```bash
cd "DinoV2 LabelMe1250K"
python DinoV2_LabelMe1250K.py
```

This runs the non-pretrained (random weights) DinoV2 experiment. To run the pretrained variant, uncomment `acc1 = run_experiment("exp1_pretrained_dinov2", pretrained=True)` in the `__main__` block.

---

### Reproducing GLISTER Baselines

**DenseNet-121 on CIFAR-10:**
```bash
cd "DenseNet121 CIFAR10"
python glister_CIFAR10_DenseNet121.py --num_runs 3 --epochs_per_round 50
```

**ResNet-18 on TinyImageNet:**
```bash
cd "ResNet18 TinyImageNet"
python glister_TinyImageNet_ResNet18.py --num_runs 3 --epochs_per_round 100
```

---

### Class Imbalance Experiment

```bash
cd "Class Imbalance VGG16"
python class_imbalance_vgg.py
```

This creates a 10:1 imbalanced CIFAR-10 variant and evaluates LC, HCLC, and HC sampling methods.

---

## How the Pipeline Works

The active learning pipeline works as follows for all experiments:

1. **Initialize** a model with **random weights** (no pre-training).
2. **Select initial samples** (10K or 4% of dataset) using the chosen confidence criterion (LC or HC) from the untrained model's output.
3. **Train** the model on this initial labeled set (100 epochs).
4. **Iteratively**: select the next batch (5K or 5% of dataset) from remaining unlabeled data using the trained model's confidence scores, add to the labeled pool, and retrain.
5. **Repeat** until the annotation budget is exhausted.

The key insight is that Step 1 uses **no pre-trained candidate model** — confidence scores from a randomly initialized network are sufficient to guide informative sample selection.

---

## Sampling Strategies

| Strategy | Formula | Description |
|---|---|---|
| High Confidence (HC) | `max_k P(y=k\|x)` | Selects samples the model is most certain about |
| Low Confidence (LC) | `1 - max_k P(y=k\|x)` | Selects samples the model is most uncertain about |
| HCLC | HC initially, then LC | Starts with reliable samples, then shifts to uncertain ones |

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{mohan2026candidate,
  title={Are Candidate Models Really Needed for Active Learning?},
  author={Mohan, Harshini Mridula and Manjunath, Maanya and Arya, Vipul and Basha, S.H. Shabbeer and Cheekatla, Nitin},
  journal={Computer Vision and Image Understanding},
  year={2026}
}
```

