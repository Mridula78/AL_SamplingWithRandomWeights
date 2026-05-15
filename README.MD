# Are Candidate Models Really Needed for Active Learning?

**Paper:** *Are Candidate Models Really Needed for Active Learning?*
Harshini Mridula Mohan, Maanya Manjunath, Vipul Arya, S.H. Shabbeer Basha, Nitin Cheekatla
*Preprint submitted to Computer Vision and Image Understanding, May 2026*

This repository contains the official implementation of our Deep Active Learning (DAL) framework. We demonstrate that models with **randomly initialized weights** can achieve competitive or superior performance compared to active learning methods that rely on pre-trained candidate models — eliminating the computational overhead of candidate model training entirely.

We evaluate three confidence-based sampling strategies:
- **HC** — High Confidence: selects samples the model is most certain about
- **LC** — Low Confidence: selects samples the model is most uncertain about
- **HCLC** — High Confidence initially, then Low Confidence in subsequent rounds

---

## Results

### Table 1: CIFAR-10

| Model | Method | Candidate Model? | Time Saved (hrs) | Accuracy (%) |
|---|---|---|---|---|
| **DenseNet-121** | Gaussian Switch Sampling | Yes | 0.00 | 72.00 |
| | Triplet AL | Yes | 0.00 | 76.00 |
| | Uncertainty Estimation | Yes | 0.00 | 91.00 |
| | TLC | Yes | 0.00 | 85.96 |
| | Yun et al. (2020) | Yes | 0.00 | 91.70 |
| | Rakesh & Jain (2021) | Yes | 0.00 | 82.00 |
| | Band-limited Training | No | — | 92.00 |
| | GLISTER | Yes | 0.00 | 92.86 ± 0.12 |
| | BADGE | Yes | 0.00 | 91.24 ± 0.09 |
| | BAIT | Yes | 0.00 | 70.91 ± 0.20 |
| | Beluch et al. (2018) | Yes | 0.00 | 91.20 |
| | COPS (10K) | Yes | 0.00 | 80.50 |
| | **Ours (LC) (10K)** | **No** | **0.16** | **91.37 ± 0.20** |
| | **Ours (LC)** | **No** | **0.16** | **92.87 ± 0.07** |
| | **Ours (HC)** | **No** | **0.16** | **92.72 ± 0.16** |
| | **Ours (HCLC)** | **No** | **0.16** | **93.08 ± 0.23** |
| **ResNet-56** | Double-Cross Attacks | Yes | 0.00 | 78.44 |
| | Superconvergence | No | — | 91.10 |
| | Adversarial Sampling | Yes | 0.00 | 89.00 |
| | PrAC | Yes | 0.00 | 92.00 |
| | PruneFuse | Yes | 0.00 | 93.65 |
| | COPS (10K) | Yes | 0.00 | 74.50 |
| | **Ours (LC)** | **No** | **0.26** | **91.60 ± 0.12** |
| | **Ours (HC)** | **No** | **0.26** | **91.30 ± 0.09** |
| | **Ours (HCLC)** | **No** | **0.26** | **91.60 ± 0.12** |
| **VGG-16** | FF-Active | Yes | 0.00 | 91.00 |
| | VAAL | Yes | 0.00 | 90.16 |
| | Coreset-WSM | Yes | 0.00 | 90.00 |
| | Coreset-FSM | Yes | 0.00 | 90.00 |
| | BADGE (40K) | Yes | 0.00 | 82.00 |
| | Jung et al. (2023) (25K) | Yes | 0.00 | 81.44 |
| | DAL (25K) | Yes | 0.00 | 84.00 |
| | **Ours (LC) (20K)** | **No** | **0.5** | **84.26** |
| | **Ours (LC) (40K)** | **No** | **0.5** | **91.89** |
| | **Ours (LC)** | **No** | **0.5** | **94.21 ± 0.14** |
| | **Ours (HC)** | **No** | **0.5** | **93.94 ± 0.19** |
| | **Ours (HCLC)** | **No** | **0.5** | **94.20 ± 0.11** |
| **ResNet-18** | DEAL | Yes | 0.00 | 88.00 |
| | VAAL | Yes | 0.00 | 81.00 |
| | ST-CoNAL (5K) | Yes | 0.00 | 83.05 |
| | Yun et al. (2020) (10K) | Yes | 0.00 | 90.43 |
| | ActiveLossNet (10K) | Yes | 0.00 | 77.22 |
| | CoreGCN (10K) | Yes | 0.00 | 91.50 |
| | UncertainGCN (10K) | Yes | 0.00 | 91.00 |
| | BADGE (40K) | Yes | 0.00 | 59.00 |
| | Ensemble AL | Yes | 0.00 | 96.18 |
| | Gaussian Switch Sampling | Yes | 0.00 | 66.11 |
| | SSAL | Yes | 0.00 | 92.70 |
| | **Ours (LC) (5K)** | **No** | **0.25** | **81.27 ± 0.12** |
| | **Ours (LC) (10K)** | **No** | **0.25** | **90.12 ± 0.07** |
| | **Ours (LC) (40K)** | **No** | **0.25** | **92.69 ± 0.09** |
| | **Ours (LC)** | **No** | **0.25** | **93.53 ± 0.08** |
| | **Ours (HC)** | **No** | **0.25** | **93.28 ± 0.03** |
| | **Ours (HCLC)** | **No** | **0.25** | **93.48 ± 0.12** |
| **Swin Transformer** | MESA | Yes | 0.00 | 81.30 |
| | NHT | No | — | 78.00 |
| | **Ours (LC)** | **No** | **0.06** | **86.23 ± 0.10** |
| | **Ours (HC)** | **No** | **0.06** | **83.88 ± 0.35** |
| | **Ours (HCLC)** | **No** | **0.06** | **85.80 ± 0.02** |
| **ViT-Small** | Knowledge Distillation | Yes | 0.00 | 73.80 |
| | Bootstrapping ViTs | Yes | 0.00 | 87.32 |
| | Interpretability-Aware ViT | Yes | 0.00 | 88.93 |
| | **Ours (LC)** | **No** | **0.52** | **83.92 ± 0.36** |
| | **Ours (HC)** | **No** | **0.52** | **82.70 ± 0.18** |
| | **Ours (HCLC)** | **No** | **0.52** | **83.81 ± 0.21** |
| **MobileNetV2** | FPGA | Yes | 0.00 | 86.00 |
| | AOT | Yes | 0.00 | 80.00 |
| | PA-CL | Yes | 0.00 | 82.00 |
| | COPS (10K) | Yes | 0.00 | 74.00 |
| | **Ours (LC) (10K)** | **No** | **0.52** | **82.53** |
| | **Ours (LC)** | **No** | **0.20** | **94.16 ± 0.03** |
| | **Ours (HC)** | **No** | **0.20** | **92.84 ± 0.23** |
| | **Ours (HCLC)** | **No** | **0.20** | **94.16 ± 0.03** |
| **CNN (6-layer)** | LAL-IGradV-VAE | Yes | 0.00 | 83.10 |

---

### Table 2: CIFAR-100

| Model | Method | Candidate Model? | Time Saved (hrs) | Accuracy (%) |
|---|---|---|---|---|
| **DenseNet-121** | Bayesian Neural Networks | Yes | 0.00 | 60.00 |
| | GLISTER | Yes | 0.00 | 68.36 ± 0.12 |
| | BADGE | Yes | 0.00 | 71.82 ± 0.20 |
| | BAIT | Yes | 0.00 | 65.91 ± 0.19 |
| | Rakesh & Jain (2021) (10K) | Yes | — | 59.00 |
| | TLC | Yes | 0.00 | 62.05 |
| | Yun et al. (2020) (10K) | Yes | 0.00 | 57.43 |
| | **Ours (LC) (10K)** | **No** | **0.19** | **59.03** |
| | **Ours (LC)** | **No** | **0.19** | **71.65 ± 0.15** |
| | **Ours (HC)** | **No** | **0.19** | **70.98 ± 0.19** |
| | **Ours (HCLC)** | **No** | **0.19** | **71.33 ± 0.24** |
| **ResNet-56** | Superconvergence | No | — | 60.80 |
| | Adversarial Sampling | Yes | 0.00 | 57.50 |
| | Passive Batch Injection Training | No | — | 71.60 |
| | PrAC | Yes | 0.00 | 71.20 |
| | PruneFuse | Yes | 0.00 | 67.87 |
| | Linear Feature Disentanglement | No | — | 70.73 |
| | **Ours (LC)** | **No** | **0.31** | **66.22 ± 0.21** |
| | **Ours (HC)** | **No** | **0.31** | **66.30 ± 0.47** |
| | **Ours (HCLC)** | **No** | **0.31** | **66.39 ± 0.06** |
| **VGG-16** | FF-Active | Yes | 0.00 | 66.00 |
| | VAAL | Yes | 0.00 | 47.60 |
| | Coreset-WSM | Yes | 0.00 | 70.00 |
| | Coreset-FSM | Yes | 0.00 | 65.00 |
| | FF-Active (variant) | Yes | 0.00 | 67.00 |
| | **Ours (LC)** | **No** | **0.57** | **66.46 ± 0.63** |
| | **Ours (HC)** | **No** | **0.57** | **64.87 ± 0.35** |
| | **Ours (HCLC)** | **No** | **0.57** | **66.11 ± 0.16** |
| **ResNet-18** | Influence Selection | Yes | 0.00 | 64.00 |
| | active-iNAS | Yes | 0.00 | 70.50 |
| | ST-CoNAL (10K) | Yes | 0.00 | 57.49 |
| | CoreGCN (10K) | Yes | 0.00 | 56.50 |
| | UncertainGCN (10K) | Yes | 0.00 | 55.00 |
| | Yun et al. (2020) (10K) | Yes | 0.00 | 59.73 |
| | Ensemble AL | Yes | 0.00 | 81.86 |
| | **Ours (LC) (10K)** | **No** | **0.10** | **59.01 ± 0.22** |
| | **Ours (LC)** | **No** | **0.10** | **73.24 ± 0.19** |
| | **Ours (HC)** | **No** | **0.10** | **71.97 ± 0.05** |
| | **Ours (HCLC)** | **No** | **0.10** | **73.42 ± 0.24** |
| **MobileNetV2** | LAL-IGradV-VAE | Yes | 0.00 | 64.50 |
| | GLISTER | Yes | 0.00 | 64.78 |
| | Coreset | Yes | 0.00 | 66.27 |
| | Moderate Coreset | Yes | 0.00 | 64.45 |
| | **Ours (LC)** | **No** | **0.25** | **73.79 ± 0.13** |
| | **Ours (HC)** | **No** | **0.25** | **72.82 ± 0.13** |
| | **Ours (HCLC)** | **No** | **0.25** | **73.79 ± 0.13** |

---

### Table 3: SVHN

| Model | Method | Candidate Model? | Time Saved (hrs) | Accuracy (%) |
|---|---|---|---|---|
| **DenseNet-121** | ALFA-Mix (large budget) | Yes | 0.00 | 90.00 |
| | CUCB | Yes | 0.00 | 94.00 |
| | GLISTER | Yes | 0.00 | 91.26 ± 0.11 |
| | BADGE | Yes | 0.00 | 95.80 ± 0.07 |
| | BAIT | Yes | 0.00 | 91.37 ± 0.14 |
| | **Ours (LC)** | **No** | **0.94** | **95.77 ± 0.01** |
| | **Ours (HC)** | **No** | **0.94** | **95.48 ± 0.15** |
| | **Ours (HCLC)** | **No** | **0.94** | **95.76 ± 0.08** |
| **ResNet-56** | Double-Cross Attacks | Yes | 0.00 | 67.19 |
| | Linear Feature Disentanglement | No | — | 96.66 |
| | **Ours (LC)** | **No** | **0.35** | **96.12 ± 0.11** |
| | **Ours (HC)** | **No** | **0.35** | **95.99 ± 0.05** |
| | **Ours (HCLC)** | **No** | **0.35** | **96.12 ± 0.11** |
| **VGG-16** | Pretext-based AL | Yes | 0.00 | 94.00 |
| | BADGE (50K) | Yes | 0.00 | 92.00 |
| | RL-based AL | Yes | 0.00 | 97.00 |
| | BAL at Scale | Yes | 0.00 | 93.00 |
| | Co-Auxiliary Learning | Yes | 0.00 | 96.00 |
| | **Ours (LC) (50K)** | **No** | **0.27** | **94.22** |
| | **Ours (LC)** | **No** | **0.27** | **95.51 ± 0.08** |
| | **Ours (HC)** | **No** | **0.27** | **95.45 ± 0.07** |
| | **Ours (HCLC)** | **No** | **0.27** | **95.61 ± 0.09** |
| **ResNet-18** | Beck et al. (2021) | Yes | 0.00 | 95.10 |
| | ActiveLossNet (10K) | Yes | 0.00 | 91.30 |
| | BAIT (15K) | Yes | 0.00 | 75.00 |
| | BADGE (50K) | Yes | 0.00 | 91.00 |
| | Random baseline (10K) | No | 0.00 | 90.50 |
| | VAAL (10K) | Yes | 0.00 | 92.00 |
| | Learning Loss (10K) | Yes | 0.00 | 92.50 |
| | CoreGCN (10K) | Yes | 0.00 | 92.50 |
| | UncertainGCN (10K) | Yes | 0.00 | 92.20 |
| | Coreset (10K) | Yes | 0.00 | 93.00 |
| | FeatProp (10K) | Yes | 0.00 | 90.50 |
| | **Ours (LC) (15K)** | **No** | **0.29** | **91.80 ± 0.06** |
| | **Ours (LC) (50K)** | **No** | **0.29** | **93.23 ± 0.30** |
| | **Ours (LC)** | **No** | **0.29** | **95.84 ± 0.09** |
| | **Ours (HC)** | **No** | **0.29** | **95.65 ± 0.08** |
| | **Ours (HCLC)** | **No** | **0.29** | **95.83 ± 0.02** |

---

### Table 4: TinyImageNet (ResNet-18)

| Method | Candidate Model? | Time Saved (hrs) | Annotation Sim. Time (hrs) | Accuracy (%) |
|---|---|---|---|---|
| Sinha et al. (2021) | Yes | 0.00 | — | 35.00 |
| Hu et al. (2021) | Yes | 0.00 | — | 52.00 |
| Yoo et al. (2018) | Yes | 0.00 | — | 42.70 |
| Random baseline (10K) | No | — | — | 34.50 |
| VAAL (10K) | Yes | 0.00 | — | 39.80 |
| Learning Loss (10K) | Yes | 0.00 | — | 42.10 |
| Coreset (50K) | Yes | 0.00 | — | 45.20 |
| ALPF (50K) | Yes | 0.00 | — | 42.70 |
| FeatProp (10K) | Yes | 0.00 | — | 41.80 |
| Jung et al. (2023) | Yes | 0.00 | — | 39.79 |
| Hu et al. (2019) | Yes | 0.00 | — | 44.10 |
| Bengar et al. (2021) | Yes | 0.00 | — | 51.00 |
| Progressive AL | Yes | 0.00 | — | 52.10 |
| Bengar et al. (2022) | Yes | 0.00 | — | 56.00 |
| DQAS | Yes | 0.00 | — | 53.42 |
| BUAL | Yes | 0.00 | — | 46.50 |
| BADGE | Yes | 0.00 | 44.66 | 56.73 ± 0.12 |
| BAIT | Yes | 0.00 | 53.92 | 34.13 ± 0.03 |
| GLISTER | Yes | 0.00 | 27.21 | 50.85 ± 0.02 |
| SSAL | Yes | 0.00 | — | 54.50 |
| LC (Pre-trained) | Yes | 0.00 | 29.87 | 56.43 ± 0.12 |
| **Ours (LC)** | **No** | **1.20** | **29.87** | **55.99 ± 0.12** |
| **Ours (HC)** | **No** | **1.20** | **29.87** | **54.94 ± 0.14** |
| **Ours (HCLC)** | **No** | **1.20** | **29.87** | **55.92 ± 0.11** |

> BADGE took ~45 hrs — 1.5× longer than our methods — for only a marginal accuracy gain of 0.74%.

---

### Table 5: PASCAL VOC 2012 — Object Detection (SSD / VGG-16)

| Method | SSD Variant | Candidate Model? | Time Saved (hrs) | mAP |
|---|---|---|---|---|
| SSD-300 | SSD-300 / VGG-16 | No | — | 73.10 |
| FSSD-300 | FSSD-300 / VGG-16 | Yes | 0.00 | 82.70 |
| FF-SSD-300/512 | FF-SSD / VGG-16 | No | — | 80.80 |
| Concat-SSD-300/512 | Concat-SSD / VGG-16 | No | — | 80.80 |
| PPAL | ISD-SSD / VGG-16 | Yes | 0.00 | 74.00 |
| **Ours (LC)** | **SSD / VGG-16** | **No** | **2.58** | **81.53 ± 0.09** |
| **Ours (HC)** | **SSD / VGG-16** | **No** | **2.58** | **80.97 ± 0.21** |
| **Ours (HCLC)** | **SSD / VGG-16** | **No** | **2.58** | **78.13 ± 0.09** |

---

### Table 6: Ablation Study — CIFAR-10 (Different Acquisition Function Combinations)

| Model | Method | Accuracy (%) |
|---|---|---|
| **DenseNet-121** | LCHC | 94.49 |
| | **HLH (hybrid)** | **94.52** |
| | RHC | 94.47 |
| | RLC | 94.35 |
| **ResNet-56** | LCHC | 92.99 |
| | **HLH (hybrid)** | **93.41** |
| | RHC | 91.89 |
| | RLC | 92.30 |
| **VGG-16** | LCHC | 93.97 |
| | **HLH (hybrid)** | **94.35** |
| | RHC | 94.05 |
| | RLC | 94.17 |
| **ResNet-18** | LCHC | 95.24 |
| | HLH (hybrid) | 95.62 |
| | RHC | 95.24 |
| | **RLC** | **95.64** |
| **MobileNetV2** | LCHC | 94.67 |
| | HLH (hybrid) | 95.60 |
| | RHC | 94.69 |
| | **RLC** | **95.61** |

Acquisition function combinations tested: **LCHC** = Low Confidence + High Confidence, **HLH** = Hybrid Least Confidence + High Confidence, **RHC** = Random + High Confidence, **RLC** = Random + Low Confidence.

The hybrid acquisition function consistently achieves the best or near-best accuracy across architectures.

---

### DinoV2 on LabelMe1250K — Foundation Model Ablation

Pre-trained DinoV2 outperforms randomly initialized DinoV2 in most settings. However, with the **HCLC hybrid sampling** strategy, the from-scratch DinoV2 achieves results **comparable** to the pre-trained version — confirming that the proposed sampling methods are effective even with large foundation models and are readily combined with strong pre-trained backbones in practice.

---

### Class Imbalance Ablation (CIFAR-10, 10:1 imbalance ratio, VGG-16)

| Method | Accuracy (%) |
|---|---|
| LC | 92.95 |
| **HCLC** | **93.85** |
| HC | 92.50 |

Under class imbalance, HCLC is the preferred strategy. Pure uncertainty sampling (LC) alone can ignore minority classes; the hybrid approach handles skewed distributions better.

---

## Repository Structure

```
├── CIFAR10/
│   ├── densenet121_c10.py          # DenseNet-121 on CIFAR-10
│   ├── resnet18_c10.py             # ResNet-18 on CIFAR-10
│   ├── resnet56_c10.py             # ResNet-56 on CIFAR-10
│   ├── vgg16_c10.py                # VGG-16 on CIFAR-10
│   ├── mobilenet_c10.py            # MobileNetV2 on CIFAR-10
│   ├── swin_c10.py                 # Swin Transformer on CIFAR-10
│   └── smallvit_c10.py             # ViT-Small on CIFAR-10
├── CIFAR100/
│   ├── resnet18_c100_svhn.py       # ResNet-18 on CIFAR-100 / SVHN
│   ├── resnet56_c100_svhn.py       # ResNet-56 on CIFAR-100 / SVHN
│   ├── vgg16_c100_svhn.py          # VGG-16 on CIFAR-100 / SVHN
│   └── mobilenet_c100_new.py       # MobileNetV2 on CIFAR-100
├── SVHN/
│   ├── densenet121_svhn.py         # DenseNet-121 on SVHN
│   ├── resnet18_c100_svhn.py       # ResNet-18 on SVHN
│   ├── resnet56_svhn.py            # ResNet-56 on SVHN
│   └── vgg16_svhn.py               # VGG-16 on SVHN
├── ResNet18 TinyImageNet/
│   ├── resnet18_tin_new.py         # ResNet-18 on TinyImageNet
│   └── glister_TinyImageNet_ResNet18.py  # GLISTER baseline reproduction
├── PascalVOC SSD/
│   └── pascal_voc_ssd.py           # SSD object detection on VOC 2012
├── DinoV2 LabelMe1250K/
│   └── DinoV2_LabelMe1250K.py      # DinoV2 on LabelMe1250K
├── DenseNet121 CIFAR10/
│   └── glister_CIFAR10_DenseNet121.py   # GLISTER baseline reproduction
└── Class Imbalance VGG16/
    └── class_imbalance_vgg.py      # Class imbalance ablation
```

---

## Requirements

```bash
pip install torch torchvision numpy scipy matplotlib tqdm
pip install transformers          # DinoV2 experiments
pip install torchmetrics          # Pascal VOC mAP evaluation
pip install timm einops           # Swin Transformer
```

Tested with Python 3.8+, PyTorch 2.0+. All experiments run on NVIDIA GeForce RTX 4080 (16 GB).

---

## Reproducing Results

All scripts share the same structure. Each experiment runs with **3 seeds** (`42`, `789`, `101112`) and reports mean ± std.

The 4 experiment variants selectable via `--exp` are:

| `--exp` | Initial selection | Subsequent rounds | Maps to |
|---|---|---|---|
| 1 | Low Confidence | Low Confidence | LC |
| 2 | High Confidence | Low Confidence | HCLC |
| 3 | High Confidence | High Confidence | HC |
| 4 | Low Confidence | High Confidence | — |

---

### CIFAR-10

```bash
cd CIFAR10
python densenet121_c10.py --all        # runs all 4 variants across 3 seeds
python densenet121_c10.py --exp 1      # LC only
python resnet18_c10.py --all
python resnet56_c10.py --all
python vgg16_c10.py
python mobilenet_c10.py --all
python swin_c10.py --all
python smallvit_c10.py --all
```

### CIFAR-100

```bash
cd CIFAR100
python resnet18_c100_svhn.py --dataset cifar100
python resnet56_c100_svhn.py --dataset cifar100
python vgg16_c100_svhn.py --dataset cifar100
python mobilenet_c100_new.py --all
```

### SVHN

```bash
cd SVHN
python densenet121_svhn.py --all
python resnet18_c100_svhn.py --dataset svhn
python resnet56_svhn.py --all
python vgg16_svhn.py --all
```

### TinyImageNet

The script will auto-download TinyImageNet from Stanford if not already present (~237 MB).

```bash
cd "ResNet18 TinyImageNet"
python resnet18_tin_new.py --all --data-dir ./tiny-imagenet-200
python resnet18_tin_new.py --exp 1 --data-dir ./tiny-imagenet-200   # LC only
```

### Pascal VOC 2012 (Object Detection)

Download Pascal VOC 2012 and place it at `./data/VOCdevkit/VOC2012`, then:

```bash
cd "PascalVOC SSD"
python pascal_voc_ssd.py              # single seed
python pascal_voc_ssd.py --seeds 3   # 3 seeds for mean ± std
```

### DinoV2 on LabelMe1250K (Foundation Model Ablation)

Set `dataset_path` in the script to your LabelMe1250K directory, then:

```bash
cd "DinoV2 LabelMe1250K"
python DinoV2_LabelMe1250K.py        # runs non-pretrained (random weights)
```

To also run the pre-trained DinoV2, uncomment the `exp1_pretrained_dinov2` call in the `__main__` block.

### Reproducing GLISTER Baselines

```bash
# DenseNet-121 on CIFAR-10
cd "DenseNet121 CIFAR10"
python glister_CIFAR10_DenseNet121.py --num_runs 3 --epochs_per_round 50

# ResNet-18 on TinyImageNet
cd "ResNet18 TinyImageNet"
python glister_TinyImageNet_ResNet18.py --num_runs 3 --epochs_per_round 100
```

### Class Imbalance Ablation

```bash
cd "Class Imbalance VGG16"
python class_imbalance_vgg.py
```

---

## How the Pipeline Works

1. **Initialize** a model with **random weights** — no pre-training.
2. **Select initial samples** (~10K or 4% of the dataset) from the unlabeled pool using the chosen confidence criterion applied to the random model's softmax output.
3. **Train** the model on this labeled set for 100 epochs.
4. **Iteratively**: select the next batch (~5K or 5%) from the remaining unlabeled data using the trained model's confidence scores, add to the labeled pool, and retrain for 100 epochs.
5. **Repeat** until the annotation budget is exhausted.

No candidate model is ever trained. The core insight — motivated by the Lottery Ticket Hypothesis — is that randomly initialized networks already produce useful signal for guiding sample selection.

---

## Sampling Strategies

| Strategy | Acquisition Function | Description |
|---|---|---|
| **HC** | `φ_HC(x) = max_k P(y=k\|x)` | Selects highest-confidence (easy) samples — good for imbalanced data |
| **LC** | `φ_LC(x) = 1 − max_k P(y=k\|x)` | Selects most uncertain samples — best for balanced datasets |
| **HCLC** | HC first, then LC | Builds stable foundation first, then explores uncertain regions |

