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

| Model | Method | Time Saved (hrs) | Accuracy (%) |
| --- | --- | --- | --- |
|  | **(LC) (10K)** | **0.16** | **91.37 ± 0.20** |
|  | **(LC)** | **0.16** | **92.87 ± 0.07** |
|  | **(HC)** | **0.16** | **92.72 ± 0.16** |
|  | **(HCLC)** | **0.16** | **93.08 ± 0.23** |
|  | **(LC)** | **0.26** | **91.60 ± 0.12** |
|  | **(HC)** | **0.26** | **91.30 ± 0.09** |
|  | **(HCLC)** | **0.26** | **91.60 ± 0.12** |
|  | **(LC) (20K)** | **0.5** | **84.26** |
|  | **(LC) (40K)** | **0.5** | **91.89** |
|  | **(LC)** | **0.5** | **94.21 ± 0.14** |
|  | **(HC)** | **0.5** | **93.94 ± 0.19** |
|  | **(HCLC)** | **0.5** | **94.20 ± 0.11** |
|  | **(LC) (5K)** | **0.25** | **81.27 ± 0.12** |
|  | **(LC) (10K)** | **0.25** | **90.12 ± 0.07** |
|  | **(LC) (40K)** | **0.25** | **92.69 ± 0.09** |
|  | **(LC)** | **0.25** | **93.53 ± 0.08** |
|  | **(HC)** | **0.25** | **93.28 ± 0.03** |
|  | **(HCLC)** | **0.25** | **93.48 ± 0.12** |
|  | **(LC)** | **0.06** | **86.23 ± 0.10** |
|  | **(HC)** | **0.06** | **83.88 ± 0.35** |
|  | **(HCLC)** | **0.06** | **85.80 ± 0.02** |
|  | **(LC)** | **0.52** | **83.92 ± 0.36** |
|  | **(HC)** | **0.52** | **82.70 ± 0.18** |
|  | **(HCLC)** | **0.52** | **83.81 ± 0.21** |
|  | **(LC) (10K)** | **0.52** | **82.53** |
|  | **(LC)** | **0.20** | **94.16 ± 0.03** |
|  | **(HC)** | **0.20** | **92.84 ± 0.23** |
|  | **(HCLC)** | **0.20** | **94.16 ± 0.03** |

---

### Table 2: CIFAR-100

| Model | Method | Time Saved (hrs) | Accuracy (%) |
| --- | --- | --- | --- |
|  | **(LC) (10K)** | **0.19** | **59.03** |
|  | **(LC)** | **0.19** | **71.65 ± 0.15** |
|  | **(HC)** | **0.19** | **70.98 ± 0.19** |
|  | **(HCLC)** | **0.19** | **71.33 ± 0.24** |
|  | **(LC)** | **0.31** | **66.22 ± 0.21** |
|  | **(HC)** | **0.31** | **66.30 ± 0.47** |
|  | **(HCLC)** | **0.31** | **66.39 ± 0.06** |
|  | **(LC)** | **0.57** | **66.46 ± 0.63** |
|  | **(HC)** | **0.57** | **64.87 ± 0.35** |
|  | **(HCLC)** | **0.57** | **66.11 ± 0.16** |
|  | **(LC) (10K)** | **0.10** | **59.01 ± 0.22** |
|  | **(LC)** | **0.10** | **73.24 ± 0.19** |
|  | **(HC)** | **0.10** | **71.97 ± 0.05** |
|  | **(HCLC)** | **0.10** | **73.42 ± 0.24** |
|  | **(LC)** | **0.25** | **73.79 ± 0.13** |
|  | **(HC)** | **0.25** | **72.82 ± 0.13** |
|  | **(HCLC)** | **0.25** | **73.79 ± 0.13** |

---

### Table 3: SVHN

| Model | Method | Time Saved (hrs) | Accuracy (%) |
| --- | --- | --- | --- |
|  | **(LC)** | **0.94** | **95.77 ± 0.01** |
|  | **(HC)** | **0.94** | **95.48 ± 0.15** |
|  | **(HCLC)** | **0.94** | **95.76 ± 0.08** |
|  | **(LC)** | **0.35** | **96.12 ± 0.11** |
|  | **(HC)** | **0.35** | **95.99 ± 0.05** |
|  | **(HCLC)** | **0.35** | **96.12 ± 0.11** |
|  | **(LC) (50K)** | **0.27** | **94.22** |
|  | **(LC)** | **0.27** | **95.51 ± 0.08** |
|  | **(HC)** | **0.27** | **95.45 ± 0.07** |
|  | **(HCLC)** | **0.27** | **95.61 ± 0.09** |
|  | **(LC) (15K)** | **0.29** | **91.80 ± 0.06** |
|  | **(LC) (50K)** | **0.29** | **93.23 ± 0.30** |
|  | **(LC)** | **0.29** | **95.84 ± 0.09** |
|  | **(HC)** | **0.29** | **95.65 ± 0.08** |
|  | **(HCLC)** | **0.29** | **95.83 ± 0.02** |

---

### Table 4: TinyImageNet (ResNet-18)

| Method | Time Saved (hrs) | Annotation Sim. Time (hrs) | Accuracy (%) |
| --- | --- | --- | --- |
| **(LC)** | **No** | **29.87** | **55.99 ± 0.12** |
| **(HC)** | **No** | **29.87** | **54.94 ± 0.14** |
| **(HCLC)** | **No** | **29.87** | **55.92 ± 0.11** |

> BADGE took ~45 hrs — 1.5× longer than our methods — for only a marginal accuracy gain of 0.74%.

---

### Table 5: PASCAL VOC 2012 — Object Detection (SSD / VGG-16)

| Method | SSD Variant | Time Saved (hrs) | mAP |
| --- | --- | --- | --- |
| **(LC)** | **SSD / VGG-16** | **2.58** | **81.53 ± 0.09** |
| **(HC)** | **SSD / VGG-16** | **2.58** | **80.97 ± 0.21** |
| **(HCLC)** | **SSD / VGG-16** | **2.58** | **78.13 ± 0.09** |

---

### Table 6: Ablation Study — CIFAR-10 (Different Acquisition Function Combinations)

| Model | Method | Accuracy (%) |
| --- | --- | --- |

Acquisition function combinations tested: **LCHC** = Low Confidence + High Confidence, **HLH** = Hybrid Least Confidence + High Confidence, **RHC** = Random + High Confidence, **RLC** = Random + Low Confidence.

The hybrid acquisition function consistently achieves the best or near-best accuracy across architectures.

---

### DinoV2 on LabelMe1250K — Foundation Model Ablation

Pre-trained DinoV2 outperforms randomly initialized DinoV2 in most settings. However, with the **HCLC hybrid sampling** strategy, the from-scratch DinoV2 achieves results **comparable** to the pre-trained version — confirming that the proposed sampling methods are effective even with large foundation models and are readily combined with strong pre-trained backbones in practice.

---

### Class Imbalance Ablation (CIFAR-10, 10:1 imbalance ratio, VGG-16)

| Method | Accuracy (%) |
| --- | --- |

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

| --- | --- | --- |

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
