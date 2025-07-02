# Active Learning Without Candidate Models

This repository contains the official implementation of the paper:

> **Are Candidate Models Really Needed for Active Learning?**  
> Harshini Mridula Mohan, Maanya Manjunatha, S.H. Shabbeer Basha, Nitin Cheekatla  

## Overview

This work challenges the traditional assumption in deep active learning that a candidate (pre-trained) model is necessary for effective sample selection. Instead, it demonstrates that models with **random initialization** can guide effective sample acquisition using confidence-based heuristics. The study evaluates three strategies:

- **High Confidence (HC)** sampling
- **Low Confidence (LC)** sampling
- **High Confidence + Low Confidence (HCLC)** sampling

Across multiple datasets and architectures, the hybrid HCLC strategy consistently delivers competitive or superior performance compared to traditional active learning methods.

## Architecture

The framework includes:

1. **Random Initialization**: Models start with random weights (no candidate model).
2. **Sample Selection**: Based on model prediction confidence:
   - `HC`: Selects top-𝑘 confident samples.
   - `LC`: Selects least confident samples.
   - `HCLC`: Combines both (HC initially, LC in later rounds).
3. **Iterative Training**: Samples are selected, labeled, and added to the training set in each iteration.

## Datasets

The experiments are conducted on:

- **CIFAR-10**
- **CIFAR-100**
- **SVHN**
- **PASCAL VOC 2012** (for object detection)

## Results Summary

| Dataset      | Best Accuracy (HCLC) | Baseline Comparison |
|--------------|----------------------|----------------------|
| CIFAR-10     | 94.79% (DenseNet-121) | ALFA-Mix: 91% |
| CIFAR-100    | 77.71% (ResNet-56)    | Bayesian NN: 60% |
| SVHN         | 96.77% (ResNet-56)    | ALFA Mix: 90% |
| VOC 2012     | 81.75% mAP (SSD+LCHC) | PPAL: ~74% |

## Ablation Studies

We also include ablation variants like:
- `rhc`: Random + High Confidence
- `rlc`: Random + Low Confidence
- `hlh`: Hybrid Low + High Confidence

These can be used to analyze trade-offs between exploration and exploitation in sampling.

## Citation

If you use this work, please cite:

```bibtex
@article{Mohan2025candidate,
  title={Are Candidate Models Really Needed for Active Learning?},
  author={Mohan, Harshini Mridula and Manjunatha, Maanya and Basha, S.H. Shabbeer and Cheekatla, Nitin},
  journal={Computer Vision and Image Understanding},
  year={2025}
}
