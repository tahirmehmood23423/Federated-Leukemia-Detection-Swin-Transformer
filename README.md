# Federated Leukemia Detection using Swin Transformer

Deep-learning research on **automatic leukemia classification from microscopic blood-cell images** using a **Swin Transformer** (`swin_tiny_patch4_window7_224`), extended to **federated learning** so that hospitals could train a shared model without sharing patient images.

## Experiments

| Folder | What it does |
|---|---|
| `all_healthy/` | 4-class **ALL subtype** classification: Benign, Early Pre-B, Pre-B, Pro-B |
| `all_healthy_2/` | Improved ALL-subtype pipeline with stratified group splits (no patient leakage) and early stopping |
| `cll_all_healthy/` | 4-class **leukemia type** classification: ALL, CLL, CML, Healthy |
| `cross_validation/` | **K-Fold** and **Leave-One-Subject-Out (LOSO)** validation to test how well the model generalises |
| `federated learning/` | Simulated multi-client **federated training** with weight averaging (FedAvg) |

## Highlights

- Transfer learning from ImageNet-pretrained Swin-T through `timm`
- Data augmentation, stratified group splitting, early stopping
- Evaluation with confusion matrices, classification reports and **t-SNE** feature visualisation
- ~95–97% held-out accuracy in the centralised experiments (see notebook outputs)

## Tech stack

Python · PyTorch · timm · torchvision · scikit-learn · Matplotlib / Seaborn · Kaggle GPUs

## Author

**Engr. Tahir Mehmood**: MS CS @ NUST SEECS · [LinkedIn](https://www.linkedin.com/in/tahir-mehmood-596131209/)
