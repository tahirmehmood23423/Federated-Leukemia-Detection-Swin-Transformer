# Federated Leukemia Detection using Swin Transformer

Deep learning research on **automatic leukemia classification from microscopic blood cell images** using a **Swin Transformer** (swin-tiny, patch 4, window 7, 224 px), extended to **federated learning** so that hospitals could train a shared model without sharing patient images.

## Experiments

- **all healthy:** 4-class ALL subtype classification (Benign, Early Pre-B, Pre-B, Pro-B)
- **all healthy 2:** improved ALL subtype pipeline with stratified group splits (no patient leakage) and early stopping
- **cll all healthy:** 4-class leukemia type classification (ALL, CLL, CML, Healthy)
- **cross validation:** K-Fold and Leave-One-Subject-Out (LOSO) validation to test how well the model generalises
- **federated learning:** simulated multi-client federated training with weight averaging (FedAvg)

## Highlights

- Transfer learning from an ImageNet-pretrained Swin-T model through timm
- Data augmentation, stratified group splitting and early stopping
- Evaluation with confusion matrices, classification reports and t-SNE feature visualisation
- About 95 to 97 percent held-out accuracy in the centralised experiments (see notebook outputs)

## Tech Stack

- Python, PyTorch, timm, torchvision
- scikit-learn, Matplotlib, Seaborn
- Kaggle GPUs

## Author

- Engr. Tahir Mehmood, MS Computer Science, NUST SEECS
- LinkedIn: [linkedin.com/in/tahir-mehmood-596131209](https://www.linkedin.com/in/tahir-mehmood-596131209/)
