# Model Card — MedMentor ResNet18 Skin Lesion Classifier

## 1. Model Details

- **Model name:** `best_resnet18_augmented_lr1e4.pth` (deployed as `model.pth` in the MedMentor web application)
- **SHA-256 checksum:** `3e539dc435001646a348ab131719c4f3691183f413c234406a105dbb67fbe3c6`
- **Architecture:** Pretrained `torchvision.models.resnet18` (`ResNet18_Weights.DEFAULT`, ImageNet-1K) with the final linear layer replaced by `nn.Linear(512, 9)`
- **Task:** Multi-class image classification of dermatoscopic skin lesion images across 9 diagnostic classes
- **Author (ML component):** Bassma Alaoui (UEMF — EIDIA, 2025–2026, Prof. Abdrahaman Waga)
- **Framework:** PyTorch / Torchvision

## 2. Training Configuration

| Parameter | Value |
| :--- | :--- |
| Base model | Pretrained ResNet18 (`ResNet18_Weights.DEFAULT`) |
| Input resolution | `224 × 224` RGB |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Training augmentation | `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=10)` |
| Loss function | `CrossEntropyLoss` (unweighted) |
| Optimizer | `Adam(lr=1e-4, weight_decay=1e-4)` |
| Batch size | `16` |
| Epochs | `5` (best checkpoint saved at Epoch 4, Validation Accuracy `66.67%`, Validation Loss `1.0029`) |
| Hardware | Local CPU |

## 3. Experiment Progression

All four models were trained on the same clean stratified split (`1,649` train / `354` validation / `354` test) and independently re-evaluated on the held-out `354`-image test set:

| Experiment | Checkpoint | Best Val Acc | Test Acc | Test Loss | Macro F1 | Weighted F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Simple CNN Baseline (`128×128`) | `best_simple_cnn.pth` | 54.52% | 47.18% | 1.6217 | 0.3570 | 0.4523 |
| 2. ResNet18 Baseline (`lr=1e-3`) | `best_resnet18_skin_lesion.pth` | 61.30% | 54.52% | 1.3690 | 0.4749 | 0.5317 |
| 3. ResNet18 + Aug + WD (`lr=1e-3`) | `best_resnet18_augmented.pth` | 63.28% | 58.76% | 1.1454 | 0.4995 | 0.5711 |
| 4. **ResNet18 + Aug + WD + Lower LR (`lr=1e-4`)** | `best_resnet18_augmented_lr1e4.pth` | **66.67%** | **65.25%** | **1.0335** | **0.5700** | **0.6374** |

## 4. Per-Class Test Performance (Final Model, `n = 354`)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `actinic keratosis` | 0.40 | 0.20 | 0.27 | 20 |
| `basal cell carcinoma` | 0.69 | 0.71 | 0.70 | 59 |
| `dermatofibroma` | 0.85 | 0.69 | 0.76 | 16 |
| `melanoma` | 0.63 | 0.71 | 0.67 | 68 |
| `nevus` | 0.50 | 0.66 | 0.57 | 56 |
| `pigmented benign keratosis` | 0.76 | 0.76 | 0.76 | 72 |
| `seborrheic keratosis` | 0.00 | 0.00 | 0.00 | 12 |
| `squamous cell carcinoma` | 0.74 | 0.47 | 0.57 | 30 |
| `vascular lesion` | 0.74 | 0.95 | 0.83 | 21 |
| **Overall / Accuracy** | — | — | **0.65** | **354** |
| **Macro Avg** | **0.59** | **0.57** | **0.57** | **354** |
| **Weighted Avg** | **0.64** | **0.65** | **0.64** | **354** |

## 5. Error Analysis & Known Limitations

- **Severe minority-class failure on `seborrheic keratosis`:** With only 56 training images and 12 test images (and unweighted cross-entropy loss), all 12 test samples of `seborrheic keratosis` were misclassified as `melanoma` (0.00 precision / recall / F1).
- **Actinic keratosis vs. nevus / squamous cell carcinoma:** `actinic keratosis` (20 test images) achieved only 0.20 recall, frequently confused with `nevus` (8 samples) and `basal cell carcinoma` (3 samples).
- **Short CPU training budget:** Models were trained for 5 epochs on CPU; longer training with class-weighted loss or focal loss, learning rate scheduling, and stronger augmentations would be needed to improve minority-class recall.
- **Domain shift:** Trained strictly on dermatoscopic images from ISIC; smartphone photographs taken under uncontrolled lighting are out of distribution.
- **Not a medical device:** Built exclusively as a university deep learning coursework prototype. It has undergone zero clinical validation and must never be used for diagnosis or medical decision-making.
