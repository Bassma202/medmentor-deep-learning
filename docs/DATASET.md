# Dataset Documentation — ISIC 9-Class Skin Lesion Dataset

## 1. Source & Permitted Access

- **Dataset:** Skin Cancer ISIC (9-class dermatoscopic image dataset hosted on Kaggle, derived from the International Skin Imaging Collaboration — ISIC Archive).
- **Access & Redistribution:** The full 2,357-image dataset is **not** committed to this repository to respect repository size limits and external dataset distribution terms. To reproduce full evaluation or training, download the ISIC 9-class skin cancer dataset from Kaggle / ISIC Archive and organize it into `dataset_split/{train,val,test}/<class_name>/`.
- **Included Sample Images:** Four representative dermatoscopic test images are included in `samples/` strictly to allow immediate, reproducible CLI and API inference verification.

## 2. Raw Dataset Issue Identified & Fixed During Preprocessing

Inspection of the original downloaded folder (`dataset/Train`, `dataset/Validation`, `dataset/Test`) revealed two critical issues in `notebooks/medmentor_model_work.ipynb` (Cells 1–5):

1. `Validation` (`2,239` images) was an **exact duplicate** of `Train` (`2,239` images), which would have caused severe data leakage and inflated validation scores.
2. The original `Test` split contained only `118` images (with just `3` images each for `seborrheic keratosis` and `vascular lesion`).

To eliminate leakage and obtain reliable stratified evaluation:
- `Train` (`2,239`) and `Test` (`118`) were merged into a single clean pool of **2,357** images.
- A stratified **70% / 15% / 15%** split (`random_state=42`) was generated with `sklearn.model_selection.train_test_split`.

## 3. Class Distribution (Stratified 70 / 15 / 15 Split)

| Index | Class Name | Total | Train (70%) | Val (15%) | Test (15%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | `actinic keratosis` | 130 | 91 | 19 | 20 |
| 1 | `basal cell carcinoma` | 392 | 274 | 59 | 59 |
| 2 | `dermatofibroma` | 111 | 78 | 17 | 16 |
| 3 | `melanoma` | 454 | 318 | 68 | 68 |
| 4 | `nevus` | 373 | 261 | 56 | 56 |
| 5 | `pigmented benign keratosis` | 478 | 334 | 72 | 72 |
| 6 | `seborrheic keratosis` | 80 | 56 | 12 | 12 |
| 7 | `squamous cell carcinoma` | 197 | 138 | 29 | 30 |
| 8 | `vascular lesion` | 142 | 99 | 22 | 21 |
| — | **Total** | **2,357** | **1,649** | **354** | **354** |

## 4. Preprocessing Pipeline

- **Training (Final Tuned Model):**
  - `Resize((224, 224))`
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomRotation(degrees=10)`
  - `ToTensor()`
  - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- **Validation / Test / Inference:**
  - `Resize((224, 224))`
  - `ToTensor()`
  - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

## 5. Note on Project Report Discrepancy

The joint academic report (`docs/Rapport_MedMentor_AI.pdf`, drafted earlier in the course) references HAM10000 and 7 classes. During model development, the ML pipeline was built and trained on the **9-class ISIC skin lesion dataset** documented above, and the final integrated weights (`models/best_resnet18_augmented_lr1e4.pth` / `AI_Menthor/model/model.pth`) plus `models/class_mapping.json` output 9 classes.
