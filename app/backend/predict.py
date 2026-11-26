import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# app/backend/predict.py -> repo root is two levels up
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = REPO_ROOT / "models" / "best_resnet18_augmented_lr1e4.pth"
MAPPING_PATH = REPO_ROOT / "models" / "class_mapping.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_class_names():
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    return mapping_data["class_names"]


def build_model(num_classes=9):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model


CLASS_NAMES = load_class_names()
MODEL = build_model(num_classes=len(CLASS_NAMES))


def predict_skin_lesion(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    top3_probs, top3_indices = torch.topk(probabilities, 3)

    prediction_main = CLASS_NAMES[top3_indices[0].item()]
    confidence_value = top3_probs[0].item()

    top3 = [
        {
            "class_name": CLASS_NAMES[idx.item()],
            "probability": f"{prob.item() * 100:.2f}%",
        }
        for prob, idx in zip(top3_probs, top3_indices)
    ]

    return {
        "prediction_main": prediction_main,
        "confidence": f"{confidence_value * 100:.2f}%",
        "top3": top3,
    }
