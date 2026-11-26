import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "best_resnet18_augmented_lr1e4.pth"
DEFAULT_MAPPING_PATH = REPO_ROOT / "models" / "class_mapping.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_class_names(mapping_path: Path = DEFAULT_MAPPING_PATH) -> list[str]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    return mapping_data["class_names"]


def build_model(
    num_classes: int = 9,
    model_path: Path = DEFAULT_MODEL_PATH,
    device: torch.device = DEVICE,
) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


_CACHED_MODEL = None
_CACHED_CLASS_NAMES = None


def _get_default_artifacts():
    global _CACHED_MODEL, _CACHED_CLASS_NAMES
    if _CACHED_CLASS_NAMES is None:
        _CACHED_CLASS_NAMES = load_class_names(DEFAULT_MAPPING_PATH)
    if _CACHED_MODEL is None:
        _CACHED_MODEL = build_model(
            num_classes=len(_CACHED_CLASS_NAMES),
            model_path=DEFAULT_MODEL_PATH,
            device=DEVICE,
        )
    return _CACHED_MODEL, _CACHED_CLASS_NAMES


def predict_skin_lesion(
    image_path: str | Path,
    model: nn.Module | None = None,
    class_names: list[str] | None = None,
    device: torch.device = DEVICE,
) -> dict:
    if model is None or class_names is None:
        model, class_names = _get_default_artifacts()

    image = Image.open(image_path).convert("RGB")
    image_tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    top3_probs, top3_indices = torch.topk(probabilities, 3)

    prediction_main = class_names[top3_indices[0].item()]
    confidence_value = top3_probs[0].item()

    top3 = [
        {
            "class_name": class_names[idx.item()],
            "probability": f"{prob.item() * 100:.2f}%",
        }
        for prob, idx in zip(top3_probs, top3_indices)
    ]

    return {
        "prediction_main": prediction_main,
        "confidence": f"{confidence_value * 100:.2f}%",
        "top3": top3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 9-class ResNet18 skin lesion inference on an image."
    )
    parser.add_argument("image", type=Path, help="Path to an input skin lesion image.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the .pth weights file.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
        help="Path to class_mapping.json.",
    )
    args = parser.parse_args()

    class_names = load_class_names(args.mapping)
    model = build_model(len(class_names), args.model, DEVICE)
    result = predict_skin_lesion(args.image, model=model, class_names=class_names)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
