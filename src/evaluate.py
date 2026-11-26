import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from src.predict import (
    DEFAULT_MAPPING_PATH,
    DEFAULT_MODEL_PATH,
    DEVICE,
    INFERENCE_TRANSFORM,
    build_model,
    load_class_names,
)


def evaluate(test_dir: Path, model_path: Path, mapping_path: Path, batch_size: int = 32) -> dict:
    class_names = load_class_names(mapping_path)
    dataset = datasets.ImageFolder(str(test_dir), transform=INFERENCE_TRANSFORM)
    if dataset.classes != class_names:
        raise ValueError(
            f"Dataset classes {dataset.classes} do not match mapping {class_names}"
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = build_model(len(class_names), model_path, DEVICE)
    criterion = nn.CrossEntropyLoss()

    all_preds: list[int] = []
    all_true: list[int] = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_true.extend(labels.cpu().tolist())

    acc = accuracy_score(all_true, all_preds)
    avg_loss = total_loss / total_samples
    report_text = classification_report(
        all_true, all_preds, target_names=class_names, zero_division=0
    )
    report_dict = classification_report(
        all_true, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_true, all_preds)

    print(f"Test Samples : {total_samples}")
    print(f"Test Loss    : {avg_loss:.4f}")
    print(f"Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)\n")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)

    return {
        "test_samples": total_samples,
        "test_loss": round(avg_loss, 4),
        "test_accuracy": round(acc, 4),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MedMentor ResNet18 model on an ImageFolder test split."
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Path to the test split directory organized by class name.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    metrics = evaluate(args.test_dir, args.model, args.mapping, args.batch_size)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
