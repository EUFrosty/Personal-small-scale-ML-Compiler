#!/usr/bin/env python3
"""Exports a torchvision ResNet-18 in ONNX: batch=1, static shapes"""

from pathlib import Path
import torch
import torchvision

OPSET = 17
OUT = Path(__file__).resolve().parents[1] / "models" / "resnet18.onnx"

def main():
    model = torchvision.models.resnet18(
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    model.eval()

    torch.manual_seed(0)
    dummy = torch.randn(1, 3, 224, 224)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(OUT),
            opset_version=OPSET,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=False,
            dynamo=False
        )

    print(f"Written: {OUT}")

if __name__ == "__main__":
    main()