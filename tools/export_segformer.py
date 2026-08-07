#!/usr/bin/env python3
"""One-time export of SegFormer-B0 (ADE20K) to ONNX for path_nav.py.

Run on a machine with PyTorch (e.g. the Pi itself); the device that *runs*
inference only needs onnxruntime. Produces:
    segformer_b0_ade_onnx/model.onnx     (graph + weights)
    segformer_b0_ade_onnx/config.json    (id2label, for walkable-class mapping)

Usage (from repo root):
    python3 tools/export_segformer.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _bootstrap  # noqa: F401

import torch
from transformers import SegformerForSemanticSegmentation

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
OUT_DIR = "segformer_b0_ade_onnx"
INPUT_SIZE = 512
OPSET = 13


class LogitsOnly(torch.nn.Module):
    """Expose only the segmentation logits tensor for a clean ONNX graph."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading {MODEL_ID} ...")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).eval()

    wrapper = LogitsOnly(model)
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    onnx_path = os.path.join(OUT_DIR, "model.onnx")

    print(f"Exporting to {onnx_path} (opset {OPSET}) ...")
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "out_height", 3: "out_width"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )

    cfg_path = os.path.join(OUT_DIR, "config.json")
    model.config.to_json_file(cfg_path)
    print(f"Wrote {cfg_path} ({len(model.config.id2label)} classes)")
    print("Done.")


if __name__ == "__main__":
    main()
