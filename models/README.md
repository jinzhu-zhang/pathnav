# Model weights

Large ONNX / PyTorch weights are **not** committed (see `.gitignore`). Export them once on any machine with PyTorch, then copy onto the Pi.

## SegFormer-B0 (walkable-path segmentation)

```bash
# needs: pip install -r requirements-export.txt
python3 tools/export_segformer.py
# -> segformer_b0_ade_onnx/model.onnx + config.json
```

Source checkpoint: [`nvidia/segformer-b0-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) (ADE20K).

## YOLOv8n (obstacles)

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12
# -> yolov8n.onnx
```

Place `yolov8n.onnx` (and optionally `segformer_b0_ade_onnx/`) in the **repo root**. The runtime app only needs `onnxruntime` — not PyTorch.
