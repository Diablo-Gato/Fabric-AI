"""
FABRIC-AI YOLOv8 Training Script

CHANGES FROM ORIGINAL:
- Auto-detects NVIDIA GPU and uses it (CUDA)
- Falls back to CPU if no GPU found
- Batch size auto-scales: 16 on GPU, 4 on CPU
- Added amp=True for faster GPU training (mixed precision)
"""

from ultralytics import YOLO
from pathlib import Path
import torch


if __name__ == '__main__':

    # ── GPU / CPU detection ───────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "0"
        batch  = 16
        amp    = True
        print(f"[GPU] Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        batch  = 4
        amp    = False
        print("[CPU] No CUDA GPU found — training on CPU")

    # ── Dataset ───────────────────────────────────────────────────────────────
    DATASET_YAML = str(Path("datasets/yolo/dataset.yaml").resolve())
    print(f"Training with dataset: {DATASET_YAML}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = YOLO("yolov8n.pt")

    # ── Train ─────────────────────────────────────────────────────────────────
    results = model.train(
        data=DATASET_YAML,
        epochs=100,
        imgsz=640,
        batch=batch,
        name="fabric_ai_v1",
        project="runs/train",
        patience=20,
        save=True,
        plots=True,
        device=device,
        amp=amp,
        workers=4,        # reduced from 8 — avoids Windows worker spawn issues
    )

    print("\n[DONE] Training complete!")
    print("Best weights: runs/train/fabric_ai_v1/weights/best.pt")
    print("Results graph: runs/train/fabric_ai_v1/results.png")
