from ultralytics import YOLO
from pathlib import Path
import json

MODEL_PATH = "runs/detect/runs/train/fabric_ai_v1/weights/best.pt"
DATASET_YAML = str(Path("datasets/yolo/dataset.yaml").resolve())
IMAGES_DIR = Path("output/images")

model = YOLO(MODEL_PATH)

results_by_weather = {}

for weather in ["clear", "rainy", "foggy"]:
    weather_dir = IMAGES_DIR / weather
    images = list(weather_dir.glob("*.png"))

    if not images:
        print(f"[WARN] No images found for {weather}")
        results_by_weather[weather] = {
            "mAP50": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "count": 0
        }
        continue

    print(f"\nEvaluating {weather} — {len(images)} images...")

    # Run inference on all images of this weather type
    results = model.predict(
        source=str(weather_dir),
        conf=0.25,
        device='cpu',
        verbose=False
    )

    # Collect confidence scores as proxy for mAP
    all_confs = []
    for r in results:
        for box in r.boxes:
            all_confs.append(float(box.conf))

    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0
    det_per_image = len(all_confs) / len(images) if images else 0

    results_by_weather[weather] = {
        "count": len(images),
        "avg_confidence": round(avg_conf, 3),
        "detections_per_image": round(det_per_image, 2)
    }

    print(f"  Images: {len(images)}")
    print(f"  Avg Confidence: {avg_conf:.3f}")
    print(f"  Detections/Image: {det_per_image:.2f}")

print("\n[DONE] Weather evaluation complete")
print(json.dumps(results_by_weather, indent=2))