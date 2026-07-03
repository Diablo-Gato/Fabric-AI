import json
from pathlib import Path
import sys

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "ultralytics is required to run YOLO inference. Install it with `pip install ultralytics`."
    ) from exc


def run_yolo_stage(
    image_path: Path = Path("output/images/scene_0001.png"),
    model_path: Path = Path("yolov8n.pt"),
    output_labels_dir: Path = Path("datasets/yolo/train/labels"),
    confidence: float = 0.25,
    iou: float = 0.45,
    save_txt: bool = True,
    save_conf: bool = True,
):
    image_path = image_path.resolve()
    model_path = model_path.resolve()
    output_labels_dir = output_labels_dir.resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Target image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model weights not found: {model_path}")

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = image_path.parent

    print(f"[YOLO] Loading model from: {model_path}")
    model = YOLO(str(model_path))

    print(f"[YOLO] Running inference on: {image_path}")
    results = model.predict(
        source=str(image_path),
        conf=confidence,
        iou=iou,
        save=False,
        save_txt=False,
        verbose=False,
    )

    if len(results) == 0:
        raise RuntimeError("YOLO inference returned no results.")

    result = results[0]
    labels_file_name = image_path.with_suffix(".txt").name
    label_path = output_labels_dir / labels_file_name

    lines = []
    if hasattr(result, 'boxes') and result.boxes is not None:
        for box in result.boxes:
            if hasattr(box, 'cls'):
                cls = int(box.cls.cpu().item()) if hasattr(box.cls, 'cpu') else int(box.cls.item())
            else:
                cls = 0
            if hasattr(box, 'xyxy'):
                xyxy = box.xyxy.cpu().numpy().tolist()[0]
                x_min, y_min, x_max, y_max = xyxy
            elif hasattr(box, 'xywh'):
                x_min, y_min, w, h = box.xywh.cpu().numpy().tolist()[0]
                x_max = x_min + w
                y_max = y_min + h
            else:
                continue

            img_w, img_h = result.orig_shape[1], result.orig_shape[0]
            cx = ((x_min + x_max) / 2) / img_w
            cy = ((y_min + y_max) / 2) / img_h
            nw = (x_max - x_min) / img_w
            nh = (y_max - y_min) / img_h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if not lines:
        print(f"[YOLO] No detections found for {image_path}")

    label_path.write_text("\n".join(lines))
    print(f"[YOLO] Saved label file: {label_path}")

    output_json = output_labels_dir / f"{image_path.stem}_yolo_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "image": str(image_path),
            "labels": lines,
            "confidence": confidence,
            "iou": iou,
        }, f, indent=2)

    print(f"[YOLO] Saved result summary: {output_json}")
    return label_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on a single synthetic image and save YOLO labels.")
    parser.add_argument("--image", type=str, default="output/images/scene_0001.png")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--labels-dir", type=str, default="datasets/yolo/train/labels")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    run_yolo_stage(
        image_path=Path(args.image),
        model_path=Path(args.model),
        output_labels_dir=Path(args.labels_dir),
        confidence=args.confidence,
        iou=args.iou,
    )
