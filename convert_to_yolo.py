import json
import os
from pathlib import Path

def coco_to_yolo(coco_json_path, output_dir, images_dir):
    """Convert COCO annotations to YOLO format."""
    
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Build image lookup
    image_map = {img["id"]: img for img in coco["images"]}

    # Build annotation lookup by image_id
    ann_by_image = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        if iid not in ann_by_image:
            ann_by_image[iid] = []
        ann_by_image[iid].append(ann)

    converted = 0
    skipped = 0

    for img_id, img_info in image_map.items():
        img_w = img_info["width"]
        img_h = img_info["height"]
        img_filename = img_info["file_name"]

        # YOLO label file — same name as image but .txt
        label_filename = Path(img_filename).stem + ".txt"
        label_path = labels_dir / label_filename

        anns = ann_by_image.get(img_id, [])

        if not anns:
            # Write empty file — YOLO needs label file even for empty images
            label_path.write_text("")
            skipped += 1
            continue

        lines = []
        for ann in anns:
            cat_id = ann["category_id"] - 1  # YOLO is 0-indexed
            x, y, w, h = ann["bbox"]

            # Convert COCO (x_min, y_min, w, h) to YOLO (cx, cy, w, h) normalized
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Clamp to valid range
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path.write_text("\n".join(lines))
        converted += 1

    print(f"[DONE] Converted {converted} images, {skipped} empty")
    print(f"Labels saved to: {labels_dir}")

    # Write YOLO dataset.yaml
    yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val: images/val

nc: 5
names:
  0: car
  1: auto
  2: truck
  3: bus
  4: cow
"""
    (output_dir / "dataset.yaml").write_text(yaml_content)
    print(f"dataset.yaml written to: {output_dir}")


if __name__ == "__main__":
    coco_to_yolo(
        coco_json_path="output/latest_annotations.json",
        output_dir="datasets/yolo",
        images_dir="output/images"
    )