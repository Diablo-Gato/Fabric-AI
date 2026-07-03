import json
import cv2
from pathlib import Path

# ← Correct paths based on your folder structure
COCO_JSON = "output/latest_annotations.json"
IMAGES_DIR = Path("output/images")
OUTPUT_DIR = Path("datasets/verify")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

COLORS = {
    1: (255, 0, 0),      # car — blue
    2: (0, 255, 0),      # auto — green
    3: (0, 0, 255),      # truck — red
    4: (255, 255, 0),    # bus — cyan
    5: (0, 255, 255),    # cow — yellow
}

NAMES = {1: "car", 2: "auto", 3: "truck", 4: "bus", 5: "cow"}

with open(COCO_JSON) as f:
    coco = json.load(f)

img_map = {img["id"]: img for img in coco["images"]}
ann_by_img = {}
for ann in coco["annotations"]:
    ann_by_img.setdefault(ann["image_id"], []).append(ann)

for img_id, img_info in img_map.items():
    img_path = IMAGES_DIR / img_info["file_name"]
    if not img_path.exists():
        print(f"[SKIP] Image not found: {img_path}")
        continue

    img = cv2.imread(str(img_path))
    anns = ann_by_img.get(img_id, [])

    for ann in anns:
        x, y, w, h = [int(v) for v in ann["bbox"]]
        cat_id = ann["category_id"]
        color = COLORS.get(cat_id, (255, 255, 255))
        label = NAMES.get(cat_id, "unknown")

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_path = OUTPUT_DIR / img_info["file_name"]
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path} ({len(anns)} boxes)")

print(f"\n[DONE] Check datasets/verify/ folder")