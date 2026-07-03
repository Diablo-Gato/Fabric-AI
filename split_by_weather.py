import json
import shutil
from pathlib import Path

OUTPUT_DIR = Path("output")
IMAGES_DIR = OUTPUT_DIR / "images"
ANN_PATH = OUTPUT_DIR / "latest_annotations.json"
CONFIG_PATH = Path("configs/scene_configs_100.json")

import json
with open(CONFIG_PATH) as f:
    configs = json.load(f)

with open(ANN_PATH) as f:
    coco = json.load(f)

# Create weather folders
for weather in ["clear", "rainy", "foggy"]:
    (IMAGES_DIR / weather).mkdir(exist_ok=True)

# Copy images into weather subfolders
for i, config in enumerate(configs):
    img_id = i + 1
    img_name = f"scene_{img_id:04d}.png"
    src = IMAGES_DIR / img_name
    weather = config.get("weather", "clear")
    dst = IMAGES_DIR / weather / img_name
    if src.exists():
        shutil.copy(src, dst)
        print(f"Copied {img_name} → {weather}/")

print("\n[DONE] Images separated by weather")

# Count per weather
for weather in ["clear", "rainy", "foggy"]:
    count = len(list((IMAGES_DIR / weather).glob("*.png")))
    print(f"{weather}: {count} images")