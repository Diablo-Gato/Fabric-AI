import shutil
import random
from pathlib import Path

def split_dataset(images_dir, labels_dir, output_dir, val_split=0.2):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    # Create folders
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    images = sorted(images_dir.glob("*.png"))
    random.shuffle(images)

    val_count = int(len(images) * val_split)
    val_images = images[:val_count]
    train_images = images[val_count:]

    for split, img_list in [("train", train_images), ("val", val_images)]:
        for img_path in img_list:
            label_path = labels_dir / (img_path.stem + ".txt")

            shutil.copy(img_path, output_dir / "images" / split / img_path.name)

            if label_path.exists():
                shutil.copy(label_path, output_dir / "labels" / split / label_path.name)
            else:
                # Write empty label
                (output_dir / "labels" / split / (img_path.stem + ".txt")).write_text("")

    print(f"Train: {len(train_images)} | Val: {len(val_images)}")

if __name__ == "__main__":
    split_dataset(
        images_dir="output/images",
        labels_dir="datasets/yolo/labels",
        output_dir="datasets/yolo",
        val_split=0.2
    )