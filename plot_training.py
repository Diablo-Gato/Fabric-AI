import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Correct path based on your folder structure
results_csv = Path("runs/detect/runs/train/fabric_ai_v1/results.csv")

if not results_csv.exists():
    print(f"[ERROR] File not found: {results_csv}")
    print("Check your runs folder structure")
    exit()

df = pd.read_csv(results_csv)
df.columns = df.columns.str.strip()

print("Columns found:", list(df.columns))
print(f"Total epochs: {len(df)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("FABRIC-AI YOLOv8 Training Results", fontsize=14, fontweight='bold')

# Plot 1 — Loss curves
axes[0].plot(df['epoch'], df['train/box_loss'],
             label='Train Box Loss', color='blue', linewidth=2)
axes[0].plot(df['epoch'], df['train/cls_loss'],
             label='Train Class Loss', color='orange', linewidth=2)
axes[0].plot(df['epoch'], df['val/box_loss'],
             label='Val Box Loss', color='blue', linestyle='--', linewidth=2)
axes[0].plot(df['epoch'], df['val/cls_loss'],
             label='Val Class Loss', color='orange', linestyle='--', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2 — mAP and metrics
axes[1].plot(df['epoch'], df['metrics/mAP50(B)'],
             label='mAP@0.5', color='green', linewidth=2)
axes[1].plot(df['epoch'], df['metrics/mAP50-95(B)'],
             label='mAP@0.5:0.95', color='red', linewidth=2)
axes[1].plot(df['epoch'], df['metrics/precision(B)'],
             label='Precision', color='purple', linestyle='--', linewidth=2)
axes[1].plot(df['epoch'], df['metrics/recall(B)'],
             label='Recall', color='brown', linestyle='--', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Score')
axes[1].set_title('mAP, Precision and Recall')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = "training_results_figure4.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\n[DONE] Saved: {output_path}")