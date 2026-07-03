import matplotlib.pyplot as plt
import numpy as np

# ── FABRIC-AI results ────────────────────────────────────────
fabric_ai = {
    "Clear": 0.821,
    "Foggy": 0.743,
    "Rainy": 0.687,
}

# ── Baseline approaches ──────────────────────────────────────
carla_baseline = {
    "Clear": 0.612,
    "Foggy": 0.489,
    "Rainy": 0.421,
}

idd_baseline = {
    "Clear": 0.698,
    "Foggy": 0.523,
    "Rainy": 0.478,
}

no_aug_baseline = {
    "Clear": 0.743,
    "Foggy": 0.312,
    "Rainy": 0.287,
}

conditions = ["Clear", "Foggy", "Rainy"]
x = np.arange(len(conditions))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 7))

# Plot bars — note closing parenthesis on each
bars1 = ax.bar(
    x - 1.5 * width,
    [fabric_ai[c] for c in conditions],
    width,
    label='FABRIC-AI (Ours)',
    color='#4A6FA5',
    edgecolor='#2d2d2d',
    linewidth=0.8,
    zorder=3
)

bars2 = ax.bar(
    x - 0.5 * width,
    [idd_baseline[c] for c in conditions],
    width,
    label='YOLOv8 + IDD Dataset',
    color='#6A9E6F',
    edgecolor='#2d2d2d',
    linewidth=0.8,
    zorder=3
)

bars3 = ax.bar(
    x + 0.5 * width,
    [carla_baseline[c] for c in conditions],
    width,
    label='YOLOv8 + CARLA (Western)',
    color='#C9954C',
    edgecolor='#2d2d2d',
    linewidth=0.8,
    zorder=3
)

bars4 = ax.bar(
    x + 1.5 * width,
    [no_aug_baseline[c] for c in conditions],
    width,
    label='No Augmentation Baseline',
    color='#A85555',
    edgecolor='#2d2d2d',
    linewidth=0.8,
    zorder=3
)

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold'
        )

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# Styling
ax.set_xlabel('Environmental Condition',
              fontsize=13, fontweight='bold')
ax.set_ylabel('mAP@50',
              fontsize=13, fontweight='bold')
ax.set_title(
    'Fig. 8. Environmental Conditions Analysis:\n'
    'mAP@50 Scores Across Weather Conditions — '
    'FABRIC-AI vs. Baseline Approaches',
    fontsize=13, fontweight='bold', pad=20
)
ax.set_xticks(x)
ax.set_xticklabels(
    ['Clear', 'Foggy', 'Rainy'],
    fontsize=12
)
ax.set_ylim(0, 1.0)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda val, _: f'{val:.1f}')
)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.set_faceground = ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('white')

# Horizontal line at overall mAP
ax.axhline(
    y=0.758,
    color='#4A6FA5',
    linestyle='--',
    linewidth=1.5,
    alpha=0.7
)
ax.text(
    2.62, 0.765,
    'Overall mAP50 = 0.758',
    color='#4A6FA5',
    fontsize=9,
    fontstyle='italic'
)

plt.tight_layout()
plt.savefig(
    'fig8_weather_analysis.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.show()
print("[DONE] Saved: fig8_weather_analysis.png")