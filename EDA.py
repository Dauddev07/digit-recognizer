# ============================================================
# eda.py
# Exploratory Data Analysis on the MNIST Dataset
# Run this file to generate all EDA figures
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves files instead of showing windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.datasets import mnist

# Create folder to save EDA figures
os.makedirs("eda_output", exist_ok=True)

print("=" * 60)
print("MNIST DATASET — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── STEP 1: Load Data ────────────────────────────────────────
print("\n[1] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ── STEP 2: Basic Info ───────────────────────────────────────
print("\n[2] DATASET DIMENSIONS AND SHAPES")
print(f"    Training images shape : {x_train.shape}")
print(f"    Training labels shape : {y_train.shape}")
print(f"    Test images shape     : {x_test.shape}")
print(f"    Test labels shape     : {y_test.shape}")
print(f"    Total images          : {len(x_train) + len(x_test)}")
print(f"    Image dimensions      : {x_train.shape[1]} x {x_train.shape[2]} pixels")
print(f"    Color channels        : Grayscale (1 channel)")
print(f"    Number of classes     : {len(np.unique(y_train))}")
print(f"    Class labels          : {np.unique(y_train)}")

# ── STEP 3: Pixel Value Statistics ──────────────────────────
print("\n[3] PIXEL VALUE STATISTICS (Training Set)")
print(f"    Data type             : {x_train.dtype}")
print(f"    Minimum pixel value   : {x_train.min()}")
print(f"    Maximum pixel value   : {x_train.max()}")
print(f"    Mean pixel value      : {x_train.mean():.4f}")
print(f"    Std deviation         : {x_train.std():.4f}")
print(f"    After normalization   : values will be 0.0 to 1.0")

# ── STEP 4: Class Distribution ───────────────────────────────
print("\n[4] CLASS DISTRIBUTION")
print(f"    {'Digit':<10} {'Train Count':<15} {'Train %':<12} {'Test Count':<15} {'Test %'}")
print(f"    {'-'*60}")
for digit in range(10):
    train_count = np.sum(y_train == digit)
    test_count  = np.sum(y_test  == digit)
    train_pct   = train_count / len(y_train) * 100
    test_pct    = test_count  / len(y_test)  * 100
    print(f"    {digit:<10} {train_count:<15} {train_pct:<12.2f} {test_count:<15} {test_pct:.2f}")

# ── FIGURE 1: Sample Images Grid ─────────────────────────────
print("\n[5] Generating Figure 1: Sample images for each digit...")
fig, axes = plt.subplots(10, 10, figsize=(14, 14))
fig.suptitle("MNIST Dataset — Sample Images (10 per digit class)",
             fontsize=16, fontweight='bold', y=0.98)
fig.patch.set_facecolor('#1a1a2e')

for digit in range(10):
    indices = np.where(y_train == digit)[0][:10]
    for j, idx in enumerate(indices):
        ax = axes[digit][j]
        ax.imshow(x_train[idx], cmap='gray')
        ax.axis('off')
        if j == 0:
            ax.set_ylabel(f'Digit {digit}',
                         color='white', fontsize=11,
                         fontweight='bold', rotation=0,
                         labelpad=40, va='center')

plt.tight_layout(rect=[0.05, 0, 1, 0.97])
plt.savefig("eda_output/fig1_sample_images.png", dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("    Saved: eda_output/fig1_sample_images.png")

# ── FIGURE 2: Class Distribution Bar Chart ───────────────────
print("\n[6] Generating Figure 2: Class distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Class Distribution — Training vs Test Set",
             fontsize=14, fontweight='bold')

digits = list(range(10))
train_counts = [np.sum(y_train == d) for d in digits]
test_counts  = [np.sum(y_test  == d) for d in digits]
colors = plt.cm.tab10(np.linspace(0, 1, 10))

# Training bar chart
bars = axes[0].bar(digits, train_counts, color=colors, edgecolor='white', linewidth=0.5)
axes[0].set_title("Training Set (60,000 images)", fontweight='bold', fontsize=12)
axes[0].set_xlabel("Digit Class")
axes[0].set_ylabel("Number of Images")
axes[0].set_xticks(digits)
axes[0].set_ylim(0, max(train_counts) * 1.15)
for bar, count in zip(bars, train_counts):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                 str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

# Test bar chart
bars2 = axes[1].bar(digits, test_counts, color=colors, edgecolor='white', linewidth=0.5)
axes[1].set_title("Test Set (10,000 images)", fontweight='bold', fontsize=12)
axes[1].set_xlabel("Digit Class")
axes[1].set_ylabel("Number of Images")
axes[1].set_xticks(digits)
axes[1].set_ylim(0, max(test_counts) * 1.15)
for bar, count in zip(bars2, test_counts):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                 str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("eda_output/fig2_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: eda_output/fig2_class_distribution.png")

# ── FIGURE 3: Pixel Intensity Distribution ───────────────────
print("\n[7] Generating Figure 3: Pixel intensity distribution...")
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle("Pixel Intensity Distribution per Digit Class",
             fontsize=14, fontweight='bold')

for digit in range(10):
    row, col = digit // 5, digit % 5
    digit_images = x_train[y_train == digit].flatten()
    axes[row][col].hist(digit_images, bins=50, color=colors[digit],
                        alpha=0.8, edgecolor='white', linewidth=0.3)
    axes[row][col].set_title(f"Digit {digit}", fontweight='bold')
    axes[row][col].set_xlabel("Pixel Value (0-255)")
    axes[row][col].set_ylabel("Frequency")
    axes[row][col].axvline(digit_images.mean(), color='red',
                           linestyle='--', linewidth=1.5,
                           label=f'Mean: {digit_images.mean():.1f}')
    axes[row][col].legend(fontsize=8)

plt.tight_layout()
plt.savefig("eda_output/fig3_pixel_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: eda_output/fig3_pixel_distribution.png")

# ── FIGURE 4: Average Image per Digit ────────────────────────
print("\n[8] Generating Figure 4: Average image per digit...")
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Average Image per Digit Class\n(Mean of all training images for each digit)",
             fontsize=13, fontweight='bold')
fig.patch.set_facecolor('#1a1a2e')

for digit in range(10):
    row, col = digit // 5, digit % 5
    digit_images = x_train[y_train == digit]
    avg_image = digit_images.mean(axis=0)
    im = axes[row][col].imshow(avg_image, cmap='hot')
    axes[row][col].set_title(f'Digit {digit}\n({len(digit_images):,} samples)',
                             color='white', fontweight='bold', fontsize=10)
    axes[row][col].axis('off')
    plt.colorbar(im, ax=axes[row][col], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("eda_output/fig4_average_images.png", dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("    Saved: eda_output/fig4_average_images.png")

# ── FIGURE 5: Single Image Pixel Grid ────────────────────────
print("\n[9] Generating Figure 5: Pixel grid of one image...")
sample_idx = np.where(y_train == 7)[0][0]
sample_image = x_train[sample_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Pixel-Level View of a Single MNIST Image (Digit: {y_train[sample_idx]})",
             fontsize=13, fontweight='bold')

# Left: image display
axes[0].imshow(sample_image, cmap='gray')
axes[0].set_title("Visual Representation", fontweight='bold')
axes[0].axis('off')

# Right: pixel value heatmap with numbers
im = axes[1].imshow(sample_image, cmap='Blues')
axes[1].set_title("Pixel Values (0=black, 255=white)", fontweight='bold')
for i in range(28):
    for j in range(28):
        val = sample_image[i, j]
        if val > 30:
            axes[1].text(j, i, str(val), ha='center', va='center',
                        fontsize=3.5, color='white' if val > 127 else 'black',
                        fontweight='bold')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig("eda_output/fig5_pixel_grid.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: eda_output/fig5_pixel_grid.png")

# ── FIGURE 6: Before vs After Normalization ──────────────────
print("\n[10] Generating Figure 6: Before vs after normalization...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Effect of Normalization on Pixel Values", fontsize=13, fontweight='bold')

sample = x_train[0]
normalized = sample.astype('float32') / 255.0

axes[0].imshow(sample, cmap='gray')
axes[0].set_title(f"Original Image\nPixel range: 0 to 255", fontweight='bold')
axes[0].axis('off')

axes[1].hist(sample.flatten(), bins=30, color='steelblue', edgecolor='white')
axes[1].set_title("Pixel Distribution\nBefore Normalization", fontweight='bold')
axes[1].set_xlabel("Pixel Value")
axes[1].set_ylabel("Frequency")

axes[2].hist(normalized.flatten(), bins=30, color='coral', edgecolor='white')
axes[2].set_title("Pixel Distribution\nAfter Normalization (÷255)", fontweight='bold')
axes[2].set_xlabel("Pixel Value (0.0 to 1.0)")
axes[2].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("eda_output/fig6_normalization.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: eda_output/fig6_normalization.png")

# ── SUMMARY ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA COMPLETE — Summary")
print("=" * 60)
print(f"  Dataset          : MNIST")
print(f"  Total images     : 70,000")
print(f"  Training images  : 60,000")
print(f"  Test images      : 10,000")
print(f"  Image size       : 28 x 28 pixels")
print(f"  Color mode       : Grayscale (1 channel)")
print(f"  Classes          : 10 (digits 0-9)")
print(f"  Pixel range      : 0 to 255")
print(f"  Class balance    : Approximately equal (~6,000 per digit in training)")
print(f"  Figures saved to : eda_output/ folder")
print("=" * 60)
print("\nAll EDA figures saved successfully!")
print("Check the eda_output/ folder in your project directory.")