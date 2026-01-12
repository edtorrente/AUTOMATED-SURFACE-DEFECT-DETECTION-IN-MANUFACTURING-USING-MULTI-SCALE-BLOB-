#!/usr/bin/env python
# coding: utf-8

# Detection Overlay Visualization Demo (Optimized Version)
# =========================================================
# This version addresses performance anti-patterns identified in the original.
# See PERFORMANCE_ANALYSIS.md for detailed explanations.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import exposure, filters
from skimage.feature import blob_log
import os

# Constants
SQRT_2 = np.sqrt(2)  # Precompute constant used in visualization

def create_synthetic_defect_image(size=300, num_defects=15, seed=42):
    """
    Generate synthetic defect image with optimized defect placement.

    Optimizations applied:
    - Pre-generate all random values (vectorized)
    - Use localized bounding box for defect masks
    - In-place array operations where possible
    """
    np.random.seed(seed)

    # Create base image
    image_gray = np.ones((size, size), dtype=np.float64) * 0.3

    # Pre-generate all random values (Optimization #3)
    positions = np.random.randint(50, size - 50, (num_defects, 2))
    defect_sizes = np.random.randint(5, 15, num_defects)
    intensities = np.random.uniform(0.3, 0.5, num_defects)

    # Add synthetic defects using localized approach (Optimization #2)
    for i in range(num_defects):
        cx, cy = positions[i]
        r = defect_sizes[i]
        intensity = intensities[i]

        # Define bounding box (avoids full-image mask)
        x_min, x_max = max(0, cx - r), min(size, cx + r + 1)
        y_min, y_max = max(0, cy - r), min(size, cy + r + 1)

        # Create small local grid instead of full-image grid
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        local_mask = (xx - cx)**2 + (yy - cy)**2 <= r**2

        # Apply defect to local region
        image_gray[y_min:y_max, x_min:x_max][local_mask] += intensity

    # Add noise with in-place operations (Optimization #4)
    noise = np.random.normal(0, 0.05, image_gray.shape)
    np.add(image_gray, noise, out=image_gray)
    np.clip(image_gray, 0, 1, out=image_gray)

    return image_gray


def process_image(image_gray):
    """Apply enhancement and smoothing pipeline."""
    # Enhancement
    image_enhanced = exposure.equalize_adapthist(image_gray)

    # Smoothing
    image_smoothed = filters.gaussian(image_enhanced, sigma=2)

    return image_enhanced, image_smoothed


def detect_blobs(image_smoothed):
    """
    Detect blobs using Laplacian of Gaussian.

    Note: num_sigma=5 is often sufficient for demos.
    Use num_sigma=10 for finer scale resolution in production.
    """
    blobs = blob_log(
        image_smoothed,
        min_sigma=2,
        max_sigma=8,
        num_sigma=5,  # Reduced from 10 for demo (Optimization #7)
        threshold=0.1
    )
    return blobs


def create_visualization(image_gray, image_enhanced, image_smoothed, blobs, save_path=None):
    """
    Create 4-panel visualization with optimized rendering.

    Optimizations applied:
    - Precomputed SQRT_2 constant
    - Vectorized scatter plot for blob centers
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    ax = axes.ravel()

    # Panel 1: Original
    ax[0].imshow(image_gray, cmap='gray')
    ax[0].set_title('1. Original Grayscale Image', fontsize=14, pad=10)
    ax[0].set_axis_off()

    # Panel 2: Enhanced
    ax[1].imshow(image_enhanced, cmap='gray')
    ax[1].set_title('2. Enhanced Image', fontsize=14, pad=10)
    ax[1].set_axis_off()

    # Panel 3: Smoothed
    ax[2].imshow(image_smoothed, cmap='gray')
    ax[2].set_title('3. Enhanced and Smoothed Image', fontsize=14, pad=10)
    ax[2].set_axis_off()

    # Panel 4: Detections on SMOOTHED image
    ax[3].imshow(image_smoothed, cmap='gray')

    # Add detection circles (patches must be added individually)
    for blob in blobs:
        y, x, sigma = blob
        radius = sigma * SQRT_2  # Uses precomputed constant (Optimization #5)
        circle = mpatches.Circle(
            (x, y), radius,
            color='red', linewidth=2, fill=False, alpha=0.8
        )
        ax[3].add_patch(circle)

    # Add all centers at once with scatter (Optimization #6)
    if len(blobs) > 0:
        centers_x = blobs[:, 1]
        centers_y = blobs[:, 0]
        ax[3].scatter(
            centers_x, centers_y,
            marker='+', c='yellow', s=100, linewidths=2, alpha=0.9
        )

    ax[3].set_title(f'4. Detections on Smoothed Image: {len(blobs)} Found', fontsize=14, pad=10)
    ax[3].set_axis_off()

    plt.suptitle(
        'Detection Overlay Best Practice: Visualize on Processed Image',
        fontsize=16, fontweight='bold', y=0.98
    )
    plt.tight_layout()

    # Save the figure
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")

    return fig


def main():
    """Main execution function."""
    print("Processing image (optimized version)...")

    # Generate synthetic image
    image_gray = create_synthetic_defect_image(size=300, num_defects=15, seed=42)

    # Process image
    image_enhanced, image_smoothed = process_image(image_gray)

    # Detect blobs
    blobs = detect_blobs(image_smoothed)
    print(f"Found {len(blobs)} blobs")

    # Create visualization
    os.makedirs('images', exist_ok=True)
    create_visualization(
        image_gray, image_enhanced, image_smoothed, blobs,
        save_path='images/results_optimized.png'
    )

    plt.show()

    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("=" * 60)
    print("The detection overlay uses 'image_smoothed' as the background,")
    print("not 'image_gray'. This shows exactly why each detection was made!")
    print("=" * 60)
    print("\nOptimizations applied - see PERFORMANCE_ANALYSIS.md for details.")


if __name__ == '__main__':
    main()
