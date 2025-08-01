#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Detection Overlay Visualization Demo
# ====================================
# This version is specifically for Jupyter notebooks

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import exposure, filters
from skimage.feature import blob_log
import os

# Generate synthetic defect image
np.random.seed(42)
size = 300
num_defects = 15

# Create base image
image_gray = np.ones((size, size)) * 0.3

# Add synthetic defects
for _ in range(num_defects):
    x, y = np.random.randint(50, size-50, 2)
    defect_size = np.random.randint(5, 15)
    y_grid, x_grid = np.ogrid[:size, :size]
    mask = (x_grid - x)**2 + (y_grid - y)**2 <= defect_size**2
    image_gray[mask] += np.random.uniform(0.3, 0.5)

# Add noise
image_gray += np.random.normal(0, 0.05, image_gray.shape)
image_gray = np.clip(image_gray, 0, 1)

print("Processing image...")

# Enhancement
image_enhanced = exposure.equalize_adapthist(image_gray)

# Smoothing
image_smoothed = filters.gaussian(image_enhanced, sigma=2)

# Blob detection
blobs = blob_log(image_smoothed, min_sigma=2, max_sigma=8, num_sigma=10, threshold=0.1)
print(f"Found {len(blobs)} blobs")

# Create visualization
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

# Panel 4: Detections on SMOOTHED image (KEY INSIGHT!)
ax[3].imshow(image_smoothed, cmap='gray')

for blob in blobs:
    y, x, sigma = blob
    radius = sigma * np.sqrt(2)
    circle = mpatches.Circle((x, y), radius, color='red', linewidth=2, fill=False, alpha=0.8)
    ax[3].add_patch(circle)
    ax[3].plot(x, y, '+', color='yellow', markersize=10, markeredgewidth=2, alpha=0.9)

ax[3].set_title(f'4. Detections on Smoothed Image: {len(blobs)} Found', fontsize=14, pad=10)
ax[3].set_axis_off()

plt.suptitle('Detection Overlay Best Practice: Visualize on Processed Image', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

# Save the figure
os.makedirs('images', exist_ok=True)
plt.savefig('images/results.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Figure saved to: images/results.png")

plt.show()

print("\n" + "="*60)
print("KEY INSIGHT:")
print("="*60)
print("The detection overlay uses 'image_smoothed' as the background,")
print("not 'image_gray'. This shows exactly why each detection was made!")
print("="*60)


# In[ ]:




