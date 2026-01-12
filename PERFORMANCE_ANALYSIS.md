# Performance Analysis Report

## Summary

This document analyzes performance anti-patterns found in the codebase. While this is a small educational demo, these patterns are important to recognize as they can cause significant performance issues in production systems processing larger images or real-time video streams.

---

## Performance Issues Identified

### 1. Redundant Grid Creation in Loop

**Location:** `detection_demo.py:27-32`

**Issue:** The coordinate grid `np.ogrid[:size, :size]` is recreated for every defect iteration.

```python
# CURRENT (Inefficient)
for _ in range(num_defects):
    x, y = np.random.randint(50, size-50, 2)
    defect_size = np.random.randint(5, 15)
    y_grid, x_grid = np.ogrid[:size, :size]  # Recreated 15 times!
    mask = (x_grid - x)**2 + (y_grid - y)**2 <= defect_size**2
    image_gray[mask] += np.random.uniform(0.3, 0.5)
```

**Impact:** Creates 15 identical grid objects (for `num_defects=15`), wasting memory allocation time.

**Fix:** Move grid creation outside the loop:
```python
# OPTIMIZED
y_grid, x_grid = np.ogrid[:size, :size]  # Create once
for _ in range(num_defects):
    x, y = np.random.randint(50, size-50, 2)
    defect_size = np.random.randint(5, 15)
    mask = (x_grid - x)**2 + (y_grid - y)**2 <= defect_size**2
    image_gray[mask] += np.random.uniform(0.3, 0.5)
```

**Severity:** Medium - O(n) unnecessary allocations where n = num_defects

---

### 2. Full-Image Mask for Localized Defects

**Location:** `detection_demo.py:30-32`

**Issue:** Creates a full 300x300 boolean mask to update small circular regions (5-15 pixel radius).

```python
# CURRENT (Inefficient)
mask = (x_grid - x)**2 + (y_grid - y)**2 <= defect_size**2  # 90,000 element comparison
image_gray[mask] += np.random.uniform(0.3, 0.5)
```

**Impact:**
- Computes distance for 90,000 pixels when only ~700 pixels (for max radius 15) are in the defect
- Creates 90,000-element boolean array per defect
- Boolean indexing with sparse mask is slower than direct slicing

**Fix:** Use localized bounding box approach:
```python
# OPTIMIZED - Local patch approach
for _ in range(num_defects):
    cx, cy = np.random.randint(50, size-50, 2)
    r = np.random.randint(5, 15)

    # Define bounding box
    x_min, x_max = max(0, cx - r), min(size, cx + r + 1)
    y_min, y_max = max(0, cy - r), min(size, cy + r + 1)

    # Create small local grid
    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    local_mask = (xx - cx)**2 + (yy - cy)**2 <= r**2

    image_gray[y_min:y_max, x_min:x_max][local_mask] += np.random.uniform(0.3, 0.5)
```

**Severity:** Medium - Scales poorly with image size (O(size^2) vs O(r^2) per defect)

---

### 3. Non-Vectorized Random Generation

**Location:** `detection_demo.py:28-29`

**Issue:** Random values are generated one at a time inside the loop.

```python
# CURRENT
for _ in range(num_defects):
    x, y = np.random.randint(50, size-50, 2)
    defect_size = np.random.randint(5, 15)
```

**Fix:** Pre-generate all random values:
```python
# OPTIMIZED
positions = np.random.randint(50, size-50, (num_defects, 2))
defect_sizes = np.random.randint(5, 15, num_defects)
intensities = np.random.uniform(0.3, 0.5, num_defects)

for i in range(num_defects):
    x, y = positions[i]
    defect_size = defect_sizes[i]
    intensity = intensities[i]
    # ... rest of loop
```

**Severity:** Low - Minor overhead, but demonstrates good vectorization practice

---

### 4. Intermediate Array Allocations

**Location:** `detection_demo.py:35-36`

**Issue:** Creates temporary arrays during noise addition.

```python
# CURRENT
image_gray += np.random.normal(0, 0.05, image_gray.shape)
image_gray = np.clip(image_gray, 0, 1)
```

**Fix:** Use in-place operations:
```python
# OPTIMIZED
noise = np.random.normal(0, 0.05, image_gray.shape)
np.add(image_gray, noise, out=image_gray)
np.clip(image_gray, 0, 1, out=image_gray)
```

**Severity:** Low - Only matters for very large images or memory-constrained environments

---

### 5. Repeated Constant Calculation in Visualization Loop

**Location:** `detection_demo.py:72-77`

**Issue:** `np.sqrt(2)` is computed for each blob iteration.

```python
# CURRENT
for blob in blobs:
    y, x, sigma = blob
    radius = sigma * np.sqrt(2)  # Computed 15+ times
    circle = mpatches.Circle((x, y), radius, ...)
```

**Fix:** Precompute constant:
```python
# OPTIMIZED
SQRT_2 = np.sqrt(2)  # Compute once
for blob in blobs:
    y, x, sigma = blob
    radius = sigma * SQRT_2
```

**Severity:** Very Low - Python/NumPy likely optimizes this, but good practice

---

### 6. Individual Plot Calls Instead of Vectorized Scatter

**Location:** `detection_demo.py:77`

**Issue:** Each blob center marker is plotted individually, triggering multiple artist updates.

```python
# CURRENT (Inefficient)
for blob in blobs:
    y, x, sigma = blob
    # ...
    ax[3].plot(x, y, '+', color='yellow', markersize=10, ...)  # N separate calls
```

**Impact:**
- N separate draw operations
- N separate artist objects added to axes
- Slower rendering for large numbers of detections

**Fix:** Use vectorized scatter plot:
```python
# OPTIMIZED
SQRT_2 = np.sqrt(2)

# Add all circles
for blob in blobs:
    y, x, sigma = blob
    radius = sigma * SQRT_2
    circle = mpatches.Circle((x, y), radius, color='red', linewidth=2, fill=False, alpha=0.8)
    ax[3].add_patch(circle)

# Add all centers at once with scatter
if len(blobs) > 0:
    centers_x = blobs[:, 1]
    centers_y = blobs[:, 0]
    ax[3].scatter(centers_x, centers_y, marker='+', c='yellow', s=100, linewidths=2, alpha=0.9)
```

**Severity:** Medium - Significant improvement for hundreds/thousands of detections

---

### 7. Blob Detection Parameters May Be Overkill for Demo

**Location:** `detection_demo.py:47`

**Issue:** `num_sigma=10` creates 10 scale-space levels for a simple 300x300 demo.

```python
blobs = blob_log(image_smoothed, min_sigma=2, max_sigma=8, num_sigma=10, threshold=0.1)
```

**Impact:**
- Creates 10 Laplacian of Gaussian filtered images
- Each requires convolution over entire image
- May be excessive for educational demo

**Fix:** Reduce scale levels for demo purposes:
```python
# For demo, 5 levels is often sufficient
blobs = blob_log(image_smoothed, min_sigma=2, max_sigma=8, num_sigma=5, threshold=0.1)
```

**Severity:** Low - Acceptable for demo, but worth noting for production

---

## Summary Table

| Issue | Location | Severity | Type |
|-------|----------|----------|------|
| Redundant grid creation | Lines 27-32 | Medium | Unnecessary allocation |
| Full-image mask | Lines 30-32 | Medium | Inefficient algorithm |
| Non-vectorized randoms | Lines 28-29 | Low | Anti-pattern |
| Intermediate arrays | Lines 35-36 | Low | Memory inefficiency |
| Repeated sqrt(2) | Lines 72-77 | Very Low | Micro-optimization |
| Individual plot calls | Line 77 | Medium | Rendering inefficiency |
| Excessive scale levels | Line 47 | Low | Over-computation |

---

## Production Considerations

While these issues are minor for this 300x300 demo image, they become significant in production scenarios:

1. **Real-time video processing** (30+ fps): Every millisecond matters
2. **Large images** (4K, 8K): O(n^2) operations become bottlenecks
3. **Batch processing**: Processing thousands of images amplifies inefficiencies
4. **Edge devices**: Memory and compute constraints are tight

---

## Recommendations

1. **For this demo:** Issues are cosmetic; code clarity is more important
2. **For production:** Apply all optimizations, especially:
   - Localized mask computation (Issue #2)
   - Vectorized plotting (Issue #6)
   - Pre-allocation of arrays

3. **Testing:** Always benchmark before and after optimizations to verify improvements

---

*Analysis performed: January 2026*
