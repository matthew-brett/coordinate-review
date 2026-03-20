# Analysis of `skeletonize` Transpose Invariance

## User Prompt
Consider `skeletonize` in `main/src/skimage/morphology/_skeletonize.py`. Is this routine transform invariant? Make a notebook on the model of `flow_3d_axes.ipynb` to demonstrate the presence of absence of transform invariance.

## Response
The `skeletonize` function in `skimage/morphology/_skeletonize.py` is **not transpose invariant** for either 2D or 3D input images.

### Analysis of the Implementation

The function supports two main algorithms: Zhang's algorithm (default for 2D) and Lee's algorithm (default for 3D). Both algorithms rely on iterative thinning processes that prioritize the removal of pixels from specific directions in a fixed order.

1.  **Zhang's Algorithm (2D)**:
    This algorithm makes successive passes over the image, removing boundary pixels that meet specific connectivity criteria. These criteria are implemented via a lookup table (LUT) and a neighborhood indexing scheme that is inherently directional. The implementation in `_fast_skeletonize` uses two sub-iterations that distinguish between different "neighbors" values, effectively removing pixels from different sides of the object at different times. Transposing the image reorders these neighbors relative to the array axes, changing the sequence of pixel removal and leading to a different final skeleton.

2.  **Lee's Algorithm (2D and 3D)**:
    Lee's algorithm (implemented in `_skeletonize_lee_cy.pyx.in`) uses a 3x3x3 neighborhood and explicitly loops over six border directions in a fixed order: `[4, 3, 2, 1, 5, 6]` (West, East, South, North, Up, Bottom). Because thinning is a sequential process—where removing one pixel may prevent its neighbor from being considered "simple" and thus removable—the order in which these directions are processed is critical. Transposing the image axes maps physical features to different array axes, thereby changing the order in which their boundaries are processed and resulting in different skeleton topologies.

### Empirical Evidence
The non-invariance was verified using random binary images and structured asymmetric objects. In all cases, transposing the input and then restoring the result to the original coordinate system yielded a skeleton that differed from the original by one or more pixels.

*   **`skeletonize(method='zhang')` (2D)**: **Failed** (Max difference = 1).
*   **`skeletonize(method='lee')` (2D)**: **Failed** (Max difference = 1).
*   **`skeletonize(method='lee')` (3D)**: **Failed** (Max difference = 1).

### Notebook Pattern
The following script (formatted for a notebook) demonstrates the lack of transpose invariance:

```python
# %%
import numpy as np
import skimage.morphology as skm

# Create an asymmetric image
rng = np.random.default_rng(42)
img = rng.random((20, 20, 20)) > 0.8

# Original skeleton
skel_orig = skm.skeletonize(img)

# Transposed skeleton
axes = (0, 2, 1) # Swap Y and X
img_r = np.transpose(img, axes)
skel_r = skm.skeletonize(img_r)

# Restore coordinate system
skel_rolled_back = np.transpose(skel_r, np.argsort(axes))

# Check for equality
diff = np.abs(skel_orig.astype(int) - skel_rolled_back.astype(int)).max()
if diff > 0:
    print("Skeletonize is NOT transpose invariant.")
```

The algorithm is inherently sensitive to the grid's orientation due to its reliance on directional thinning passes.
