# Coordinate Convention Review

This document summarizes findings regarding the use of "ij" (array coordinates, row/column) versus "xy" (Cartesian coordinates, column/row) conventions in the `scikit-image` codebase.

The investigation focused on `skimage.draw`, `skimage.transform`, `skimage.feature`, `skimage.measure`, and `skimage.segmentation`.

## Summary of Findings

*   **`skimage.draw`**: Consistently uses **ij (row, column)** convention.
*   **`skimage.transform`**: Mixed.
    *   Image warping and resizing functions (`warp`, `resize`) use **ij** (row, column) for image shapes and indices.
    *   Geometric transforms (`AffineTransform`, `ProjectiveTransform`, etc.) operate in **xy** (column, row) space.
    *   `warp` expects the `inverse_map` (callable or transform) to handle **xy** coordinates.
    *   `rotate` and `swirl` use **xy** (column, row) for the `center` parameter.
    *   `warp_polar` uses **ij** (row, column) for the `center` parameter, creating an inconsistency with other transform functions.
*   **`skimage.feature`**: Predominantly **ij** (row, column).
    *   `corner` functions return `(row, column)` coordinates.
    *   Gradient-based functions (`structure_tensor`, `hessian_matrix`) default to `order='rc'` (ij) but support `order='xy'`.
*   **`skimage.measure`**: Consistently uses **ij** (row, column) for contours, centroids, and moments.
*   **`skimage.segmentation`**: `active_contour` uses **ij** (row, column) coordinates.

## Detailed Evidence

### `skimage.draw`

Functions in this module overwhelmingly use `row` and `column` parameters and return `(row, column)` indices.

*   **File**: `skimage/draw/draw.py` ([Source](https://github.com/scikit-image/scikit-image/blob/main/skimage/draw/draw.py))
*   **Convention**: **ij (row, column)**

**Evidence:**
*   `line(r0, c0, r1, c1)`
*   `disk(center, radius)` where `center` is `(row, column)`.
*   `polygon(r, c)`
*   `ellipse(r, c, ...)`
*   `rectangle(start, end, ...)` where `start` is `([plane,] row, column)`.

### `skimage.transform`

This module shows the most significant mixing of conventions, distinguishing between "array space" (ij) and "transform space" (xy).

*   **File**: `skimage/transform/_geometric.py` ([Source](https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_geometric.py))
*   **Convention**: **xy (column, row)**

**Evidence:**
*   `ProjectiveTransform` docstring examples imply $x, y$ usage standard in computer vision (where $x$ is horizontal/column).
    ```python
    X = (a0*x + a1*y + a2) / (c0*x + c1*y + 1)
    ```
*   `AffineTransform` parameters include `scale`, `rotation` (around origin), `shear`, `translation`. These are applied in a Cartesian-like 2D plane (xy).

*   **File**: `skimage/transform/_warps.py` ([Source](https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_warps.py))
*   **Convention**: Mixed.

**Evidence:**
*   **`warp`**:
    *   Generates coordinates for the inverse map in **xy (column, row)** order.
    *   Expects `inverse_map` to accept and return **xy** coordinates.
    *   Converts the result back to **ij (row, column)** indices for `scipy.ndimage.map_coordinates`.
    *   *Code snippet (internal `warp_coords`)*:
        ```python
        # Reshape grid coordinates into a (P, 2) array of (row, col) pairs
        tf_coords = np.indices((cols, rows), dtype=dtype).reshape(2, -1).T
        # Note: np.indices((cols, rows)) produces first dimension 0..cols-1 (x), second 0..rows-1 (y)
        # ...
        tf_coords = coord_map(tf_coords) # coord_map sees (x, y)
        ```
*   **`rotate`**:
    *   Uses **xy** for `center`.
    *   *Code snippet*:
        ```python
        if center is None:
            center = np.array((cols, rows)) / 2.0 - 0.5
        # ...
        tform1 = SimilarityTransform(translation=center)
        ```
*   **`swirl`**:
    *   Uses **xy** for `center` (docstring says `(column, row)`).
*   **`warp_polar`**:
    *   Uses **ij** for `center`.
    *   *Code snippet*:
        ```python
        if center is None:
            center = (np.array(image.shape)[:2] / 2) - 0.5
        # ...
        rr = ((output_coords[:, 0] / k_radius) * np.sin(angle)) + center[0]
        # center[0] is used for row calculation
        ```

### `skimage.feature`

*   **File**: `skimage/feature/corner.py` ([Source](https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/corner.py))
*   **Convention**: **ij (row, column)** (default).

**Evidence:**
*   `corner_peaks`: Returns `(row, column)`.
*   `corner_subpix`: Input `corners` is `(row, col)`, returns `(row, col)`.
*   `structure_tensor`: Has `order` parameter.
    *   `order='rc'` (default): "indicates the use of the first axis initially". (**ij**)
    *   `order='xy'`: "indicates the usage of the last axis initially". (**xy**)

### `skimage.measure`

*   **File**: `skimage/measure/_find_contours.py` ([Source](https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_find_contours.py))
*   **Convention**: **ij (row, column)**

**Evidence:**
*   `find_contours`: Returns list of arrays where each row is `(row, column)`.

*   **File**: `skimage/measure/_moments.py` ([Source](https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_moments.py))
*   **Convention**: **ij (row, column)**

**Evidence:**
*   `moments`, `moments_central`: `M[1, 0]` corresponds to the moment along the first axis (row).
*   `centroid`: Returns `(row_center, col_center, ...)` based on array axes.

### `skimage.segmentation`

*   **File**: `skimage/segmentation/active_contour_model.py` ([Source](https://github.com/scikit-image/scikit-image/blob/main/skimage/segmentation/active_contour_model.py))
*   **Convention**: **ij (row, column)**

**Evidence:**
*   `active_contour`:
    *   Input `snake` is used as `x = snake[:, 1]`, `y = snake[:, 0]` to interpolate into image. This implies `snake[:, 0]` is row, `snake[:, 1]` is column.
    *   Returns `np.stack([y, x], axis=1)`, preserving the `(row, col)` order.
