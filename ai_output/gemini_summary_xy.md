# Summary of Coordinate Conventions in `scikit-image`

This review identifies the usage of **Numpy ("ij" / "rc")** and **Imaging ("xy")** coordinate conventions across various modules. In the "ij" convention, the first coordinate refers to the row (axis 0), while in the "xy" convention, the first coordinate refers to the column (axis 1).

## 1. Numpy / "ij" / "rc" Convention (Row-first)

This is the most common convention in the codebase, particularly for functions that return coordinates intended for direct array indexing.

*   **`skimage.draw` module**
    *   **Functions:** `line`, `circle_perimeter`, `circle_perimeter_aa`, `polygon`, etc.
    *   **Evidence:** Parameters are explicitly named `r` and `c` (or `r0, c0`). The returns are `rr, cc` arrays used as `img[rr, cc]`.
    *   **Source:** [src/skimage/draw/draw.py](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/draw/draw.py)
*   **`skimage.measure.find_contours`**
    *   **Evidence:** Docstring states it returns `(row, column)` coordinates.
    *   **Source:** [src/skimage/measure/_find_contours.py](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/measure/_find_contours.py)
*   **Feature Detectors (`peak_local_max`, `corner_peaks`, `blob_log`, `blob_dog`, `blob_doh`)**
    *   **Evidence:** These functions return coordinate lists where the first element is the row index, allowing for `img[coord[0], coord[1]]`.
    *   **Source:** [src/skimage/feature/peak.py](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/peak.py), [src/skimage/feature/corner.py](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/corner.py)
*   **`skimage.registration.phase_cross_correlation`**
    *   **Evidence:** Returns a `shift` vector where the order matches the axes of the input array.
    *   **Source:** [src/skimage/registration/_phase_cross_correlation.py](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/registration/_phase_cross_correlation.py)
*   **`ORB` and `SIFT` Classes**
    *   **Evidence:** The `keypoints` and `positions` attributes are explicitly documented as `(row, col)`.
    *   **Source:** [src/skimage/feature/orb.py](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/orb.py), [src/skimage/feature/sift.py](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/sift.py)

## 2. Imaging / "xy" Convention (Column-first)

This convention is primarily found in coordinate transformation and interpolation routines, matching standard mathematical and computer vision library (e.g., OpenCV) expectations.

*   **`skimage.transform.warp` and `warp_coords`**
    *   **Evidence:** When a callable `inverse_map` is provided, it receives an array of `(col, row)` coordinates.
    *   **Source:** [src/skimage/transform/_warps.py](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py)
*   **Geometric Transforms (`AffineTransform`, `ProjectiveTransform`, etc.)**
    *   **Evidence:** Parameters like `translation=(tx, ty)` and `scale=(sx, sy)` refer to `(column, row)` shifts and scales. The docstrings define transformations using `X` and `Y` where `X` is the horizontal axis (axis 1).
    *   **Source:** [src/skimage/transform/_geometric.py](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_geometric.py)
*   **`skimage.filters.rank` (Generic filters)**
    *   **Evidence:** Parameters `shift_x` and `shift_y` are used to offset the footprint. `shift_x` corresponds to axis 1 and `shift_y` to axis 0.
    *   **Source:** [src/skimage/filters/rank/generic.py](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/filters/rank/generic.py)
*   **`structure_tensor` and `hessian_matrix`**
    *   **Evidence:** Both functions include an optional `order='xy'` parameter (defaulting to `'rc'`), which reverses the axis order for gradient computation.
    *   **Source:** [src/skimage/feature/corner.py](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/corner.py)

## 3. Notable Inconsistencies and Confusion

*   **`skimage.segmentation.active_contour`**
    *   **Issue:** The implementation correctly follows the **"ij"** convention internally (using `RectBivariateSpline` with `(row, col)` coordinates). However, the docstring example contains a distance calculation (`dist = np.sqrt((45-snake[:, 0])**2 + (35-snake[:, 1])**2)`) that erroneously treats the snake as `(col, row)`, as the target circle was created at row 35 and column 45.
    *   **Source:** [src/skimage/segmentation/active_contour_model.py](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/segmentation/active_contour_model.py)
*   **`skimage.transform.warp` Docstring**
    *   **Issue:** The docstring describes `inverse_map` as receiving `(col, row)` for 2D images but `(row, col)` for N-D coordinate arrays, leading to potential user confusion.
