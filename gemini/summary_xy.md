# Summary of functions using "xy" coordinate convention

This report details functions and methods in the scikit-image library that use the "xy" coordinate convention, where 'x' corresponds to columns (axis 1) and 'y' corresponds to rows (axis 0).

## `skimage.feature.corner`

*   **`structure_tensor`**
    *   **Reasoning:** Accepts an `order` parameter that can be set to 'xy'. When set, the function uses the last axis (x) first for gradient computation.
    *   **GitHub Links:**
        *   [Docstring for `order` parameter](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/corner.py#L80)
        *   [Code using `order='xy'`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/corner.py#L125)

*   **`hessian_matrix`**
    *   **Reasoning:** Accepts an `order` parameter that can be set to 'xy'. This changes the order of the returned tensor elements from row/column-based ('rc') to x/y-based (Hxx, Hxy, Hyy).
    *   **GitHub Links:**
        *   [Docstring for `order` parameter](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/corner.py#L316)
        *   [Code using `order='xy'`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/corner.py#L351)

## `skimage.filters.rank`

*   **Rank Filters** (`autolevel`, `equalize`, `gradient`, `maximum`, `mean`, etc.)
    *   **Reasoning:** Many functions in the `skimage.filters.rank` module accept `shift_x` and `shift_y` parameters. These parameters offset the center of the footprint in a manner consistent with the 'xy' coordinate convention (shift_x for columns, shift_y for rows).
    *   **GitHub Link:**
        *   [Docstring for `shift_x`, `shift_y`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/filters/rank/generic.py#L108)

## `skimage.transform`

### `_warps` submodule

*   **`warp`**
    *   **Reasoning:** The `inverse_map` callable is expected to process coordinates in `(col, row)` format, which corresponds to `(x, y)`.
    *   **GitHub Link:**
        *   [Docstring for `inverse_map` parameter](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py#L1024)

*   **`warp_coords`**
    *   **Reasoning:** The `coord_map` callable is documented to work with `(col, row)` pairs, which is `(x, y)`.
    *   **GitHub Link:**
        *   [Docstring for `coord_map` parameter](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py#L782)

*   **`swirl`**
    *   **Reasoning:** The `center` parameter is documented as a `(column, row)` tuple, which is an `(x, y)` convention.
    *   **GitHub Link:**
        *   [Docstring for `center` parameter](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py#L707)

*   **`rotate`**
    *   **Reasoning:** The `center` parameter is documented in terms of `(cols, rows)`, which corresponds to an `(x, y)` convention.
    *   **GitHub Link:**
        *   [Docstring for `center` parameter](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py#L465)

### `hough_transform` submodule

*   **`hough_line`**
    *   **Reasoning:** The docstring explicitly states that the X and Y axes are horizontal and vertical respectively, which is the 'xy' convention. The input `theta` and output `angles` and `distances` also operate within this framework.
    *   **GitHub Link:**
        *   [Notes in Docstring](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/hough_transform.py#L261)

*   **`probabilistic_hough_line`**
    *   **Reasoning:** The function returns a list of lines in the format `((x0, y0), (x1, y1))`, where `x` is the column index and `y` is the row index.
    *   **GitHub Link:**
        *   [Returns section of Docstring](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/hough_transform.py#L330)

*   **`hough_circle_peaks`**
    *   **Reasoning:** Returns center coordinates as `cx, cy`. Internally, these are obtained from `_prominent_peaks` where `cx` corresponds to image axis 1 (columns) and `cy` to image axis 0 (rows).
    *   **GitHub Link:**
        *   [Returns section of Docstring](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/hough_transform.py#L351)

### `_geometric` submodule

*   **Geometric Transformations** (`AffineTransform`, `SimilarityTransform`, `EuclideanTransform`, `ProjectiveTransform`, etc.)
    *   **Reasoning:** These classes operate on `(N, 2)` coordinate arrays where columns are interpreted as `(x, y)` (or `(col, row)`). Their `__call__` and `estimate` methods, as well as parameters like `translation`, follow the 'xy' convention.
    *   **GitHub Link:**
        *   [Docstring for `AffineTransform`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_geometric.py#L1615)