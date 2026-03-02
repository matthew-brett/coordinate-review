# Watershed Axis-Order Sensitivity: Investigation Summary (Final)

This document summarizes the final analysis of the `watershed` function in `scikit-image` and its behavior with respect to image array axis ordering.

## Core Findings

1.  **Tie-Breaking Sensitivity:** The `watershed` function is sensitive to axis ordering when the input image contains **ties** (pixels with identical intensities). This is because the algorithm uses a priority queue where ties are broken using an `age` parameter (push order) and a fixed neighbor exploration order, both of which are tied to the array's raveled memory layout (C-order).
2.  **Invariance with Unique Values:** If the input image has **strictly unique values** (no ties) and **identical markers** are used, the algorithm is **invariant** to axis transposition. In this case, the pop order from the priority queue is strictly determined by pixel intensity, and the greedy neighbor claiming always results in the same physical boundaries.
3.  **Apparent Non-Invariance in Practice:** In real-world images (like `cells3d`) or images with insufficient noise, axis-order sensitivity often appears to persist because:
    *   **Hidden Ties:** Floating-point precision may not be sufficient to break all ties in large images with large plateaus. Even a few remaining ties will cause the boundaries to shift upon transposition.
    *   **Marker Generation:** If markers are generated automatically (e.g., via `local_minima` and `ndi.label`), the labels assigned to markers depend on a scan-order traversal, which changes when axes are transposed. While re-labeling can handle this, it adds a layer of complexity to the comparison.
    *   **Implementation Errors:** In some test cases (including the provided notebook), an `AssertionError` was triggered not by a labeling mismatch, but by the comparison function returning `None` (which is falsy in an `assert` statement).

## Conclusion
The `watershed` implementation is mathematically sound but structurally dependent on array traversal order for resolving ambiguities (ties). For images with strictly unique values and fixed seeds, the algorithm is robust to axis transposition. In typical imaging applications where ties are common (e.g., quantized integer data), axis-order sensitivity is an expected behavior of this implementation.

---

## References
*   [skimage/segmentation/_watershed.py](https://github.com/scikit-image/scikit-image/blob/main/skimage/segmentation/_watershed.py)
*   [skimage/segmentation/_watershed_cy.pyx](https://github.com/scikit-image/scikit-image/blob/main/skimage/segmentation/_watershed_cy.pyx)
*   [skimage/segmentation/heap_watershed.pxi](https://github.com/scikit-image/scikit-image/blob/main/skimage/segmentation/heap_watershed.pxi)
