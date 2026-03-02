### Summary of Investigation into `slic` Transpose Invariance

I analyzed the `slic` function in `skimage/segmentation/slic_superpixels.py` and its behavior across the images from `slic_3d_axes.ipynb`.

#### 1. Why `slic` is Transpose Invariant for `cells3d` (Images 0 & 1) but not `brain` (Image 2)

*   **Grid Seeds are Invariant:** The `slic` algorithm starts by placing seeds on a regular grid using `regular_grid`. I verified that `regular_grid` is transpose-invariant (after re-ordering), meaning it produces the same physical seed locations regardless of axis order.
*   **Tie-Breaking in K-Means:** The core k-means assignment in `_slic_cython` iterates through segments and, for each segment, through a window of pixels. When multiple segments are equidistant to a pixel, the **first** segment to process that pixel "claims" it (by setting `distance[z, y, x]` and `nearest_segments[z, y, x]`).
    *   **The Difference:** Images 0 and 1 have complex structures and few large plateaus. Even if some pixels are claimed by different segments due to iteration order, the boundaries remain largely the same, making them **equivalent** under the notebook's test (which allows for label re-ordering).
    *   **The `brain` Image (Image 2):** This image has a **very large background (29% zeros)**. In such a large plateau, many pixels are exactly equidistant to multiple seeds. Changing the axis order changes the iteration order of both the segments and the pixels within the search windows. In a large plateau, this significantly shifts the "claim boundaries" between segments, leading to non-equivalent superpixels.
*   **Verification:** I verified that adding small random noise to the `brain` image (breaking all ties) restores transpose invariance.

#### 2. Scan-Order Dependent Labeling

*   **Connectivity Enforcement:** After k-means, `slic` optionally calls `_enforce_label_connectivity_cython`. This function performs a Breadth-First Search (BFS) to group connected pixels and re-labels them.
*   **Sequential Labeling:** The labels are assigned in the order the BFS finds new components, which depends on a scan-order traversal (Z, then Y, then X).
*   **Result:** This is why even Images 0 and 1 do not produce **identical** label matrices after transposition (the labels are permuted), even though they are **equivalent** (the shapes of the segments are the same).

### Conclusion

The `slic` function's lack of transpose invariance on the `brain` image is primarily due to **tie-breaking in the k-means assignment step** in the presence of large plateaus of identical values. For images with more unique structure, the algorithm is equivalent under transposition, but the final label indices will still differ due to the **scan-order dependent labeling** in the connectivity enforcement step.

Relevant code:
*   [skimage/segmentation/slic_superpixels.py](https://github.com/scikit-image/scikit-image/blob/main/skimage/segmentation/slic_superpixels.py)
*   [skimage/segmentation/_slic.pyx](https://github.com/scikit-image/scikit-image/blob/main/skimage/segmentation/_slic.pyx)
*   [skimage/util/_regular_grid.py](https://github.com/scikit-image/scikit-image/blob/main/skimage/util/_regular_grid.py)
