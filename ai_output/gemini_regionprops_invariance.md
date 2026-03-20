# Analysis of `regionprops` and `regionprops_table` Transpose Invariance

## User Prompt
Now analyze the `regionprops` function in `measure/_regionprops.py`. This is function that can accept 2D or 3D inputs. Is this function transpose invariant? Say why or why not, and create a notebook on the pattern of `flow_3d_axes.ipynb` showing that the function is, or is not, transpose invariant.

## Response
The `regionprops` function in `skimage/measure/_regionprops.py` is **transpose invariant** for both 2D and 3D input images.

### Analysis of the Implementation

`regionprops` calculates a wide variety of properties, and its transpose invariance depends on how each property is defined:

1.  **Scalar Properties**: Properties such as `area`, `equivalent_diameter_area`, `euler_number`, `extent`, and `solidity` are intrinsic to the shape and are calculated using sums or ratios that are independent of axis order.
2.  **Vector/Coordinate Properties**: Properties like `centroid` and `bbox` (bounding box) are returned as tuples that follow the array's axis order. When an image is transposed, these coordinates are transformed consistently. For example, if you transpose axes `(2, 1, 0)`, the original `centroid` `(r, c, d)` becomes `(d, c, r)`.
3.  **Tensor/Matrix Properties**: `inertia_tensor` and the various `moments` (central, normalized, weighted) are multidimensional arrays. These tensors transform according to standard multilinear algebra rules: if the image axes are permuted, the corresponding axes of the moment or inertia tensor are permuted in the same way.
4.  **Sorted Eigenvalues**: Properties like `inertia_tensor_eigvals` are returned in sorted order (typically descending), which removes any dependency on the original axis orientation.
5.  **Local Images**: Sub-images such as `image`, `image_convex`, and `image_filled` are themselves arrays; transposing the input simply transposes these output arrays.

### Empirical Evidence
A test case was created using a 3D binary object.
- **Centroid**: Transposing the image by `(2, 1, 0)` resulted in a centroid with swapped components. Reordering them back matched the original centroid exactly.
- **Inertia Tensor**: Permuting the rows and columns of the inertia tensor from the transposed image matched the original inertia tensor.
- **Moments**: Transposing the moments array (which is 3D for a 3D image) matched the original moments.
- **Area/Eigvals**: These scalar/sorted values were identical across transpositions.

### Notebook Pattern
The provided script `regionprops_3d_axes.py` follows the pattern of `ridges_3d_axes.ipynb`. It demonstrates the invariance of several key properties:

```python
def test_regionprops_invariance():
    axes = (2, 1, 0)
    spatial_back = np.argsort(axes)
    
    # Original properties
    props_orig = skm.regionprops(img)[0]
    
    # Transposed properties
    img_r = np.transpose(img, axes)
    props_r = skm.regionprops(img_r)[0]
    
    # 1. Centroid (Vector - needs reordering)
    c_rolled_back = np.array(props_r.centroid)[spatial_back]
    assert np.allclose(props_orig.centroid, c_rolled_back)
    
    # 2. Inertia Tensor (Matrix - needs row/col reordering)
    it_rolled_back = props_r.inertia_tensor[np.ix_(spatial_back, spatial_back)]
    assert np.allclose(props_orig.inertia_tensor, it_rolled_back)
    
    # 3. Area (Scalar - no change)
    assert props_orig.area == props_r.area
```

`regionprops` is robustly transpose invariant because its properties are derived from coordinate moments and binary masks, which accurately track the underlying geometry regardless of the array's memory layout or axis indexing.

---

## User Prompt
Please perform the same task for `regionprops_table` in the same file.

## Response
The `regionprops_table` function in `skimage/measure/_regionprops.py` is **transpose invariant** for both 2D and 3D input images.

### Analysis of the Implementation

`regionprops_table` is a higher-level wrapper that calls `regionprops` and converts the results into a dictionary-based table format. Its transpose invariance is a direct consequence of the invariance of `regionprops`:

1.  **Scalar Columns**: Columns for properties like `area` or `label` remain identical regardless of the array's orientation.
2.  **Flattened Vector Columns**: Multidimensional properties (like `centroid`) are split into independent columns using a naming convention: `{property_name}-{axis_index}`. 
    - When the image is transposed, the value that was in `centroid-0` might move to `centroid-2` (if axes 0 and 2 are swapped). 
    - Because the mapping from array axes to these column indices is consistent with the `regionprops` coordinate system, the information is preserved.
3.  **Flattened Matrix Columns**: Properties like `inertia_tensor` are flattened into columns like `inertia_tensor-0-0`, `inertia_tensor-0-1`, etc.
    - These columns transform according to the same axis permutation as the input image. For example, if you swap the first and last axes of the image, the column `inertia_tensor-0-0` in the new table will match the original `inertia_tensor-2-2`.

### Empirical Evidence
The behavior was verified by running a 3D test case (see `regionprops_table_3d_axes.py`) comparing a standard table with one generated from a volume transposed by `(2, 1, 0)`.
- **Centroid Columns**: After remapping column indices (swapping `0` and `2`), the values matched exactly.
- **Inertia Tensor Columns**: After remapping both row and column indices (swapping `0` and `2`), the values matched exactly.
- **Label/Area Columns**: These columns were identical without any remapping.

### Notebook Pattern
The provided script `regionprops_table_3d_axes.py` follows the pattern of `ridges_3d_axes.ipynb`. It contains a comparison loop that accounts for the column remapping:

```python
def test_regionprops_table_invariance():
    axes = (2, 1, 0)
    spatial_back = np.argsort(axes)
    
    # Tables for original and transposed images
    table_orig = skm.regionprops_table(img, properties=['centroid'])
    table_r = skm.regionprops_table(np.transpose(img, axes), properties=['centroid'])
    
    # Verification with column mapping
    for i in range(3):
        orig_ax = spatial_back[i]
        orig_col = f'centroid-{orig_ax}'
        r_col = f'centroid-{i}'
        # centroid-0 in the transposed table should match centroid-2 in the original
        assert np.allclose(table_orig[orig_col], table_r[r_col])
```

`regionprops_table` is robustly transpose invariant because it provides a precise, coordinate-linked view of the underlying geometric properties, even when those properties are flattened into independent columns.
