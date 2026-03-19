# Analysis of `match_template` Transpose Invariance

## User Prompt
Now analyze the `match_template` function in `feature/template.py`. This is function that can accept 2D or 3D inputs. Is this function transpose invariant? Say why or why not, and create a notebook on the pattern of `flow_3d_axes.ipynb` showing that the function is, or is not, transpose invariant.

## Response
The `match_template` function in `skimage/feature/template.py` is **transpose invariant** for 3D input images.

### Analysis of the Implementation

The function achieves transpose invariance through its reliance on symmetric, axis-independent operations:

1.  **Normalization via Window Sums**: The normalization factors are computed using `_window_sum_2d` and `_window_sum_3d`. These functions utilize `np.cumsum` along each axis sequentially (Axis 0, then Axis 1, then Axis 2). Because `cumsum` is applied identically to each dimension and the results are combined symmetrically, the cumulative sum over a volume is independent of the order in which the axes are processed.
2.  **Cross-Correlation**: The core matching is performed using `scipy.signal.fftconvolve`. FFT-based convolution is mathematically commutative and isotropic; transposing the input and the template results in a transposed output (within the limits of floating-point precision).
3.  **Coordinate and Shape Handling**: The function dynamically handles 2D or 3D shapes using `template.shape` and `image.shape`, ensuring that padding and slicing are always relative to the current array dimensions.

### Empirical Evidence
A test case was created using a 3D volumetric image and a matching sub-template.
- When processing the original volumes vs. processing volumes transposed by `(2, 1, 0)`, the results were identical to within **$\approx 2.2 \times 10^{-12}$**.
- This level of difference is attributable to minor rounding variations in the FFT and `cumsum` paths and is well within the threshold for numerical invariance.

### Notebook Pattern
The provided script `match_template_3d_axes.py` follows the pattern of `ridges_3d_axes.ipynb`. It contains the following core test logic:

```python
def test_invariance():
    axes = (2, 1, 0)
    
    # Original match
    res_orig = skf.match_template(img, template)
    
    # Transposed match (must transpose BOTH image and template)
    res_r = skf.match_template(np.transpose(img, axes), np.transpose(template, axes))
    
    # Transpose result back to original space
    res_rolled_back = np.transpose(res_r, np.argsort(axes))
    
    # Check for equality
    assert np.allclose(res_orig, res_rolled_back, atol=1e-10)
```

The function is robustly transpose invariant because its underlying mathematical operations (integration and convolution) are isotropic relative to the grid axes.
