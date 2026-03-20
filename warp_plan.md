# Plan for refactoring warp functions

## warp

See start of refactor in `./warp_refactor.py` (initially from [JNI's
gist](https://gist.github.com/jni/08d0328b89047deceb249540635aee72).

## `_linear_polar_mapping` / `_log_polar_mapping` (used by `warp_polar`)

**Convention: xy for output coords; center as (row, col) = ij**

**Evidence:**

- Docstring: `output_coords` are “(col, row)”; `center` is “tuple (row, col)”.
- Implementation uses `center[0]` for radial offset in the first axis and `center[1]` for the second, consistent with (row, col) for center.

**Code:**
[`src/skimage/transform/_warps.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py)
— `_linear_polar_mapping`, `_log_polar_mapping`.

## `warp` and `_warp_fast` (Cython)

**Convention: xy**

**Evidence:**

- In
  [`_warps_cy.pyx`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps_cy.pyx),
  the output loop iterates over `(tfr, tfc)` (output row, col). The inverse
  map is called with `(tfc, tfr)` as `(x, y)`:

```python
for tfr in range(out_r):
    for tfc in range(out_c):
        transform_func(tfc, tfr, &M[0, 0], &c, &r)
        interp_func(&img[0, 0], rows, cols, r, c, ...)
```

- So output coordinates passed to the transform are **(x, y) = (column, row)**. The transform returns source `(c, r)` (col, row), and the image is sampled at `img[r, c]` (row, col). The _meaning_ of coordinates in the transform chain is xy.

**Callers:** `rotate`, `swirl`, `radon`, `warp_polar`, geometric transforms
used as `inverse_map`. Examples:
[`doc/examples/transform/plot_geometric.py`](https://github.com/scikit-image/scikit-image/blob/main/doc/examples/transform/plot_geometric.py)
uses `warp` with `SimilarityTransform`;
[`tests/.../test_warps.py`](https://github.com/scikit-image/scikit-image/blob/main/tests/skimage/transform/test_warps.py)
upses `warp` with `AffineTransform` / `ProjectiveTransform`.

### 1.2 `warp_coords`

**Convention: xy (user-facing coord_map)**

**Evidence:**

- Docstring says coordinates are “(row, col) pairs”, but the implementation builds a grid with **columns first**:

```python
# _warps.py
tf_coords = np.indices((cols, rows), dtype=dtype).reshape(2, -1).T  # (col, row) order
tf_coords = coord_map(tf_coords)
# ...
coords[1, ...] = tf_coords[0, ...]  # map_coords axis 1
coords[0, ...] = tf_coords[1, ...]  # map_coords axis 0
```

- The example uses `xy` and a shift `[-20, 10]` (x, y):

```python
def shift_up10_left20(xy):
    return xy - np.array([-20, 10])[None, :]
```

- So the **coord_map** receives and returns **(x, y) = (col, row)**. The result is then rearranged so that `map_coordinates` gets (axis0, axis1) = (row, col).

**Code:**
[`src/skimage/transform/_warps.py`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py)
— `warp_coords`, and the `warp` branch that uses it (e.g. when `inverse_map`
is a callable).

