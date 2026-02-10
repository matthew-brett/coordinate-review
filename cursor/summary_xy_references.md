# x and y References in the scikit-image Codebase

This report lists **variables**, **docstrings**, and **function/method names** that use `x` and `y` (or related names like `x0`, `y0`, `xc`, `yc`, `cx`, `cy`, `xcoords`, `ycoords`, `xs`, `ys`, `dx`, `dy`) and classifies each as:

- **xy convention:** x = column, y = row (imaging convention). Typical when used with `warp`, geometric transforms, Hough line/circle API, etc.
- **ij convention:** Row first, column second. Image indexing `img[row, col]`. Use of `x`/`y` in ij contexts is rare; when present, we note it.
- **agnostic:** Generic 2D (x,y) in math or API, feature space (e.g. Color-(x,y)), fit models, or meshgrids where axis ↔ row/col is not specified. Could be either depending on usage.

**Scope:** `src/skimage` (and selected doc/tests). Excluded: color-space X,Y,Z or xy chromaticity, generic symbols in equations (e.g. “x” in entropy Σ p(x) log p(x)), and Python loop variables like `x for x in iter` unless they denote coordinates.

---

## 1. `skimage.transform`

### 1.1 `_warps.py`

| Reference                                                                                                  | Type           | Convention   | Notes                                                                                                                             |
| ---------------------------------------------------------------------------------------------------------- | -------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| `_swirl_mapping(xy, center, ...)`                                                                          | function param | **xy**       | `xy` = (col, row) grid; `x, y = xy.T`; polar math uses `x0,y0` = center, `rho`, `theta`; output `xy[..., 0]`, `xy[..., 1]` = x,y. |
| `x, y = xy.T`; `x0, y0 = center`; `rho = sqrt((x-x0)**2 + (y-y0)**2)`; `theta = ... + arctan2(y-y0, x-x0)` | variables      | **xy**       | Swirl mapping: (x,y) = (col, row).                                                                                                |
| “Place the **y-coordinate** mapping” / “**x-coordinate** mapping”                                          | docstring      | **xy**       | `warp_coords` / `map_coordinates`: axis 0 → row, axis 1 → col; coord_map receives (x,y)=(col,row).                                |
| `any(x < y for x, y in zip(...))`                                                                          | variable       | **agnostic** | Loop vars over shape tuples; not image coords.                                                                                    |

### 1.2 `_warps_cy.pyx`

| Reference                                                                                          | Type            | Convention | Notes                                                             |
| -------------------------------------------------------------------------------------------------- | --------------- | ---------- | ----------------------------------------------------------------- |
| `_transform_metric(x, y, ...)`, `_transform_affine(x, y, ...)`, `_transform_projective(x, y, ...)` | function params | **xy**     | Transform input (x,y) = (col, row); output `x_[0]`, `y_[0]` same. |
| Docstring “\mathbf{x} = [x, y, 1]^T”; “translate **x** by 10 and **y** by 20”                      | docstring       | **xy**     | Homogeneous coords; geometric convention.                         |

### 1.3 `_radon_transform.pyx`

| Reference                                              | Type      | Convention | Notes                                                                                           |
| ------------------------------------------------------ | --------- | ---------- | ----------------------------------------------------------------------------------------------- |
| “(s, t) is the (**x**, **y**) system rotated by theta” | docstring | **xy**     | Physical (x,y) ↔ (s,t) rotated; mapped to `index_i`, `index_j` (row, col).                     |
| `ds, dx, dy, x0, y0, x, y, di, dj`                     | variables | **xy**     | (x,y) in rotated frame; `index_i = x + rotation_center`, `index_j = y + ...` map to array axes. |

### 1.4 `radon_transform.py`

| Reference                                           | Type      | Convention   | Notes                                                                   |
| --------------------------------------------------- | --------- | ------------ | ----------------------------------------------------------------------- |
| `x = np.arange(...)`; `xp=x, fp=col` in `np.interp` | variables | **agnostic** | `x` = 1D abscissa for interpolation over sinogram; not image (row,col). |

### 1.5 `hough_transform.py`

| Reference                                                                            | Type      | Convention   | Notes                                                                                                                  |
| ------------------------------------------------------------------------------------ | --------- | ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| “(**yc**, **xc**) is the center” (hough_ellipse result)                              | docstring | **xy**       | yc = row, xc = col.                                                                                                    |
| “((**x0**, **y0**), (**x1**, **y1**))” (probabilistic_hough_line)                    | docstring | **xy**       | (x,y) = (col, row).                                                                                                    |
| “**x** dimension” / “**y** dimension” (min_xdistance, min_ydistance)                 | docstring | **xy**       | x = col, y = row.                                                                                                      |
| “**cx**, **cy**” / “**x** and **y** center coordinates” (hough_circle_peaks)         | docstring | **xy**       | (cx,cy) = (col, row).                                                                                                  |
| Example “**y**, **x** = draw.circle_perimeter(**y_0**, **x_0**, …)”; “img[x, y] = 1” | docstring | **ij** (bug) | Draw returns (rr,cc); example swaps to (x,y) and uses `img[x,y]` (row,col), so x=row, y=col here. Known inconsistency. |
| `label_distant_points(xs, ys, min_xdistance, min_ydistance, …)`                      | function  | **xy**       | xs = x-coords (col), ys = y-coords (row).                                                                              |
| `cx`, `cy` lists; `cx.extend(x_p)`, `cy.extend(y_p)` from `_prominent_peaks`         | variables | **xy**       | (cx,cy) = (col, row).                                                                                                  |

### 1.6 `_hough_transform.pyx`

| Reference                                                                     | Type      | Convention      | Notes                                                                             |
| ----------------------------------------------------------------------------- | --------- | --------------- | --------------------------------------------------------------------------------- |
| “(**yc**, **xc**)”; “**yc**, **xc**” in result dtype                          | docstring | **xy**          | Center (row, col) stored as (yc, xc).                                             |
| `x, y = np.nonzero(img)`                                                      | variables | **ij** (naming) | Actually (row indices, col indices); name `x,y` suggests xy. Internal use varies. |
| `xc`, `yc`, `dx`, `dy`, `p1x`, `p2x`, `p1y`, `p2y` (ellipse)                  | variables | **xy**          | Ellipse center and deltas in (x,y) = (col, row).                                  |
| `x`, `y`, `x0`, `y0`, `x1`, `y1`, `dx`, `dy`, `px`, `py` (probabilistic line) | variables | **xy**          | Line endpoints and walking; `mask[y,x]` etc. API returns (x,y) = (col, row).      |

### 1.7 `_geometric.py`

| Reference                                                                                                        | Type      | Convention | Notes                                                                                 |
| ---------------------------------------------------------------------------------------------------------------- | --------- | ---------- | ------------------------------------------------------------------------------------- |
| “\mathbf{x} = [**x**, **y**, 1]^T”; “**x**, **y**” in Projective/Affine formulas                                 | docstring | **xy**     | Transform math; when used for images, (x,y) = (col, row).                             |
| “translate **x** by 10 and **y** by 20”                                                                          | docstring | **xy**     | Same.                                                                                 |
| “scale factors … **x** and **y** directions”; “**x** and **y** shear”; “translation … **x** (a2) and **y** (b2)” | docstring | **xy**     | Geometric params.                                                                     |
| “**x**, **y**[, z] translation parameters”                                                                       | docstring | **xy**     | Same.                                                                                 |
| “**x**, **y** coordinates to transform” (PolynomialTransform)                                                    | docstring | **xy**     | Abstract 2D; for images, xy.                                                          |
| `xs = src[:, 0]`, `ys = src[:, 1]`; `x = coords[:, 0]`, `y = coords[:, 1]`                                       | variables | **xy**     | Point sets for estimation / `__call__`; (x,y) = (col, row) when used with image warp. |

### 1.8 `_thin_plate_splines.py`

| Reference                               | Type      | Convention | Notes                         |
| --------------------------------------- | --------- | ---------- | ----------------------------- |
| “**x**, **y** coordinates to transform” | docstring | **xy**     | Same as geometric transforms. |

---

## 2. `skimage.feature`

### 2.1 `peak.py` (`_prominent_peaks`)

| Reference                                                                                                                                                | Type      | Convention | Notes                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---------- | -------------------------------------------------------- |
| “**x** and **y** indices”; “**xcoords**”, “**ycoords**”; “**x** dimension” / “**y** dimension”                                                           | docstring | **xy**     | x = column index, y = row index.                         |
| `ycoords_size`, `xcoords_size`; `ycoords_ext`, `xcoords_ext`; `ycoords_nh`, `xcoords_nh`; `ycoords_peaks`, `xcoords_peaks`; `ycoords_idx`, `xcoords_idx` | variables | **xy**     | Used with `img[ycoords_idx, xcoords_idx]`; y=row, x=col. |

### 2.2 `corner.py`

| Reference                                     | Type      | Convention | Notes                                                                     |
| --------------------------------------------- | --------- | ---------- | ------------------------------------------------------------------------- |
| `y, x = np.mgrid[-wext:wext+1, -wext:wext+1]` | variables | **xy**     | mgrid axis order; used for symmetric structure; (y,x) = (row, col) patch. |

### 2.3 `_haar.pyx`

| Reference                                                         | Type      | Convention   | Notes                                                           |
| ----------------------------------------------------------------- | --------- | ------------ | --------------------------------------------------------------- |
| `FEATURE_TYPE = {'type-2-**x**': 0, 'type-2-**y**': 1, …}`        | name      | **agnostic** | Feature orientation (horizontal vs vertical), not image coords. |
| `x`, `y`, `dx`, `dy` in rectangle loops; `integral(y, x, …)` etc. | variables | **ij**       | (y,x) used as (row, col) for integral image indexing.           |

### 2.4 `_gabor.py`

| Reference                 | Type      | Convention   | Notes                                     |
| ------------------------- | --------- | ------------ | ----------------------------------------- |
| `y, x = np.meshgrid(...)` | variables | **agnostic** | Kernel grid; axis order depends on usage. |

### 2.5 `_canny_cy.pyx`

| Reference             | Type      | Convention | Notes                   |
| --------------------- | --------- | ---------- | ----------------------- |
| `x`, `y` loop indices | variables | **ij**     | Pixel (row, col) loops. |

---

## 3. `skimage.measure`

### 3.1 `profile.py`

| Reference                                                             | Type           | Convention   | Notes                                                  |
| --------------------------------------------------------------------- | -------------- | ------------ | ------------------------------------------------------ |
| Example “**x** = np.array([[1,1,1,2,2,2]])”                           | variable       | **agnostic** | `x` is 1D profile data, not coords.                    |
| `src`, `dst` (row, col); `src_row`, `src_col`, `line_row`, `line_col` | docstring/impl | **ij**       | No x,y for image coords; profile_line uses (row, col). |

### 3.2 `_line_profile_coordinates`

| Reference                             | Type      | Convention | Notes                         |
| ------------------------------------- | --------- | ---------- | ----------------------------- |
| “**row** values (axis=1) are flipped” | docstring | **ij**     | Axis 0/1 and row/col; no x,y. |

### 3.3 `pnpoly.py` / `_pnpoly.pyx`

| Reference                                                                      | Type      | Convention   | Notes                                                |
| ------------------------------------------------------------------------------ | --------- | ------------ | ---------------------------------------------------- |
| “Input points, `(**x**, **y**)`”                                               | docstring | **agnostic** | Generic 2D points; `points[:, 0]` → x, `[:, 1]` → y. |
| `x = points[:, 0]`, `y = points[:, 1]`; `points_in_polygon(vx, vy, x, y, out)` | variables | **agnostic** | Same.                                                |

### 3.4 `fit.py`

| Reference                                                                               | Type         | Convention   | Notes                                                                |
| --------------------------------------------------------------------------------------- | ------------ | ------------ | -------------------------------------------------------------------- |
| `predict_x(self, y, ...)`, `predict_y(self, x, ...)`                                    | method names | **agnostic** | 2D line: predict x given y or y given x; axis=0 vs 1.                |
| “**x**-coordinates” / “**y**-coordinates”; “r² = (**x** − **xc**)² + (**y** − **yc**)²” | docstring    | **agnostic** | Circle/ellipse model; (x,y) generic 2D.                              |
| `xc`, `yc`, `predict_xy`, `(x, y)` in residuals                                         | variables    | **agnostic** | Model params; when fit to image coords, user chooses interpretation. |

### 3.5 `_marching_cubes_lewiner_cy.pyx` / `_marching_cubes_lewiner.py`

| Reference                                           | Type      | Convention | Notes                                                    |
| --------------------------------------------------- | --------- | ---------- | -------------------------------------------------------- |
| `add_vertex(…, x, y, …)`, `set_cube(…, x, y, z, …)` | params    | **ij**     | (x,y,z) map to array indices; order matches volume axes. |
| `self.x`, `self.y`; “step in **x**” / “**y**”       | variables | **ij**     | Same.                                                    |

### 3.6 `_ccomp.pyx`

| Reference                                                                     | Type      | Convention | Notes                                                       |
| ----------------------------------------------------------------------------- | --------- | ---------- | ----------------------------------------------------------- |
| `res.x`, `res.y` (shapeinfo); “**x**=0, **y**=-1”; `ravel_index*(x, y, z, …)` | variables | **ij**     | Array shape / index layout; x,y,z as axis sizes or indices. |
| “**y** V” (comment)                                                           | docstring | **ij**     | Axis convention.                                            |

### 3.7 `_find_contours_cy.pyx`

| Reference        | Type     | Convention   | Notes                      |
| ---------------- | -------- | ------------ | -------------------------- |
| `npy_isnan(… x)` | function | **agnostic** | Generic float, not coords. |

---

## 4. `skimage.segmentation`

### 4.1 `active_contour_model.py`

| Reference                                                               | Type      | Convention | Notes                                                                                                 |
| ----------------------------------------------------------------------- | --------- | ---------- | ----------------------------------------------------------------------------------------------------- |
| `snake_xy = snake[:, ::-1]`; `x = snake_xy[:, 0]`, `y = snake_xy[:, 1]` | variables | **xy**     | Snake stored (row,col); flipped to (x,y)=(col,row) for spline. `RectBivariateSpline` uses (col, row). |
| `fx`, `fy`, `dx`, `dy`, `xn`, `yn`; `return np.stack([y, x], axis=1)`   | variables | **xy**     | Same; output converted back to (row, col).                                                            |

### 4.2 `_watershed.py`

| Reference                                                                                                        | Type      | Convention   | Notes                                                          |
| ---------------------------------------------------------------------------------------------------------------- | --------- | ------------ | -------------------------------------------------------------- |
| “**x**, **y** = np.indices((80, 80))”; “**x1**, **y1**, **x2**, **y2**”; “(**x** − **x1**)² + (**y** − **y1**)²” | docstring | **agnostic** | `np.indices` → (row, col) order; example uses x,y as abstract. |

### 4.3 `_quickshift.py` / `_quickshift_cy.pyx`

| Reference                   | Type      | Convention   | Notes                                                      |
| --------------------------- | --------- | ------------ | ---------------------------------------------------------- |
| “Color-(**x**,**y**) space” | docstring | **agnostic** | Feature space (spatial + color); not image row/col per se. |

### 4.4 `slic_superpixels.py`

| Reference                         | Type      | Convention   | Notes |
| --------------------------------- | --------- | ------------ | ----- |
| “Color-(**x**,**y**,**z**) space” | docstring | **agnostic** | Same. |

### 4.5 `_slic.pyx`

| Reference                                                                                                              | Type      | Convention | Notes                                               |
| ---------------------------------------------------------------------------------------------------------------------- | --------- | ---------- | --------------------------------------------------- |
| “distances along **z**, **y**, and **x**”; “(**z**, **y**, **x**) order” vs “(**x**, **y**, **z**)”; “**x** and **y**” | docstring | **ij**     | Array order (plane, row, col); x,y,z as axis names. |
| `cx`, `cy`, `cz`, `dx`, `dy`, `dz`; `x`, `y`, `z` loops; `image_zyx[z, y, x, ...]`                                     | variables | **ij**     | (z,y,x) = (plane, row, col).                        |

### 4.6 `random_walker_segmentation.py`

| Reference                                    | Type      | Convention   | Notes                             |
| -------------------------------------------- | --------- | ------------ | --------------------------------- |
| “grid … **x** direction” / “**y** direction” | docstring | **agnostic** | Grid layout.                      |
| “**x**.T L **x**”; “**x** = 1 … **x** = 0”   | docstring | **agnostic** | Linear algebra, not image coords. |

### 4.7 `_chan_vese.py`

| Reference                          | Type      | Convention   | Notes                        |
| ---------------------------------- | --------- | ------------ | ---------------------------- |
| “sin(**x**/5*pi)*sin(**y**/5\*pi)” | docstring | **agnostic** | Synthetic init; x,y as grid. |

---

## 5. `skimage.filters`

### 5.1 `rank` (`shift_x`, `shift_y`)

| Reference                                                                                 | Type            | Convention | Notes                                                                                  |
| ----------------------------------------------------------------------------------------- | --------------- | ---------- | -------------------------------------------------------------------------------------- |
| `shift_x`, `shift_y` (and `shift_z`) in `generic.py`, `generic_cy`, `percentile_cy`, etc. | function params | **xy**     | shift_x → column, shift_y → row (see `summary_rc` / `COORDINATE_CONVENTIONS_SUMMARY`). |

### 5.2 `thresholding.py`

| Reference                                                              | Type      | Convention   | Notes                             |
| ---------------------------------------------------------------------- | --------- | ------------ | --------------------------------- |
| “m(**x**,**y**)”, “s(**x**,**y**)”; “pixel (**x**,**y**) neighborhood” | docstring | **agnostic** | Generic (x,y) for pixel location. |

---

## 6. `skimage.morphology`

### 6.1 `footprints.py`

| Reference                                  | Type      | Convention   | Notes                      |
| ------------------------------------------ | --------- | ------------ | -------------------------- |
| “(**x**/width+1)² + (**y**/height+1)² = 1” | docstring | **agnostic** | Ellipse eqn; local coords. |

### 6.2 `grayreconstruct.py`

| Reference                                                                        | Type      | Convention   | Notes                                                 |
| -------------------------------------------------------------------------------- | --------- | ------------ | ----------------------------------------------------- |
| “**x** = np.linspace…”, “**y**\_mask”, “**y**\_seed”; “**y**, **x** = np.mgrid…” | docstring | **agnostic** | 1D signals and 2D grid; not image row/col convention. |

### 6.3 `extrema.py`, `max_tree.py`

| Reference                                                              | Type      | Convention   | Notes                           |
| ---------------------------------------------------------------------- | --------- | ------------ | ------------------------------- |
| “**x**, **y** = np.mgrid[0:w, 0:w]”; “(**x** − w/2)² + (**y** − w/2)²” | docstring | **agnostic** | Synthetic 2D; mgrid axis order. |

### 6.4 `convex_hull.py`

| Reference         | Type      | Convention   | Notes                |
| ----------------- | --------- | ------------ | -------------------- |
| “[ ]–[**x**]–[ ]” | docstring | **agnostic** | Diagram placeholder. |

### 6.5 `_util.py`

| Reference                         | Type     | Convention   | Notes             |
| --------------------------------- | -------- | ------------ | ----------------- |
| `any(x < y for x, y in zip(...))` | variable | **agnostic** | Shape comparison. |

---

## 7. `skimage.restoration`

### 7.1 `_nl_means_denoising.pyx`

| Reference                                  | Type      | Convention | Notes   |
| ------------------------------------------ | --------- | ---------- | ------- |
| “**row** axis” / “**column** axis” (shift) | docstring | **ij**     | No x,y. |

### 7.2 `_rolling_ball_cy.pyx`

| Reference                              | Type      | Convention   | Notes               |
| -------------------------------------- | --------- | ------------ | ------------------- |
| “position `(**x**,**y**)`” (ellipsoid) | docstring | **agnostic** | Local patch coords. |

### 7.3 `_denoise.py`

| Reference                                          | Type      | Convention   | Notes    |
| -------------------------------------------------- | --------- | ------------ | -------- |
| “**x**, **y**, **z** = np.ogrid[0:20, 0:20, 0:20]” | docstring | **agnostic** | 3D grid. |

### 7.4 `_denoise_cy.pyx`

| Reference                    | Type      | Convention   | Notes                                |
| ---------------------------- | --------- | ------------ | ------------------------------------ |
| `dx`, `dy` (gradient arrays) | variables | **agnostic** | Derivative buffers; not spatial x,y. |

### 7.5 `_cycle_spin.py`

| Reference                                                                                | Type           | Convention   | Notes                    |
| ---------------------------------------------------------------------------------------- | -------------- | ------------ | ------------------------ |
| “shifted versions of **x**”; “**x** as its first argument”; “**xs** = np.roll(**x**, …)” | docstring/vars | **agnostic** | Input array, not coords. |

### 7.6 `deconvolution.py`

| Reference                                   | Type      | Convention   | Notes                                      |
| ------------------------------------------- | --------- | ------------ | ------------------------------------------ |
| “**y** = H**x** + n”; “**x**”; “\hat **x**” | docstring | **agnostic** | Linear algebra (signal, not image coords). |

### 7.7 `uft.py`

| Reference                 | Type      | Convention   | Notes             |
| ------------------------- | --------- | ------------ | ----------------- |
| “**y** : complex ndarray” | docstring | **agnostic** | Transform output. |

---

## 8. `skimage.registration`

### 8.1 `_optical_flow.py`

| Reference                         | Type      | Convention   | Notes        |
| --------------------------------- | --------- | ------------ | ------------ |
| “ndim **x** ndim” (linear system) | docstring | **agnostic** | Matrix size. |

### 8.2 `_masked_phase_cross_correlation.py`

| Reference         | Type     | Convention   | Notes      |
| ----------------- | -------- | ------------ | ---------- |
| `def ifft(**x**)` | function | **agnostic** | FFT input. |

---

## 9. `skimage.io`

### 9.1 `sift.py` (SIFT reader)

| Reference                                   | Type      | Convention | Notes                  |
| ------------------------------------------- | --------- | ---------- | ---------------------- |
| “**row**” / “**column**” (feature position) | docstring | **ij**     | No x,y in io; row/col. |

---

## 10. `skimage.graph`

### 10.1 `_rag.py`

| Reference                                                                              | Type           | Convention | Notes                                                               |
| -------------------------------------------------------------------------------------- | -------------- | ---------- | ------------------------------------------------------------------- |
| “(**x**,**y**)”; “matplotlib uses (**x**,**y**)”; `for **x**, **y**, d in graph.edges` | docstring/vars | **xy**     | Plotting (x,y) = (col, row); “tuple[::-1] … skimage (row, column)”. |

### 10.2 `_graph_cut.py`

| Reference                                 | Type     | Convention   | Notes                              |
| ----------------------------------------- | -------- | ------------ | ---------------------------------- |
| `(**x**, **y**) for x, y, d in rag.edges` | variable | **agnostic** | Edge node pairs; not pixel coords. |

### 10.3 `_graph.py`

| Reference                                | Type     | Convention   | Notes                                   |
| ---------------------------------------- | -------- | ------------ | --------------------------------------- |
| `edge_function(**x**, **y**, distances)` | function | **agnostic** | Edge weight; x,y = node IDs or similar. |

### 10.4 `_mcp.pyx`

| Reference                                                                       | Type      | Convention | Notes                           |
| ------------------------------------------------------------------------------- | --------- | ---------- | ------------------------------- |
| “(**x**, **y**)”; “predecessor of [**x**, **y**]”; “edge_map at (**x**, **y**)” | docstring | **ij**     | 2D index into cost/edge arrays. |

---

## 11. `skimage.future`

### 11.1 `manual_segmentation.py`

| Reference                                                                                | Type      | Convention | Notes                                               |
| ---------------------------------------------------------------------------------------- | --------- | ---------- | --------------------------------------------------- |
| `pr = [**y** for **x**, **y** in vertices]`, `pc = [**x** for **x**, **y** in vertices]` | variables | **xy**     | Vertices as (x,y) = (col, row); pr = row, pc = col. |

---

## 12. `skimage.color`

### 12.1 `colorconv.py`

| Reference                                                     | Type           | Convention   | Notes                                 |
| ------------------------------------------------------------- | -------------- | ------------ | ------------------------------------- |
| `xyz2rgb`, `xyz2lab`, `xyz2luv`                               | function names | **agnostic** | Color space X,Y,Z.                    |
| `**x**, **y**, **z** = arr[..., 0], arr[..., 1], arr[..., 2]` | variables      | **agnostic** | Channel split.                        |
| `_cart2polar_2pi(**x**, **y**)`                               | function       | **agnostic** | 2D Cartesian → polar; not image axes. |

---

## 13. `skimage.data`

### 13.1 `_fetchers.py`

| Reference                          | Type      | Convention | Notes             |
| ---------------------------------- | --------- | ---------- | ----------------- |
| “`(**z**, c, **y**, **x**)` order” | docstring | **ij**     | Array axis order. |

---

## 14. `skimage.draw`

### 14.1 `draw3d.py`

| Reference                           | Type      | Convention   | Notes    |
| ----------------------------------- | --------- | ------------ | -------- |
| “**x**, **y**, **z** = np.mgrid[…]” | docstring | **agnostic** | 3D grid. |

### 14.2 `_random_shapes.py`

| Reference                  | Type      | Convention   | Notes            |
| -------------------------- | --------- | ------------ | ---------------- |
| “bounding box coordinates” | docstring | **agnostic** | No explicit x,y. |

---

## 15. `skimage._shared`

### 15.1 `geometry.pyx`

| Reference                                                | Type             | Convention   | Notes                      |
| -------------------------------------------------------- | ---------------- | ------------ | -------------------------- |
| “**x**, **y** : np_floats”; `*_clipper(…, **x**, **y**)` | docstring/params | **agnostic** | Generic 2D polygon points. |

---

## 16. `skimage.metrics`

### 16.1 `simple_metrics.py`

| Reference                            | Type      | Convention   | Notes                       |
| ------------------------------------ | --------- | ------------ | --------------------------- |
| “**x** ∈ X”; “p(**x**)”; “**x** ∈ X” | docstring | **agnostic** | Entropy sum; generic value. |

### 16.2 `_variation_of_information.py`

| Reference                                     | Type     | Convention   | Notes                |
| --------------------------------------------- | -------- | ------------ | -------------------- |
| `_xlogx(**x**)`; “**y** : same type as **x**” | function | **agnostic** | x log x; not coords. |

---

## 17. `skimage.util`

### 17.1 `_rolling_ball.py`

| Reference                             | Type     | Convention   | Notes            |
| ------------------------------------- | -------- | ------------ | ---------------- |
| `np.arange(-**x**, **x**+1)` (radius) | variable | **agnostic** | Radius / extent. |

---

## Summary

| Convention   | Typical use                                                                                                                                                                                                                                     |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **xy**       | `warp`, `warp_coords`, geometric transforms, Hough (lines, circle peaks, ellipse), swirl/warp_polar, `_prominent_peaks` (xcoords/ycoords), active contour (x,y) internal, RAG plotting, manual_segmentation vertices, rank `shift_x`/`shift_y`. |
| **ij**       | `profile_line` (row,col only), draw/corner indexing, Hough Cython `nonzero` naming, `_haar` (y,x) integral, `_slic` (z,y,x), `_ccomp` shape, `_mcp` grid, `_marching_cubes` indices, data `(z,c,y,x)`.                                          |
| **agnostic** | Transform math (x,y generic 2D), Color-(x,y) / XYZ, fit models (predict_x/y, xc,yc), pnpoly, meshgrid/mgrid examples, cycle_spin/deconv uft, graph edge weight callbacks, metrics.                                                              |

**Inconsistencies:** Hough circle example uses `img[x,y]` with (x,y) from `circle_perimeter` swap, effectively (row,col) while API elsewhere uses (x,y)=(col,row). `_hough_transform` Cython uses `x,y = np.nonzero(img)` (row, col) but names suggest xy; internal line walking uses (x,y) as (col, row) for API. These are documented in `COORDINATE_CONVENTIONS_SUMMARY.md`.
