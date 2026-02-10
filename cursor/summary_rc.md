# Row/Column References in the scikit-image Codebase

This report lists **docstring** and **variable** references to "row" and "column" (or `r`/`c`-style names) **in image contexts** across the codebase. Non-image uses (e.g. DataFrame columns, matrix rows in linear algebra, graph nodes) are excluded.

**Scope:** `src/skimage` (and selected `doc`, `tests`). Image = 2D/3D arrays indexed as `[row, col]` or `[plane, row, col]`, and related APIs.

---

## 1. `skimage.draw`

### 1.1 `draw.py`

**Docstrings:**

| Location              | Reference                                    | Context                          |
| --------------------- | -------------------------------------------- | -------------------------------- |
| `_ellipse_in_shape`   | `(row, column) position of center`           | center inside shape              |
|                       | `Size of two half axes (for row and column)` | radii                            |
|                       | `rows` / `cols`                              | return row/col coordinates       |
| `ellipse`             | `r, c` / `r_radius, c_radius`                | center and semi-axes (row, col)  |
|                       | `rr, cc`                                     | pixel coords; `img[rr, cc]`      |
| `disk`                | `rr, cc`                                     | pixel coords; `img[rr, cc]`      |
| `polygon_perimeter`   | `rr, cc`                                     | indices; `img[rr, cc]`           |
| `line`                | `r0, c0` / `r1, c1`                          | start/end `(row, column)`        |
|                       | `rr, cc`                                     | indices; `img[rr, cc]`           |
| `line_aa`             | `r0, c0` / `r1, c1`                          | start/end `(row, column)`        |
|                       | `rr, cc`                                     | indices; `img[rr, cc]`           |
| `polygon`             | `r` / `c`                                    | row/column coords of vertices    |
|                       | `rr, cc`                                     | pixel coords; `img[rr, cc]`      |
| `circle_perimeter`    | `r, c`                                       | centre (row, col); `img[rr, cc]` |
| `circle_perimeter_aa` | `r, c`                                       | centre of circle                 |
|                       | `rr, cc`                                     | indices; `img[rr, cc]`           |
| `ellipse_perimeter`   | `r, c` / `r_radius, c_radius`                | centre and semi-axes             |
|                       | `rr, cc`                                     | indices; `img[rr, cc]`           |
| `bezier_curve`        | `r0, c0` … `r2, c2`                          | control points (row, col)        |
|                       | `rr, cc`                                     | indices; `img[rr, cc]`           |
| `rectangle`           | `rr, cc`                                     | indices; `img[rr, cc]`           |

**Variables:** `r`, `c`, `r0`, `c0`, `r1`, `c1`, `r2`, `c2`, `rr`, `cc`, `r_lim`, `c_lim`, `r_org`, `c_org`, `r_rad`, `c_rad`, `line_r`, `line_c`; `upper_left[0]`/`[1]` used as row/col offsets. All used for image indexing `img[rr, cc]` or `img[r, c]`.

### 1.2 `_draw.pyx`

**Docstrings:** `_coords_inside_image`: "`rr`, `cc`"; "coordinates `[rr, cc]`"; "`rr`, `cc`". `_line` / `_line_aa`: "**(row, column)**" start/end; "`rr`, `cc`"; "`img[rr, cc] = 1`".

**Variables:** `rr`, `cc` (coords, shape checks); `r`, `c` (current position); `r0`, `c0`, `r1`, `c1`; `dr`, `dc`; `sr`, `sc`. Bresenham loop updates `r`, `c` and fills `rr[i]`, `cc[i]` (with swap when stepping along columns). All for `img[rr, cc]`.

---

## 2. `skimage.transform`

### 2.1 `_warps.py`

**Docstrings:**

| Location                | Reference                                                 | Context                                                          |
| ----------------------- | --------------------------------------------------------- | ---------------------------------------------------------------- |
| `warp_coords`           | `(rows, cols[, ...][, dim])`                              | output image size                                                |
| `warp`                  | `(rows, cols[, ...])`                                     | output size                                                      |
| `rotate`                | `center=(cols / 2 - 0.5, rows / 2 - 0.5)`; `(cols, rows)` | center is (col, row)                                             |
|                         | `corners`                                                 | `[[0,0], [0, rows-1], [cols-1, rows-1], [cols-1, 0]]` (col, row) |
| `warp_coords`           | `(row, col)` pairs                                        | coord_map; also `(2, rows, cols)`                                |
|                         | `(col, row)` in output / input                            | coord_map convention                                             |
| `_warp_fast` / `rotate` | `center : tuple (row, col)` vs `(col, row)`               | doc inconsistency                                                |
|                         | `output_shape : tuple (row, col)`                         |                                                                  |

**Variables:** `rows`, `cols` from `image.shape[0]`, `image.shape[1]`; `rr`, `cc` in `swirl` / `warp_polar` (e.g. `rr = ... + center[0]`, `cc = ... + center[1]`); `center[0]`/`[1]` as row/col; `minr`, `maxr`, `minc`, `maxc`, `out_rows`, `out_cols` for crop/resize.

### 2.2 `_warps_cy.pyx`

**Docstrings:** `output_shape : tuple (rows, cols)`.

**Variables:** `rows`, `cols` = `img.shape[0]`, `img.shape[1]`; `r`, `c` (floats) as inverse-mapped source row/col; loop over output row/col, transform receives `(tfc, tfr)` as (x,y).

### 2.3 `integral.py`

**Docstrings:**

| Location    | Reference                                      | Context               |
| ----------- | ---------------------------------------------- | --------------------- |
| `integrate` | `starting row, col, ...` / `end row, col, ...` | `start` / `end` lists |

**Variables:** `rows` = `start.shape[0]`; `start[r]`, `end[r]`; indexing uses (row, col) in tuples.

### 2.4 `radon_transform.py`

**Docstrings:** "Each **column** of" (sinogram); "Number of **rows** and **columns** in the reconstruction"; "i'th **column** of `radon_image`".

**Variables:** `col` in loop over `radon_filtered.T` (sinogram columns).

### 2.5 `finite_radon_transform.py`

**Docstrings:** "**row** x n **column**"; "**row** of ai"; "**row**s of ai".

**Variables:** `row` in loops over array rows.

### 2.6 `hough_transform.py`

**Docstrings:** Examples use `rr`, `cc` from `line` / `circle_perimeter` / `ellipse_perimeter`; `img[rr, cc]` or `img[cc, rr]`; `r, c` from `unravel_index` (center).

### 2.7 `_hough_transform.pyx`

**Docstrings:** `(yc, xc)` center; "**yc**, **xc**" in result dtype.

**Variables:** `xc`, `yc` (center), `p1x`, `p2x`, `p3x`, `p1y`, etc.; `yc`, `xc` in structured result.

### 2.8 `_geometric.py`

**Docstrings:** Mostly transform coefficients (e.g. `c0`, `c1` in projective); "**rows**" of `src` (point arrays). Image-specific row/column are in callers (e.g. `warp`).

---

## 3. `skimage.feature`

### 3.1 `util.py`

**Docstrings:** Keypoint locations "`(row, col)`"; "First keypoint … `(row, col)`"; image shape "`(rows, cols)`"; keypoints "`(rows, cols)`".

**Variables:** `rows = image_shape[0]`; `keypoints[:, 0]` < rows, `keypoints[:, 1]` cols.

### 3.2 `corner.py`

**Docstrings:** "Corner coordinates `(row, col)`"; "**(row, column**, ...) coordinates of peaks"; "`(row, col)`".

**Variables:** `row`, `col` in `for idx, (row, col) in enumerate(...)`; `symmetric_image[..., row, col]`, `[..., col, row]`.

### 3.3 `corner_cy.pyx`

**Variables:** `r`, `c`, `r0`, `c0`; `corners[i, 0]` (row), `corners[i, 1]` (col); `cimage[r0+r, c0+c]`.

### 3.4 `peak.py`

**Docstrings:** "**x** and **y** indices"; "**xcoords**", "**ycoords**"; "**x** dimension" / "**y** dimension".

**Variables:** `rows`, `cols = img.shape`; `ycoords_size`, `xcoords_size`; `ycoords_ext`, `xcoords_ext`; `ycoords_nh`, `xcoords_nh`; `ycoords_peaks`, `xcoords_peaks`; `ycoords_idx`, `xcoords_idx`. Used with `img[ycoords_idx, xcoords_idx]` (y=row, x=col).

### 3.5 `sift.py`

**Docstrings:** "Keypoint coordinates as `(row, col)`"; "Subpixel … `(row, col)`".

**Variables:** `row`, `col` in `_rotate(row, col, angle)`; `rot_row`, `rot_col`; comment "0 = row, 1 = col".

### 3.6 `orb.py`

**Docstrings:** "Keypoint coordinates as `(row, col)`".

### 3.7 `brief.py`

**Docstrings:** "Keypoint coordinates as `(row, col)`".

### 3.8 `censure.py`

**Docstrings:** "Keypoint coordinates as `(row, col)`".

**Variables:** `rows`, `cols`, `scales = np.nonzero(...)`; `keypoints = np.column_stack([rows, cols])` (row, col).

### 3.9 `blob.py`

**Docstrings:** "`(row, col, sigma)`" or "`(pln, row, col, sigma)`"; "**row**, **col**" as coordinates; "**row**s" of blobs array.

**Variables:** Various; `blob1[-1]` etc. for radius (not row/col). Coordinate layout in arrays is (row, col) or (pln, row, col).

### 3.10 `haar.py`

**Docstrings:** "**Row**-coordinate of top left …"; "**Column**-coordinate of top left …".

**Variables:** `rr`, `cc` from `rectangle(...)`; `output[rr, cc]`.

### 3.11 `texture.py`

**Docstrings:** "**Row**-coordinate of top left …"; "**Column**-coordinate of top left …".

### 3.12 `_hog.py`

**Variables:** `rr`, `cc` from `draw.line` or `np.meshgrid`; `hog_image[rr, cc]`; `r`, `c` in orientation histogram loops.

### 3.13 `_texture.pyx`

**Variables:** `rr`, `cc` from `R * cos/sin` (circle); `rp`, `cp` rounded.

---

## 4. `skimage.measure`

### 4.1 `_regionprops.py`

**Docstrings:** "Centroid … `(row, col)`"; "`(row, col)`, relative to region bounding"; "Coordinate list `(row, col)`"; "**row**, **col** coordinates of the region"; "**rows** \* **cols**"; "**row**-axis". "**column**" used for DataFrame/tensor layout too (excluded when not image axes).

**Variables:** Centroid and coords as (row, col) in docs.

### 4.2 `_find_contours.py`

**Docstrings:** "Each contour is … `(row, column)` coordinates".

### 4.3 `_find_contours_cy.pyx`

**Variables:** `r0`, `r1`, `c0`, `c1` for 2×2 cell; `mask[r0, c0]` etc.; `top = r0, c0+...`, `left = r0+..., c0`.

### 4.4 `_moments.py`

**Docstrings:** "`[[row, col] ...]`"; "**cr**, **cc**" (center row, col); "**rows** 1 and 3", "**columns** 1 and 3" (moment matrix).

**Variables:** `row`, `col` in examples; `cr`, `cc`.

### 4.5 `_polygon.py`

**Variables:** `r0`, `c0`, `r1`, `c1` from `coords[start,:]`, `coords[end,:]`; `dr`, `dc`; `segment_coords[:, 0]` (r), `[:, 1]` (c).

### 4.6 `pnpoly.py` / `_pnpoly.pyx`

**Docstrings:** "For each `(r, c)` coordinate on a grid".

### 4.7 `profile.py`

**Docstrings:** "**row** values (axis=1)"; "the full first **row**".

---

## 5. `skimage.filters`

### 5.1 `rank` — `generic.py`, `core_cy.pyx`, `core_cy_3d.pyx`, `percentile_cy.pyx`, `generic_cy.pyx`

**Docstrings:** `shift_x`, `shift_y` (and `shift_z` for 3D).

**Variables:**

| File             | Variables                       | Context                                                                                   |
| ---------------- | ------------------------------- | ----------------------------------------------------------------------------------------- | ---------- |
| `generic.py`     | `shift_x`, `shift_y`, `shift_z` | footprint shift params                                                                    |
|                  | `centre_r`, `centre_c`          | `centre_r = shape[0]//2 + shift_y`, `centre_c = shape[1]//2 + shift_x` (2D); 3D analagous |
| `core_cy.pyx`    | `centre_r`, `centre_c`          | footprint center (row, col); `shift_y` → row, `shift_x` → col                             |
|                  | `r`, `c`, `rr`, `cc`            | current pixel and neighbor offsets; `image[rr, cc]`, `image[r, c]`                        |
|                  | `se_e_r`, `se_e_c`, `se_w_r`, … | N/S/E/W row/col offsets                                                                   |
| `core_cy_3d.pyx` | `planes`, `rows`, `cols`        | 3D shape; `centre_r`, `centre_c` (also `centre_p`); `r`, `c`, `rr`, `cc`, `pp`            | 3D indices |

### 5.2 `thresholding.py`

**Docstrings:** "Number of **columns**" (image).

### 5.3 `edges.py`

**Docstrings:** "**row**" in Table (Canny ref); no image row/col variables.

### 5.4 `lpi_filter.py`

**Docstrings:** "`f(r, c, **filter_params)`"; "**row** and **column** positions"; "`impulse_response(r, c, ...)`".

**Variables:** `r`, `c` in callable signature (image row, col).

---

## 6. `skimage.morphology`

### 6.1 `footprints.py`

**Docstrings:** "Number of **rows**" / "**columns** of the rectangle"; "central **row** and **column**"; `rows, cols = draw.ellipse(...)`; `footprint[rows, cols]`.

**Variables:** `_cross(r0, r1)` — `r0`/`r1` as radii (not row/col). `rows`, `cols` from `draw.ellipse` for footprint indexing.

### 6.2 `_convex_hull.pyx`

**Docstrings:** "`(row, column)` coordinates"; "**rows** storage … **cols** storage"; "**row** or **column**".

**Variables:** `rows`, `cols` = `img.shape`; `r`, `c` in loops; `img[r, c]`; `coords` layout.

### 6.3 `_skeletonize_various_cy.pyx`

**Variables:** `row`, `col`, `nrows`, `ncols`; `skeleton[row, col]`; `rows`, `cols` from `result.shape`.

### 6.4 `convex_hull.py`

**Docstrings:** "**row** or **column**" (boundary pixels).

---

## 7. `skimage.restoration`

### 7.1 `_nl_means_denoising.pyx`

**Docstrings:** "**row** axis"; "**column** axis"; "Iterate over **rows**" / "**columns**".

**Variables:** `row`, `col`; `n_row`, `n_col`; `t_row`, `t_col` (shift); `integral[row, col]`, `result[row, col, channel]`; `pln`, `row`, `col` in 3D; `time`, `pln`, `row`, `col` in 4D.

### 7.2 `_denoise_cy.pyx`

**Variables:** `rows`, `cols` = `image.shape[0]`, `[1]`; `r`, `c` in loops; `rr`, `cc` neighbor indices; `image[r, c, d]`, `out[r, c, k]`; `dx`, `dy`, `bx`, `by` at `[r, c, k]`.

### 7.3 `_denoise.py`

**Variables:** `rows`, `cols`; `rr`, `cc` from `np.meshgrid(..., indexing='ij')` (distance grid).

### 7.4 `uft.py`

**Docstrings:** "**row** and **column**" (second-order difference).

---

## 8. `skimage.segmentation`

### 8.1 `morphsnakes.py`

**Docstrings:** "Coordinates of the center … (**row**, **column**)".

### 8.2 `boundaries.py`

**Docstrings:** "**row** and **column**" between actual rows/columns; "**rows** and **columns**".

### 8.3 `active_contour_model.py`

**Examples:** `rr`, `cc` from `circle_perimeter`; `img[rr, cc]`.

### 8.4 `_quickshift_cy.pyx`

**Variables:** `r`, `c`, `r_`, `c_`, `r_min`, `r_max`, `c_min`, `c_max`; `densities[r, c]`, `parent[r, c]`.

---

## 9. `skimage.io`

### 9.1 `sift.py`

**Docstrings:** "**row** position of feature"; "**column** position of feature"; dtype `('row', float)`, `('column', float)`.

**Variables:** `row`, `column` in dtypes and struct.

### 9.2 `_plugins/pil_plugin.py`

**Comments:** "R, G, B **columns**" (palette); not image row/col.

### 9.3 `_plugins/matplotlib_plugin.py`

**Variables:** `r1`, `c1`, `r2`, `c2`, `nrows`, `ncols` for layout of multi-image figures (grid layout, not image axes).

---

## 10. `skimage.util`

### 10.1 `compare.py`

**Docstrings:** "tiles (**row**, **column**) to divide the image".

### 10.2 `shape.py`

**Docstrings:** "**row** or **column**" (shift).

### 10.3 `unique.py`

**Docstrings:** "**rows**" of 2D array; "**columns**" in `ar`. Treated as array layout; image only if used for images.

### 10.4 `_map_array.py`

**Variables:** `rows = range(...)` for mapping display; not image row/col.

---

## 11. `skimage.graph`

### 11.1 `_rag.py`

**Comments:** "skimage uses (**row**, **column**)" (vs. other convention).

### 11.2 `_ncut_cy.pyx`

**Variables:** `row`, `col` as indices into graph weight matrix, not image pixels. Excluded.

### 11.3 `spath.py`

**Docstrings:** "one **row** up or down" (path step). Image-related.

---

## 12. `skimage.registration`

### 12.1 `_phase_cross_correlation.py`

**Comments:** "one **row** or **column**" (shift dimension). Image-related.

---

## 13. `skimage.future`

### 13.1 `manual_segmentation.py`

**Variables:** `pr`, `pc`; `rr`, `cc = polygon(pr, pc, shape)`; `mask[rr, cc]`.

---

## 14. `doc/examples` (selected)

| File                                  | Reference                                              | Context                |
| ------------------------------------- | ------------------------------------------------------ | ---------------------- |
| `plot_ssim.py`                        | `rows, cols = img.shape`                               | image shape            |
| `plot_pyramid.py`                     | `rows`, `cols`, `composite_rows`                       | pyramid layout         |
| `plot_piecewise_affine.py`            | `rows`, `cols`, `src_rows`                             | image shape and coords |
| `plot_regionprops.py`                 | "one region per **row**"                               | table layout           |
| `plot_masked_register_translation.py` | "offset (**row**, **col**)"                            | shift                  |
| `plot_phase_unwrap.py`                | "**row** of the image"                                 |                        |
| `plot_windowed_histogram.py`          | "**row**", "**column**"                                | grid layout            |
| `plot_3d_structure_tensor.py`         | "**row**", "**col**"; "**plane**, **row**, **column**" | 3D coords              |
| `plot_3d_image_processing.py`         | "**row**", "**column**"; "**[plane, row, column]**"    | array layout           |
| `plot_cornea_spot_inpainting.py`      | "(**row**, **column**) coordinates"                    | contours               |

---

## 15. `tests` (selected)

Row/column (and `r`/`c`) appear in transform, measure, feature, io, exposure tests — e.g. `(row, col)` coords, `rows`/`cols` from `shape`, `rr`/`cc` in draw-based tests. See `test_regionprops.py`, `test_geometric.py`, `test_hog.py`, `test_sift_reader.py`, etc.

---

## Summary

- **Draw:** Consistently uses `r`, `c`, `r0`, `c0`, `rr`, `cc` and docstrings "(row, column)" / "row, col".
- **Transform:** Uses `rows`, `cols`; `(row, col)` vs `(col, row)` in docs (`rotate` center, `warp_coords`); `r`, `c` in Cython warp loop.
- **Feature:** Keypoints and peaks as `(row, col)` or `(xcoords, ycoords)` with x=col, y=row; `row`/`col` in corners, blob, etc.
- **Measure:** Contours, regionprops, moments, polygon use `(row, col)` or `r0`, `c0`, `r1`, `c1`.
- **Filters (rank):** `shift_x` (col), `shift_y` (row); `centre_r`, `centre_c`; `r`, `c`, `rr`, `cc` in Cython.
- **Morphology:** `rows`, `cols`, `row`, `col` in convex hull, skeleton, footprints.
- **Restoration:** `row`, `col`, `n_row`, `n_col` in nl-means and denoise Cython.
- **Segmentation, io, util, graph, registration:** Various docstrings and variables as above; excluded when not image-related.

This list focuses on **image** row/column semantics; DataFrame "columns," Laplacian matrix "rows," or graph node indices are omitted.
