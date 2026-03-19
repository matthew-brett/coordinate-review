# Instances of M, N, ... in codebase

These are outputs from `rg "\(M\s*,\s*N"`.

For every checked instance recorded here, assume that `M, N ...` can be safely replaced with `(I, J ...)`.

## Checked

* `future/manual_segmentation.py:    image : (M, N[, 3]) array`
* `future/manual_segmentation.py:    image : (M, N[, 3]) array`
  Zulip reply - future should not be ported.
* `io/_plugins/matplotlib_plugin.py:    image : array, shape (M, N[, 3])`
  Zulip reply - not porting.

## Not yet checked

* `measure/_find_contours.py:    image : ndarray of shape (M, N) and dtype float`
* `measure/_find_contours.py:    mask : ndarray of shape (M, N) and dtype bool`
* `measure/_marching_cubes_lewiner.py:        indexing dimensions (M, N, P) as in `volume`.`
* `measure/_marching_cubes_lewiner.py:        matches input `volume` (M, N, P). If ``allow_degenerate`` is set to`
* `measure/_marching_cubes_lewiner.py:    mask : (M, N, P) array, optional`
* `measure/_marching_cubes_lewiner.py:    volume : (M, N, P) ndarray`
* `measure/_pnpoly.pyx:         np.zeros((M, N), dtype=np.uint8)`
* `measure/_pnpoly.pyx:    mask : (M, N) ndarray of bool`
* `measure/_pnpoly.pyx:    shape : tuple (M, N)`
* `measure/_regionprops.py:    intensity_image : (M, N[, P][, C]) ndarray, optional`
* `measure/_regionprops.py:    intensity_image : (M, N[, P][, C]) ndarray, optional`
* `measure/_regionprops.py:    label_image : (M, N[, P]) ndarray`
* `measure/_regionprops.py:    label_image : (M, N[, P]) ndarray`
* `measure/_regionprops_utils.py:    image : (M, N[, P]) ndarray`
* `measure/_regionprops_utils.py:    image : ndarray of shape (M, N)`
* `measure/_regionprops_utils.py:    image : ndarray of shape (M, N)`
* `measure/entropy.py:    image : ndarray of shape (M, N)`
* `measure/pnpoly.py:    mask : ndarray of shape (M, N)`
* `measure/pnpoly.py:    shape : tuple (M, N)`
* `measure/profile.py:    image : ndarray, shape (M, N[, C])`
* `morphology/_skeletonize.py:    image : (M, N[, P]) ndarray of bool or int`
* `morphology/_skeletonize.py:    image : binary (M, N) ndarray`
* `morphology/_skeletonize.py:    image : binary ndarray, shape (M, N)`
* `morphology/_skeletonize.py:    mask : binary ndarray, shape (M, N), optional`
* `morphology/_skeletonize.py:    skeleton : (M, N[, P]) ndarray of bool`
* `morphology/convex_hull.py:    gridcoords : ndarray of shape (M, N)`
* `morphology/convex_hull.py:    hull : (M, N) array of bool`
* `morphology/convex_hull.py:    hull_equations : ndarray of shape (M, N)`
* `morphology/convex_hull.py:    image : ndarray of shape (M, N)`
* `restoration/deconvolution.py:    image : ndarray of shape (M, N)`
* `restoration/deconvolution.py:    x_postmean : ndarray of shape (M, N)`
* `util/compare.py:    comparison : ndarray, shape (M, N)`
* `util/compare.py:    image0, image1 : ndarray, shape (M, N)`
* `util/unique.py:    ar : ndarray, shape (M, N)`
