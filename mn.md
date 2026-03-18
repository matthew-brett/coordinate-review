# Instances of M, N, ... in codebase

These are outputs from `rg "\(M\s*,\s*N"`.

For every checked instance recorded here, assume that `M, N ...` can be safely replaced with `(I, J ...)`.

## Checked

* `src/skimage/future/manual_segmentation.py:    image : (M, N[, 3]) array`
* `src/skimage/future/manual_segmentation.py:    image : (M, N[, 3]) array`
  Zulip reply - future should not be ported.
* `src/skimage/io/_plugins/matplotlib_plugin.py:    image : array, shape (M, N[, 3])`
  Zulip reply - not porting.

## Not yet checked

* `src/skimage/transform/_radon_transform.pyx:    image : ndarray of float, shape (M, N)`
* `src/skimage/transform/_radon_transform.pyx:    image : ndarray of float, shape (M, N)`
* `src/skimage/transform/_radon_transform.pyx:    image_update : ndarray of float, shape (M, N)`
* `src/skimage/transform/_radon_transform.pyx:    image : ndarray of float, shape (M, N)`
* `src/skimage/transform/_radon_transform.pyx:    image_update : ndarray of float, shape (M, N)`
* `src/skimage/filters/_gabor.py:    image : ndarray of shape (M, N)`
* `src/skimage/filters/_gabor.py:    real, imag : ndarray of shape (M, N)`
* `src/skimage/segmentation/_felzenszwalb_cy.pyx:    image : (M, N, C) ndarray`
* `src/skimage/segmentation/_felzenszwalb_cy.pyx:    segment_mask : (M, N) ndarray`
* `src/skimage/transform/_geometric.py:    src : (M, N) array_like`
* `src/skimage/transform/_geometric.py:    dst : (M, N) array_like`
* `src/skimage/transform/_warps_cy.pyx:    image : ndarray, shape (M, N)`
* `src/skimage/segmentation/_felzenszwalb.py:    image : ndarray of shape (M, N[, 3])`
* `src/skimage/segmentation/_felzenszwalb.py:    segment_mask : ndarray of shape (M, N)`
* `src/skimage/segmentation/random_walker_segmentation.py:    data : (M, N[, P][, C]) ndarray`
* `src/skimage/segmentation/random_walker_segmentation.py:    labels : (M, N[, P]) array of ints`
* `src/skimage/segmentation/boundaries.py:    image : ndarray of shape (M, N[, 3])`
* `src/skimage/segmentation/boundaries.py:    label_img : ndarray of shape (M, N) and dtype int`
* `src/skimage/segmentation/boundaries.py:    marked : ndarray of shape (M, N, 3) and dtype float`
* `src/skimage/util/compare.py:    image0, image1 : ndarray, shape (M, N)`
* `src/skimage/util/compare.py:    comparison : ndarray, shape (M, N)`
* `src/skimage/util/unique.py:    ar : ndarray, shape (M, N)`
* `src/skimage/restoration/_denoise.py:    image : ndarray, shape (M, N[, 3])`
* `src/skimage/restoration/deconvolution.py:    image : ndarray of shape (M, N)`
* `src/skimage/restoration/deconvolution.py:    x_postmean : ndarray of shape (M, N)`
* `src/skimage/registration/_optical_flow.py:    reference_image : ndarray, shape (M, N[, P[, ...]])`
* `src/skimage/registration/_optical_flow.py:    moving_image : ndarray, shape (M, N[, P[, ...]])`
* `src/skimage/registration/_optical_flow.py:    reference_image : ndarray, shape (M, N[, P[, ...]])`
* `src/skimage/registration/_optical_flow.py:    moving_image : ndarray, shape (M, N[, P[, ...]])`
* `src/skimage/registration/_optical_flow.py:    reference_image : ndarray, shape (M, N[, P[, ...]])`
* `src/skimage/registration/_optical_flow.py:    moving_image : ndarray, shape (M, N[, P[, ...]])`
* `src/skimage/registration/_optical_flow.py:    reference_image : ndarray, shape (M, N[, P[, ...]])`
* `src/skimage/registration/_optical_flow.py:    moving_image : ndarray, shape (M, N[, P[, ...]])`
* `src/skimage/feature/_haar.pyx:    int_image : (M, N) ndarray`
* `src/skimage/feature/_daisy.py:    image : (M, N) array`
* `src/skimage/feature/_daisy.py:    descs_img : ndarray of shape (M, N, 3), only if visualize=True`
* `src/skimage/feature/template.py:    image : (M, N[, P]) array`
* `src/skimage/feature/template.py:        is an array with shape `(M - m + 1, N - n + 1)` for an `(M, N)` image`
* `src/skimage/feature/texture.py:    image : (M, N) array`
* `src/skimage/feature/texture.py:    output : (M, N) array`
* `src/skimage/feature/peak.py:    image : (M, N) ndarray`
* `src/skimage/feature/_hog.py:    channel : ndarray of shape (M, N)`
* `src/skimage/feature/_hog.py:    image : (M, N[, C]) ndarray`
* `src/skimage/feature/_hog.py:    hog_image : (M, N) ndarray, optional`
* `src/skimage/feature/haar.py:    int_image : ndarray of shape (M, N)`
* `src/skimage/feature/haar.py:    image : ndarray of shape (M, N)`
* `src/skimage/feature/haar.py:    features : ndarray of shape (M, N)`
* `src/skimage/morphology/_skeletonize.py:    image : (M, N[, P]) ndarray of bool or int`
* `src/skimage/morphology/_skeletonize.py:    skeleton : (M, N[, P]) ndarray of bool`
* `src/skimage/morphology/_skeletonize.py:    image : binary (M, N) ndarray`
* `src/skimage/morphology/_skeletonize.py:    image : binary ndarray, shape (M, N)`
* `src/skimage/morphology/_skeletonize.py:    mask : binary ndarray, shape (M, N), optional`
* `src/skimage/morphology/convex_hull.py:    gridcoords : ndarray of shape (M, N)`
* `src/skimage/morphology/convex_hull.py:    hull_equations : ndarray of shape (M, N)`
* `src/skimage/morphology/convex_hull.py:    hull : (M, N) array of bool`
* `src/skimage/morphology/convex_hull.py:    image : ndarray of shape (M, N)`
* `CONTRIBUTING.rst:* When documenting array parameters, use ``image : (M, N) ndarray```
* `src/skimage/measure/entropy.py:    image : ndarray of shape (M, N)`
* `src/skimage/measure/_marching_cubes_lewiner.py:    volume : (M, N, P) ndarray`
* `src/skimage/measure/_marching_cubes_lewiner.py:        indexing dimensions (M, N, P) as in `volume`.`
* `src/skimage/measure/_marching_cubes_lewiner.py:    mask : (M, N, P) array, optional`
* `src/skimage/measure/_marching_cubes_lewiner.py:        matches input `volume` (M, N, P). If ``allow_degenerate`` is set to`
* `src/skimage/measure/_pnpoly.pyx:    shape : tuple (M, N)`
* `src/skimage/measure/_pnpoly.pyx:    mask : (M, N) ndarray of bool`
* `src/skimage/measure/_pnpoly.pyx:         np.zeros((M, N), dtype=np.uint8)`
* `src/skimage/measure/_regionprops_utils.py:    image : (M, N[, P]) ndarray`
* `src/skimage/measure/_regionprops_utils.py:    image : ndarray of shape (M, N)`
* `src/skimage/measure/_regionprops_utils.py:    image : ndarray of shape (M, N)`
* `src/skimage/measure/profile.py:    image : ndarray, shape (M, N[, C])`
* `src/skimage/measure/_find_contours.py:    image : ndarray of shape (M, N) and dtype float`
* `src/skimage/measure/_find_contours.py:    mask : ndarray of shape (M, N) and dtype bool`
* `src/skimage/measure/pnpoly.py:    shape : tuple (M, N)`
* `src/skimage/measure/pnpoly.py:    mask : ndarray of shape (M, N)`
* `src/skimage/measure/_regionprops.py:    label_image : (M, N[, P]) ndarray`
* `src/skimage/measure/_regionprops.py:    intensity_image : (M, N[, P][, C]) ndarray, optional`
* `src/skimage/measure/_regionprops.py:    label_image : (M, N[, P]) ndarray`
* `src/skimage/measure/_regionprops.py:    intensity_image : (M, N[, P][, C]) ndarray, optional`
