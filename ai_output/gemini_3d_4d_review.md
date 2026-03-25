# Review of 3D and N-D Image Support in `scikit-image`

This document summarizes functions, methods, and classes in `scikit-image` that support 3D images (3D arrays for grayscale, 4D arrays for color) and N-D images.

## 1. Explicit 3D Support
These routines are specifically designed for or have optimized implementations for 3D data.

### Draw
- `skimage.draw.draw3d.ellipsoid`: Generates a 3D binary ellipsoid.
- `skimage.draw.draw3d.ellipsoid_stats`: Calculates surface area and volume of an ellipsoid.

### Filters (Rank)
The following rank filters have explicit 3D core implementations:
- `autolevel`, `equalize`, `gradient`, `maximum`, `mean`, `geometric_mean`, `subtract_mean`, `median`, `minimum`, `modal`, `enhance_contrast`, `pop`, `sum`, `threshold`, `noise_filter`, `entropy`, `otsu`, `windowed_histogram`, `majority`.

### Measure
- `skimage.measure.marching_cubes`: Generates a triangle mesh from a 3D volume (requires 3D input).

### Morphology
- `skimage.morphology.skeletonize`: Uses Lee's algorithm for 3D thinning.

### Segmentation
- `skimage.segmentation.slic`: Supports 3D volumes (2D/3D).
- `skimage.segmentation.felzenszwalb`: Supports 3D volumes (2D/3D).
- `skimage.segmentation.morphological_chan_vese`: Supports 2D and 3D.
- `skimage.segmentation.morphological_geodesic_active_contour`: Supports 2D and 3D.

---

## 2. General N-D Support
These routines use dimension-agnostic logic (often leveraging `scipy.ndimage`) and support any number of dimensions.

### Filters
- `skimage.filters.gaussian`: Isotropic or anisotropic Gaussian smoothing.
- `skimage.filters.median`: Support for N-D when `behavior='ndimage'`.
- `skimage.filters.threshold_local`: N-D adaptive thresholding.
- `skimage.filters.edges.laplace`: N-D Laplacian filter.

### Feature
- `skimage.feature.match_template`: Supports 2D and 3D (mathematically N-D).

### Measure
- `skimage.measure.label`: Connected component labeling for N-D.
- `skimage.measure.regionprops` / `regionprops_table`: Calculates properties for N-D labeled objects.
- `skimage.measure.moments` / `moments_central` / `moments_normalized`: N-D image moments.
- `skimage.measure.inertia_tensor` / `inertia_tensor_eigvals`: Derived from N-D moments.

### Registration
- `skimage.registration.phase_cross_correlation`: Sub-pixel registration for N-D.
- `skimage.registration.optical_flow_tvl1` / `optical_flow_ilk`: N-D optical flow estimation.

### Restoration
- `skimage.restoration.denoise_tv_chambolle`: Total variation denoising in N-D.
- `skimage.restoration.denoise_nl_means`: Non-local means denoising for N-D.
- `skimage.restoration.denoise_wavelet`: Wavelet denoising for N-D.
- `skimage.restoration.richardson_lucy`: Deconvolution for N-D.

### Segmentation
- `skimage.segmentation.watershed`: N-D immersion watershed.
- `skimage.segmentation.random_walker`: N-D segmentation from seeds.

### Exposure
- `skimage.exposure.rescale_intensity`: N-D intensity scaling.
- `skimage.exposure.equalize_hist`: N-D histogram equalization.
- `skimage.exposure.equalize_adapthist`: N-D CLAHE.

### Transform
- `skimage.transform.warp` / `warp_coords`: N-D image warping.
- `skimage.transform.resize` / `rescale`: N-D image resizing.

### Utilities
- `skimage.util.img_as_float`, `img_as_ubyte`, etc.: Conversion for N-D.
- `skimage.util.invert`: Bitwise inversion for N-D.
- `skimage.util.view_as_blocks` / `view_as_windows`: N-D array slicing views.
- `skimage.util.regular_grid` / `regular_seeds`: N-D grid sampling.

### Metrics
- `skimage.metrics.mean_squared_error` / `peak_signal_noise_ratio`: N-D point-wise metrics.
- `skimage.metrics.structural_similarity`: Supports N-D comparison.

### Morphology
- `skimage.morphology.binary_erosion`, `dilation`, `opening`, `closing`: N-D binary morphology.
- `skimage.morphology.erosion`, `dilation`, `opening`, `closing`: N-D grayscale morphology.
- `skimage.morphology.remove_small_objects`, `remove_small_holes`: N-D object filtering.
- `skimage.morphology.footprint_rectangle`: Generates N-D hyper-rectangular footprints.

---

## 3. Color Channel Handling (4D Support)
Most color conversion functions in `skimage.color` (e.g., `rgb2gray`, `rgb2hsv`, `rgba2rgb`) support N-D arrays by using the `channel_axis` parameter. This allows them to process 3D color images (4D arrays) where one axis is the color channel.
