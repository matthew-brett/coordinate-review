# Instances of plane in codebase

These are instances (via `ripgrep`) of `plane` in code-base, reviewed and not
relevant, and not yet reviewed.

## Checked, not relevant

* `src/skimage/graph/spath.py:    # Valid starting positions are
  anywhere on the hyperplane defined by`
* `src/skimage/graph/spath.py:    # hyperplane at position -1 along the same.`

From `shortest_path` function.  Function is clearly N-D, but asks the
user to specify (`axis=-1`) the "axis along which the path must always
move forward".

## Not checked

* `src/skimage/graph/_mcp.pyx:    """Return an array with edge points/lines/planes/hyperplanes marked.`
* `src/skimage/segmentation/_slic.pyx:    a cut-plane through the volume. So, if the order was (x, y, z) and`
* `src/skimage/segmentation/_slic.pyx:    we wanted to look at the 5th cut plane, we would write::`
* `src/skimage/segmentation/_slic.pyx:        my_z_plane = img3d[:, :, 5]`
* `src/skimage/segmentation/_slic.pyx:        my_z_plane = img3d[5]`
* `src/skimage/restoration/_denoise.py:    plane separately.`
* `src/skimage/restoration/_nl_means_denoising.pyx:    # Iterate over planes, taking padding into account`
* `src/skimage/restoration/_nl_means_denoising.pyx:        Shift along the plane axis.`
* `src/skimage/restoration/_nl_means_denoising.pyx:        Shift along the plane axis.`
* `src/skimage/restoration/_nl_means_denoising.pyx:        # Iterate over shifts along the plane axis`
* `src/skimage/restoration/_nl_means_denoising.pyx:                    # Iterate over planes, taking offset and shift into account`
* `src/skimage/restoration/_nl_means_denoising.pyx:    # Iterate over shifts along the plane axis`
* `src/skimage/restoration/_nl_means_denoising.pyx:                        # Iterate over planes, taking offset and shift into account`
