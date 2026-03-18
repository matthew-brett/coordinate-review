""" Preliminary warp refactor

Initial version by JNI from Vienna sprint.
"""

import numpy as np
from scipy import ndimage as ndi

import skimage as ski
from skimage.transform._geometric import (
    SimilarityTransform, AffineTransform, ProjectiveTransform
)


def _update_tform_for_skimage1_warp(inverse_map):
    """Fix skimage1 weirdnesses in coordinate handling.

    scikit-image 1 made a lot of terrible choices in how to handle
    coordinate transformations:

    - for 2D transforms, it swapped the order of the image axes
    - for 3D transforms, it didn't
    - in all cases, it considered pixels as little squares with coordinates
      at 0.5.

    In the initial implementation of skimage2.spatial.warp, we use skimage1
    warp as the backend, so we need to account for all this weirdness and
    undo it by updating the transform itself.
    """
    if callable(inverse_map):
        # If it's a callable, we permute axes, shift by 0.5, then shift back.
        def new_inverse_map(coords):
            coords = coords - 0.5
            ndim = coords.shape[1]
            if ndim == 2:
                coords = coords[:, ::-1]
            tformed = inverse_map(coords)
            return tformed + 0.5
        return new_inverse_map


def warp(
    image,
    inverse_map,
    map_args=None,
    output_range=None,
    interpolation_order=None,
    pad_mode='constant',
    pad_args=None,
):
    """Warp an image according to a given coordinate transformation.

    Parameters
    ----------
    image : ndarray
        Input image.
    inverse_map : transformation object, callable ``cr = f(cr, **kwargs)``, or ndarray
        Inverse coordinate map, which transforms coordinates in the output
        images into their corresponding coordinates in the input image.

        There are a number of different options to define this map, depending
        on the dimensionality of the input image. A 2-D image can have 2
        dimensions for gray-scale images, or 3 dimensions with color
        information.

         - For 2-D images, you can directly pass a transformation object,
           e.g. `skimage.transform.SimilarityTransform`, or its inverse.
         - For 2-D images, you can pass a ``(3, 3)`` homogeneous
           transformation matrix, e.g.
           `skimage.transform.SimilarityTransform.params`.
         - For 2-D images, a function that transforms a ``(M, 2)`` array of
           ``(col, row)`` coordinates in the output image to their
           corresponding coordinates in the input image. Extra parameters to
           the function can be specified through `map_args`.
         - For N-D images, you can directly pass an array of coordinates.
           The first dimension specifies the coordinates in the input image,
           while the subsequent dimensions determine the position in the
           output image. E.g. in case of 2-D images, you need to pass an array
           of shape ``(2, rows, cols)``, where `rows` and `cols` determine the
           shape of the output image, and the first dimension contains the
           ``(row, col)`` coordinate in the input image.
           See `scipy.ndimage.map_coordinates` for further documentation.

        Note, that a ``(3, 3)`` matrix is interpreted as a homogeneous
        transformation matrix, so you cannot interpolate values from a 3-D
        input, if the output is of shape ``(3,)``.

        See example section for usage.
    map_args : dict, optional
        Keyword arguments passed to `inverse_map`.
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated. By default the shape of the input
        image is preserved.  Note that, even for multi-band images, only rows
        and columns need to be specified.
    order : int, optional
        The order of interpolation. The order has to be in the range 0-5:
         - 0: Nearest-neighbor
         - 1: Bi-linear (default)
         - 2: Bi-quadratic
         - 3: Bi-cubic
         - 4: Bi-quartic
         - 5: Bi-quintic

         Default is 0 if image.dtype is bool and 1 otherwise.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Returns
    -------
    warped : double ndarray
        The warped input image.

    Notes
    -----
    - The input image is converted to a `double` image.
    - In case of a `SimilarityTransform`, `AffineTransform` and
      `ProjectiveTransform` and `order` in [0, 3] this function uses the
      underlying transformation matrix to warp the image with a much faster
      routine.

    Examples
    --------
    >>> from skimage.transform import warp
    >>> from skimage import data
    >>> image = data.camera()

    The following image warps are all equal but differ substantially in
    execution time. The image is shifted to the bottom.

    Use a geometric transform to warp an image (fast):

    >>> from skimage.transform import SimilarityTransform
    >>> tform = SimilarityTransform(translation=(-10, 0))
    >>> warped = warp(image, tform)

    Use a callable (slow):

    >>> def shift_down(coords):
    ...     coords[:, 0] -= 10
    ...     return coords
    >>> warped = warp(image, shift_down)

    Use a transformation matrix to warp an image (fast):

    >>> matrix = np.array([[1, 0, -10], [0, 1, 0], [0, 0, 1]])
    >>> warped = warp(image, matrix)
    >>> from skimage.transform import ProjectiveTransform
    >>> warped = warp(image, ProjectiveTransform(matrix=matrix))

    You can also use the inverse of a geometric transformation (fast):

    >>> warped = warp(image, tform.inverse)

    For N-D images you can pass a coordinate array, that specifies the
    coordinates in the input image for every element in the output image. E.g.
    if you want to rescale a 3-D cube, you can do:

    >>> cube_shape = np.array([30, 30, 30])
    >>> rng = np.random.default_rng()
    >>> cube = rng.random(cube_shape)

    Set up the coordinate array that defines the scaling:

    >>> scale = 0.1
    >>> output_shape = (scale * cube_shape).astype(int)
    >>> coords0, coords1, coords2 = np.mgrid[
    ...         :output_shape[0], :output_shape[1], :output_shape[2]
    ...         ]
    >>> coords = np.stack([coords0, coords1, coords2]) / scale

    >>> warped = warp(cube, coords)

    """
    legacy_inverse_map = _update_tform_for_skimage1_warp(inverse_map)
    import skimage as ski
    result = ski.transform.warp(
            image, legacy_inverse_map, map_args=map_args,
            output_shape=[end for _start, end in output_range],
            order=interpolation_order, mode=pad_mode,
            cval=pad_args['pad_constant'],
            clip=False,
            preserve_range=True
            )
    return result[tuple(slice(start, None) for start, _end in output_range)]
