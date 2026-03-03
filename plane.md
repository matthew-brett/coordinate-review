# Instances of plane in codebase

These are instances (via `ripgrep`) of `plane` in code-base, reviewed and not
relevant, and not yet reviewed.

## Checked, not relevant

* `src/skimage/graph/spath.py:    # Valid starting positions are
  anywhere on the hyperplane defined by`
* `src/skimage/graph/spath.py:    # hyperplane at position -1 along the same.`

  From `shortest_path` function.  Function is clearly N-D, but asks the user
  to specify (`axis=-1`) the "axis along which the path must always move
  forward".

* `src/skimage/graph/_mcp.pyx:    """Return an array with edge points/lines/planes/hyperplanes marked.`

  Function is clearly N-D.  Points etc are terminology for edges in 1D, 2D ...

* `src/skimage/restoration/_denoise.py:    plane separately.`

  In fact `on each color plane separately`.  Not relevant to plane in semantic
  sense.

## Not checked

All done now.
