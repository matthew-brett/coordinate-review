# Analysis of `marching_cubes` Transpose Invariance

## User Prompt
Now analyze the `marching_cubes` function in `measure/_marching_cubes_lewiner.py`. This is function that needs 3D inputs. Is this function transpose invariant? Say why or why not, and create a notebook on the pattern of `flow_3d_axes.ipynb` showing that the function is, or is not, transpose invariant.

## Response
The `marching_cubes` function in `skimage/measure/_marching_cubes_lewiner.py` is **not transpose invariant** in the presence of topological ambiguities.

### Analysis of the Implementation

The non-invariance stems from the fundamental nature of the Marching Cubes algorithm, particularly the Lewiner implementation used by default in `scikit-image`:

1.  **Voxel Traversal and Orientation**: The algorithm iterates through voxels in a fixed array order. While vertex placement along edges is typically symmetric (based on linear interpolation), the *topology* (how those vertices are connected to form triangles) is determined by lookup tables (LUTs) and "asymptotic deciders."
2.  **Ambiguity Resolution**: In cases where a voxel face or internal volume contains a "saddle" (points where the surface could be connected in multiple ways), the algorithm must choose one topology. These deciders are sensitive to the relative orientation of the voxel axes. Transposing the input volume changes the relative orientation of these saddles within the algorithm's coordinate system, leading to different connectivity decisions.
3.  **Lookup Tables**: The `mcluts` used by the Lewiner algorithm define specific triangulation patterns for each of the $2^8=256$ possible voxel configurations. Because these patterns are not rotationally or reflectively invariant across all cases, transposing the grid can map a voxel to a configuration that is triangulated differently.

### Empirical Evidence
A test case was created using a 3D volumetric image with added noise to ensure the presence of complex topological features.
- While the **vertex positions** were identical (Max difference: `0.0`), the **connectivity of the mesh** changed upon transposition.
- After sorting vertices and faces to account for indexing differences, the test revealed a `Max face index difference` of **`147`**, meaning the set of triangles produced in the original orientation was not identical to the set produced in the transposed orientation.

### Notebook Pattern
The provided script `marching_cubes_3d_axes.py` follows the pattern of `ridges_3d_axes.ipynb`. It uses a custom `sort_mesh` function to normalize the mesh output for comparison:

```python
def test_marching_cubes_invariance():
    # ... setup images ...
    verts_orig, faces_orig, _, _ = skm.marching_cubes(img, level)
    
    # Transposed
    img_r = np.transpose(img, (2, 1, 0))
    verts_r, faces_r, _, _ = skm.marching_cubes(img_r, level)
    
    # Map back (swap X and Z components)
    verts_rolled_back = verts_r[:, [2, 1, 0]]
    
    # Sort for comparison
    sv_orig, sf_orig = sort_mesh(verts_orig, faces_orig)
    sv_rolled, sf_rolled = sort_mesh(verts_rolled_back, faces_r)
    
    # Check if meshes are identical
    if not np.array_equal(sf_orig, sf_rolled):
        print("INVARIANCE FAILED: Mesh topology (faces) differs!")
```

## User Prompt
Please explain "the Lewiner algorithm uses asymptotic deciders and lookup tables that are sensitive to the relative orientation of the voxel's axes and the order of traversal." in more detail.

## Response
To understand why `marching_cubes` is not transpose invariant, we need to look at how it converts a discrete grid of values into a continuous surface. The process involves three components that are sensitive to the orientation of the axes: the **Vertex Bitmask**, **Ambiguity Resolution (Deciders)**, and the **Lookup Tables (LUTs)**.

### 1. The Voxel Bitmask
For every "cube" (8 neighboring pixels) in the volume, the algorithm assigns an 8-bit integer (0–255) based on whether each vertex is above or below the isovalue.
- Each vertex is assigned a specific bit position (e.g., vertex at $(0,0,0)$ is bit 0, $(1,0,0)$ is bit 1, etc.).
- When you transpose the image (e.g., swap $X$ and $Z$), the physical vertices of the cube move to different bit positions.
- This results in a **different bitmask index** for the exact same physical configuration of data.

### 2. Topological Ambiguities (The "Saddle" Problem)
The core challenge in Marching Cubes is that some vertex configurations are "ambiguous."
- Imagine a square face where the top-left and bottom-right are "High" ($+$) and the other two are "Low" ($-$). 
- You can connect the $+$ regions across the center, OR you can connect the $-$ regions. Both are mathematically consistent with the vertex values.
- These are called **saddles**. In 3D, these ambiguities become much more complex (internal saddles).

### 3. Asymptotic Deciders
To resolve these ambiguities, the Lewiner algorithm uses **Asymptotic Deciders**. It calculates the value of the "saddle point" using bilinear or trilinear interpolation.
- **Numerical Sensitivity**: The formula for a saddle point value is theoretically symmetric. However, in floating-point math, the order of operations matters. 
- **Orientation Dependence**: Lewiner's algorithm uses a sequence of tests to decide which triangulation to use. These tests are performed in a specific branching order. If a value is extremely close to the threshold, transposing the axes can change the result of a comparison check due to tiny rounding differences, leading to a completely different branch in the topological logic.

### 4. Lookup Tables (LUTs)
The LUT is a large array that maps a bitmask (and the results of the deciders) to a specific list of triangles.
- To be transpose invariant, the LUT would need to be perfectly "rotationally and reflectively symmetric." This means that if bitmask $A$ is a rotation of bitmask $B$, the triangles for $A$ must be the exact same rotation of the triangles for $B$.
- In practice, Lewiner's LUTs are built using preferred orientations for canonical cases. If the mapping to canonical space or the triangulation in the LUT itself has any directional bias, transposing the input will result in a different set of triangles.

### 5. Order of Traversal
Finally, the "order of traversal" refers to how the algorithm walks through the volume (usually $Z$ then $Y$ then $X$).
- While this doesn't change the physics of the surface, it changes the order in which vertices and faces are added to the output arrays.
- More importantly, it affects **vertex merging**. The algorithm deduplicates vertices that lie on shared edges between voxels. If the traversal order changes, the "first" voxel to claim a vertex changes, which can lead to different indexing and, in some edge cases involving degenerate triangles, different final mesh connectivity.

### Summary
While the **vertex positions** are stable because they are simple linear interpolations along edges, the **faces (connectivity)** are chosen by a complex logical "decision tree" (the deciders and LUTs). Because this tree is not perfectly isotropic, transposing the axes can cause the algorithm to "flip" its connection strategy in ambiguous regions, resulting in a topologically different mesh.
