# Conversation Log: Watershed Axis-Order Invariance Investigation (Final)

## Summary of Initial Analysis
The watershed algorithm was initially described as being generally sensitive to axis ordering due to the use of a priority queue with a fixed (raveled C-order) neighbor exploration. This is accurate for images with ties (equal pixel intensities).

## Correction on Unique-Value Case
A detailed re-investigation of the `watershed_3d_axes.ipynb` notebook and further testing revealed that:
1.  **Unique Image + Fixed Seeds = Invariant:** If all pixel values are unique, the pop order is purely value-based, and the outcome is invariant to axis order.
2.  **The "Notebook Error" Explained:** The `AssertionError` observed in the user's notebook was not due to a labeling mismatch. Instead, it was caused by the `assert_labels_equivalent` function returning `None` (which is falsy), triggering an `AssertionError` when called as `assert assert_labels_equivalent(...)`.
3.  **Real-World Factors:** Even when "noise" is added to break ties, large images like `cells3d` can still contain hidden ties due to floating-point precision, making them appear non-invariant in practice.

## Final Summary of findings
The `watershed` function is technically axis-invariant for strictly unique images. In presence of any ties, the algorithm resorts to axis-dependent tie-breaking (via raveled-index processing order and neighbor exploration order), which is expected behavior for this implementation of the watershed algorithm.
