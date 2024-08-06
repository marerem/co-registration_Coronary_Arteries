# Co-Registration of Coronary Artery Imaging

It's an algorithm with a graphical user interface (GUI) that assists in co-registering two sequences of images by aligning one sequence to another based on user-selected checkpoints.

## Motivation

To diagnose heart disease, doctors use a catheter with a camera to examine the coronary arteries. Imagine a stent procedure that widens an artery. Before and after the procedure, doctors receive two sets of images: one from before and one from after. These sets start and end at different points, have different angles due to the catheter's movement, and contain a different number of frames per time. Manually aligning these images would be quite challenging and time-consuming. But don't worry! This user-friendly GUI can help easily and quickly align them, making the process much smoother and stress-free for everyone involved.

## Complexity and Runtime

The complexity of the algorithm is O(n), where n is the number of frames in a sequence. This linear complexity ensures that the algorithm scales efficiently with the length of the image sequence.

### Runtime Performance

- **Average Time per 250 Frames:** 5 seconds

This performance metric demonstrates that the algorithm can process a significant amount of image data swiftly, providing timely assistance to medical professionals.

By leveraging this algorithm and GUI, doctors can save valuable time and focus more on patient care rather than the technicalities of image alignment.
