# Co-Registration of Coronary Artery Imaging

This algorithm, with a graphical user interface (GUI), assists in co-registering two sequences of images by aligning one sequence to another based on user-selected checkpoints.

## Motivation

Doctors use a catheter with a camera to examine coronary arteries. Before and after procedures like stent placements, two sets of images are obtained. These sets start and end at different points, have different angles, and contain a different number of frames. Manually aligning these images is challenging and time-consuming. This user-friendly GUI simplifies the alignment process, making it quicker and easier.

## Complexity and Runtime

The algorithm has a complexity of O(n), where n is the number of frames in a sequence. 

### Runtime Performance

- **Average Time per 250 Frames:** 5 seconds

This performance shows the algorithm can swiftly process large sets of image data, aiding medical professionals efficiently.
