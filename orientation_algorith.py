import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.measure import regionprops
from skimage.color import rgb2gray
from scipy.spatial import ConvexHull

def calculate_orientation_circle(binary_image):
    # Label the image
    label_image = measure.label(binary_image)

    # Remove small objects
    min_size = 50  # Adjust this threshold based on the size of objects you consider small
    cleaned_image = morphology.remove_small_objects(label_image, min_size=min_size)

    # Find the largest object
    regions = regionprops(cleaned_image)
    largest_region = max(regions, key=lambda r: r.area)

    # Get the coordinates of the largest object's region
    coords = largest_region.coords

    # Calculate the centroid of the largest object
    centroid = largest_region.centroid

    # Calculate the convex hull
    hull = ConvexHull(coords)
    hull_points = coords[hull.vertices]

    # Fit a circle around the convex hull
    distances = np.linalg.norm(hull_points - centroid, axis=1)
    radius = np.mean(distances)
    
    # Calculate orientation (arbitrary direction in this case)
    direction_vertex = hull_points[np.argmax(distances)]
    orientation = np.arctan2(direction_vertex[1] - centroid[1], direction_vertex[0] - centroid[0])

    # Plotting the cleaned image with the fitted circle and orientation
    #fig, ax = plt.subplots(figsize=(6, 6))
    #axes[0].imshow(cleaned_image, cmap='gray')

    # Plot the centroid
    #axes[0].plot(centroid[1], centroid[0], 'ro')

    # Plot the orientation arrow (from centroid to direction vertex)
    #axes[0].arrow(centroid[1], centroid[0], direction_vertex[1] - centroid[1], direction_vertex[0] - centroid[0], 
    #         head_width=5, head_length=10, fc='r', ec='r', label='Orientation')

    # Plot the fitted circle
    circle = plt.Circle((centroid[1], centroid[0]), radius, color='b', fill=False, label='Fitted Circle')
    #axes[0].add_artist(circle)

    # Set titles and show the plot
    #ax.set_title('Center and Orientation of the Object (Direction Vertex)')
    #ax.legend()
    #plt.show()

    return orientation,centroid,circle,direction_vertex
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology, io
from skimage.measure import regionprops
from sklearn.decomposition import PCA

# Assuming binary_image is defined elsewhere in your code
# Example: binary_image = io.imread('path_to_your_image')
def pca(binary_image):
    # Label the image
    label_image = measure.label(binary_image)

    # Remove small objects
    min_size = 1000  # Adjust this threshold based on the size of objects you consider small
    cleaned_image = morphology.remove_small_objects(label_image, min_size=min_size)

    # Find the largest object
    regions = regionprops(cleaned_image)
    largest_region = max(regions, key=lambda r: r.area)

    # Get the center of mass
    center_of_mass = largest_region.centroid

    # Get the coordinates of the object's pixels
    coords = np.column_stack(np.nonzero(cleaned_image == largest_region.label))

    # Apply PCA to find the orientation
    pca = PCA(n_components=2)
    pca.fit(coords)
    principal_axes = pca.components_
    explained_variance = pca.explained_variance_

    # Calculate the endpoints of the major principal axis for plotting
    length_major = 50  # Length of the major axis line
    x0, y0 = center_of_mass
    x1_major = x0 + length_major * principal_axes[0, 0]
    y1_major = y0 + length_major * principal_axes[0, 1]
    x2_major = x0 - length_major * principal_axes[0, 0]
    y2_major = y0 - length_major * principal_axes[0, 1]

    # Plot the cleaned image with the detected center and major principal axis
    #fig, ax = plt.subplots(figsize=(6, 6))
    #ax.imshow(cleaned_image, cmap='gray')

    # Plot the center of mass
    #ax.plot(center_of_mass[1], center_of_mass[0], 'ro')

    # Plot the major axis in two directions
    #ax.arrow(center_of_mass[1], center_of_mass[0], x1_major - x0, y1_major - y0, head_width=10, head_length=15, fc='blue', ec='blue', label='Major Axis (Positive)')
    #ax.arrow(center_of_mass[1], center_of_mass[0], x2_major - x0, y2_major - y0, head_width=10, head_length=15, fc='red', ec='red', label='Major Axis (Negative)')

    # Set titles and show the plot
    #ax.set_title('Center and Major Axis of the Object (PCA)')
    #ax.legend()
    #plt.show()
    return center_of_mass,x1_major,x0,y1_major,y0,x2_major,y2_major

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology, io
from skimage.measure import regionprops
from sklearn.decomposition import PCA

# Assuming binary_image is defined elsewhere in your code
# Example: binary_image = io.imread('path_to_your_image')
def pca_change_direction(binary_image):
    # Label the image
    label_image = measure.label(binary_image)

    # Remove small objects
    min_size = 1000  # Adjust this threshold based on the size of objects you consider small
    cleaned_image = morphology.remove_small_objects(label_image, min_size=min_size)

    # Find the largest object
    regions = regionprops(cleaned_image)
    largest_region = max(regions, key=lambda r: r.area)

    # Get the coordinates of the object's pixels
    coords = np.column_stack(np.nonzero(cleaned_image == largest_region.label))

    # Get the center of mass
    center_of_mass = largest_region.centroid

    # Get the boundary of the object
    boundary = largest_region.coords

    # Function to apply PCA and return the principal axes
    def get_principal_axes(coords):
        pca = PCA(n_components=2)
        pca.fit(coords)
        return pca.components_

    # Divide the boundary into segments
    num_segments = 10
    segment_length = len(boundary) // num_segments
    principal_axes_segments = []

    for i in range(num_segments):
        segment_coords = boundary[i * segment_length:(i + 1) * segment_length]
        principal_axes = get_principal_axes(segment_coords)
        principal_axes_segments.append(principal_axes[0])  # Major axis

    # Calculate the angle changes between consecutive segments
    angles = [np.arctan2(axis[1], axis[0]) for axis in principal_axes_segments]
    angle_changes = np.diff(angles)
    max_change_index = np.argmax(np.abs(angle_changes))

    # Get the coordinates of the segment with the maximum change
    change_point_start = boundary[max_change_index * segment_length]
    change_point_end = boundary[(max_change_index + 1) * segment_length - 1]

    # Plot the cleaned image with the detected center and principal axes
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cleaned_image, cmap='gray')

    # Plot the boundary
    #ax.plot(boundary[:, 1], boundary[:, 0], 'yellow', lw=2)

    # Plot the center of mass
    #ax.plot(center_of_mass[1], center_of_mass[0], 'ro')

    # Plot the change points
    #ax.plot(change_point_start[1], change_point_start[0], 'go')
    #ax.plot(change_point_end[1], change_point_end[0], 'go')

    # Plot an arrow indicating the direction of change from the center of mass
    dx = (change_point_start[1] + change_point_end[1]) / 2 - center_of_mass[1]
    dy = (change_point_start[0] + change_point_end[0]) / 2 - center_of_mass[0]
    #ax.arrow(center_of_mass[1], center_of_mass[0], dx, dy, head_width=10, head_length=15, fc='green', ec='green', label='Change Direction')

    # Set titles and show the plot
    #ax.set_title('Shape Change Detection Using PCA')
    #ax.legend()
    #plt.show()
    return center_of_mass,change_point_start,change_point_end,dx,dy
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology, io, draw
from skimage.measure import regionprops
from skimage.color import rgb2gray
from scipy.spatial.distance import cdist
def fit_ellipse(binary_image):
    def calculate_overlap(binary_image, center, major_axis, minor_axis, orientation):
        # Create an empty image to draw the ellipse
        ellipse_image = np.zeros_like(binary_image)
        rr, cc = draw.ellipse_perimeter(int(center[0]), int(center[1]), int(major_axis), int(minor_axis), orientation)

        # Make sure the ellipse coordinates are within the image bounds
        rr = np.clip(rr, 0, binary_image.shape[0] - 1)
        cc = np.clip(cc, 0, binary_image.shape[1] - 1)

        ellipse_image[rr, cc] = 1

        # Calculate the overlap between the ellipse and the binary image
        overlap = np.logical_and(ellipse_image, binary_image)
        return np.sum(overlap)


    rr, cc = draw.disk((50, 50), 30)
    #binary_image[rr, cc] = 1

    # Label the image
    label_image = measure.label(binary_image)

    # Remove small objects
    min_size = 1000  # Adjust this threshold based on the size of objects you consider small
    cleaned_image = morphology.remove_small_objects(label_image, min_size=min_size)

    # Find the largest object
    regions = regionprops(cleaned_image)
    largest_region = max(regions, key=lambda r: r.area)

    # Get the center of mass and axis lengths of the fitted ellipse
    center_of_mass = largest_region.centroid
    major_axis_length = largest_region.major_axis_length / 2.0
    minor_axis_length = largest_region.minor_axis_length / 2.0

    # Iterate over a range of orientations to find the best fit
    best_orientation = 0
    max_overlap = 0
    for angle in np.linspace(0, np.pi, 180):
        overlap = calculate_overlap(cleaned_image, center_of_mass, major_axis_length, minor_axis_length, angle)
    if overlap > max_overlap:
        max_overlap = overlap
        best_orientation = angle

    # Get the ellipse perimeter for the best orientation
    rr, cc = draw.ellipse_perimeter(int(center_of_mass[0]), int(center_of_mass[1]), int(major_axis_length), int(minor_axis_length), best_orientation)

    # Find edge coordinates of the largest region
    edges = np.array(largest_region.coords)

    # Calculate distances from the center to all edge points
    distances = cdist([center_of_mass], edges)[0]

    # Find the farthest point
    farthest_point_idx = np.argmax(distances)
    farthest_point = edges[farthest_point_idx]

    # Plot the cleaned image with the best fit ellipse and orientation
    #fig, ax = plt.subplots(figsize=(6, 6))
    #ax.imshow(cleaned_image, cmap='gray')

    # Plot the center of mass
    #ax.plot(center_of_mass[1], center_of_mass[0], 'ro')

    # Calculate the endpoints of the major axis for the best orientation
    x0, y0 = center_of_mass
    x1 = x0 + major_axis_length * np.cos(best_orientation)
    y1 = y0 + major_axis_length * np.sin(best_orientation)

    # Plot the orientation arrow (major axis)
    #ax.arrow(y0, x0, y1 - y0, x1 - x0, head_width=5, head_length=10, fc='r', ec='r', label='Orientation')

    # Plot the ellipse
    #ax.plot(cc, rr, 'b-', label='Fitted Ellipse')

    # Plot the arrow to the farthest point
    #ax.arrow(center_of_mass[1], center_of_mass[0], farthest_point[1] - center_of_mass[1], farthest_point[0] - center_of_mass[0],
    #        head_width=5, head_length=10, fc='g', ec='g', label='Farthest Point')

    # Set titles and show the plot
    #ax.set_title('Best Fit Ellipse and Orientation with Farthest Point')
    #ax.legend()
    #plt.show()
    return center_of_mass,farthest_point,y0,x0,y1,x1,rr,cc

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology, io
from skimage.measure import regionprops
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean

# Load binary image (replace this with actual binary image)
def convex_hull(binary_image):

    # Label the image
    label_image = measure.label(binary_image)

    # Remove small objects
    min_size = 50  # Adjust this threshold based on the size of objects you consider small
    cleaned_image = morphology.remove_small_objects(label_image, min_size=min_size)

    # Find the largest object
    regions = regionprops(cleaned_image)
    largest_region = max(regions, key=lambda r: r.area)

    # Get the coordinates of the largest object
    coords = largest_region.coords

    # Find the convex hull of the largest object
    hull = ConvexHull(coords)
    hull_points = coords[hull.vertices]

    # Find the two longest edges of the convex hull
    edges = [(hull_points[i], hull_points[(i + 1) % len(hull_points)]) for i in range(len(hull_points))]
    longest_edges = sorted(edges, key=lambda edge: euclidean(edge[0], edge[1]), reverse=True)[:2]

    # Function to find line intersection
    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('Lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    # Calculate the intersection point of the two longest edges
    intersection = line_intersection(longest_edges[0], longest_edges[1])

    # Plot the cleaned image with the convex hull and longest edges
    #fig, ax = plt.subplots(figsize=(6, 6))
    #ax.imshow(cleaned_image, cmap='gray')

    # Plot the convex hull
    #for simplex in hull.simplices:
    #    ax.plot(coords[simplex, 1], coords[simplex, 0], 'b-')

    # Plot the two longest edges
    #for edge in longest_edges:
    #    ax.plot([edge[0][1], edge[1][1]], [edge[0][0], edge[1][0]], 'g-', linewidth=2)

    # Plot the intersection point
    #ax.plot(intersection[1], intersection[0], 'ro')

    # Set titles and show the plot
    #ax.set_title('Convex Hull with Longest Edges and Intersection Point')
    #plt.show()
    return intersection,longest_edges, coords, hull

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.measure import regionprops
from scipy.spatial import ConvexHull

def triangle(binary_image):
    # Label the image
    label_image = measure.label(binary_image)

    # Remove small objects
    min_size = 50  # Adjust this threshold based on the size of objects you consider small
    cleaned_image = morphology.remove_small_objects(label_image, min_size=min_size)

    # Find the largest object
    regions = regionprops(cleaned_image)
    if not regions:
        raise ValueError("No regions found in the image.")
        
    largest_region = max(regions, key=lambda r: r.area)

    # Get the coordinates of the largest object's region
    coords = largest_region.coords

    # Calculate the convex hull
    hull = ConvexHull(coords)
    hull_points = coords[hull.vertices]

    # Find the triangle with the maximum area in the convex hull
    max_area = 0
    triangle = None
    for i in range(len(hull_points)):
        for j in range(i+1, len(hull_points)):
            for k in range(j+1, len(hull_points)):
                # Calculate the area of the triangle formed by points i, j, k
                p1, p2, p3 = hull_points[i], hull_points[j], hull_points[k]
                area = 0.5 * np.abs(np.cross(p2-p1, p3-p1))
                if area > max_area:
                    max_area = area
                    triangle = (p1, p2, p3)

    if triangle is None:
        raise ValueError("Unable to determine the largest triangle.")

    # Calculate the centroid of the triangle
    triangle = np.array(triangle)
    centroid = triangle.mean(axis=0)

    # Identify the longest edge of the triangle and the associated vertex
    max_distance = 0
    longest_edge_vertex = None
    for i in range(len(triangle)):
        for j in range(i+1, len(triangle)):
            distance = np.linalg.norm(triangle[i] - triangle[j])
            if distance > max_distance:
                max_distance = distance
                longest_edge_vertex = triangle[i]

    # Calculate the orientation from the centroid to the vertex with the longest edge
    orientation = np.arctan2(longest_edge_vertex[1] - centroid[1], longest_edge_vertex[0] - centroid[0])

    # Plotting the cleaned image with the fitted triangle and orientation
    #fig, ax = plt.subplots(figsize=(6, 6))
    #ax.imshow(cleaned_image, cmap='gray')

    # Plot the centroid
    #ax.plot(centroid[1], centroid[0], 'ro')

    # Plot the orientation arrow (from centroid to the vertex with the longest edge)
    #ax.arrow(centroid[1], centroid[0], longest_edge_vertex[1] - centroid[1], longest_edge_vertex[0] - centroid[0], 
    #         head_width=5, head_length=10, fc='r', ec='r', label='Orientation')

    # Plot the triangle
    triangle_points = np.vstack([triangle, triangle[0]])  # Close the triangle loop
    #ax.plot(triangle_points[:, 1], triangle_points[:, 0], 'b-', label='Fitted Triangle')

    # Set titles and show the plot
    #ax.set_title('Center and Orientation of the Object (Longest Edge Vertex)')
    #ax.legend()
    #plt.show()

    return centroid, triangle_points, longest_edge_vertex

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.measure import regionprops
from scipy.spatial import ConvexHull

def triangle_max_edges(binary_image):
    # Label the image
    label_image = measure.label(binary_image)

    # Remove small objects
    min_size = 50  # Adjust this threshold based on the size of objects you consider small
    cleaned_image = morphology.remove_small_objects(label_image, min_size=min_size)

    # Find the largest object
    regions = regionprops(cleaned_image)
    if not regions:
        raise ValueError("No regions found in the image.")
        
    largest_region = max(regions, key=lambda r: r.area)

    # Get the coordinates of the largest object's region
    coords = largest_region.coords

    # Calculate the convex hull
    hull = ConvexHull(coords)
    hull_points = coords[hull.vertices]

    # Find the triangle with the maximum area in the convex hull
    max_area = 0
    triangle = None
    for i in range(len(hull_points)):
        for j in range(i+1, len(hull_points)):
            for k in range(j+1, len(hull_points)):
                # Calculate the area of the triangle formed by points i, j, k
                p1, p2, p3 = hull_points[i], hull_points[j], hull_points[k]
                area = 0.5 * np.abs(np.cross(p2-p1, p3-p1))
                if area > max_area:
                    max_area = area
                    triangle = (p1, p2, p3)

    if triangle is None:
        raise ValueError("Unable to determine the largest triangle.")

    # Calculate the centroid of the triangle
    triangle = np.array(triangle)
    centroid = triangle.mean(axis=0)

    # Identify the two longest edges of the triangle
    edges = [
        (np.linalg.norm(triangle[0] - triangle[1]), (triangle[0], triangle[1])),
        (np.linalg.norm(triangle[1] - triangle[2]), (triangle[1], triangle[2])),
        (np.linalg.norm(triangle[2] - triangle[0]), (triangle[2], triangle[0]))
    ]
    edges.sort(reverse=True, key=lambda x: x[0])
    longest_edges = edges[:2]

    # Find the common vertex in the two longest edges
    common_vertex = None
    for vertex in triangle:
        if all(np.array_equal(vertex, edge[1][0]) or np.array_equal(vertex, edge[1][1]) for edge in longest_edges):
            common_vertex = vertex
            break

    if common_vertex is None:
        raise ValueError("Unable to determine the common vertex for the longest edges.")

    # Calculate the orientation from the centroid to the common vertex
    orientation = np.arctan2(common_vertex[1] - centroid[1], common_vertex[0] - centroid[0])

    # Plotting the cleaned image with the fitted triangle and orientation
    #fig, ax = plt.subplots(figsize=(6, 6))
    #ax.imshow(cleaned_image, cmap='gray')

    # Plot the centroid
    #ax.plot(centroid[1], centroid[0], 'ro')

    # Plot the orientation arrow (from centroid to the common vertex)
    #ax.arrow(centroid[1], centroid[0], common_vertex[1] - centroid[1], common_vertex[0] - centroid[0], 
    #         head_width=5, head_length=10, fc='r', ec='r', label='Orientation')

    # Highlight the common vertex
    #ax.plot(common_vertex[1], common_vertex[0], 'go', markersize=10, label='Common Vertex')

    # Plot the triangle
    triangle_points = np.vstack([triangle, triangle[0]])  # Close the triangle loop
    #ax.plot(triangle_points[:, 1], triangle_points[:, 0], 'b-', label='Fitted Triangle')

    # Set titles and show the plot
    #ax.set_title('Center and Orientation of the Object (Longest Edges Vertex)')
    #ax.legend()
    #plt.show()

    return centroid, triangle_points, common_vertex
"""
def mesh_bif()
        # stores angle measurements from selected landmark bifurcations
        self.OCT_bif_angles = []
        self.CT_bif_angles = []
        
        # rotational alignment
        # +-3 frames for each selected index
        range_size = 3
        
        masks=np.array(OCT_sdf_cpr > 0)
        # masks_center=np.zeros_like(masks)
        frames=masks.shape[1]
        centerline=np.zeros((frames,3))
       
        for f in range(frames):
            mask=masks[:, f, :, :]
            # finding center of mass for each frame? where is the 2 coming from?
            centerline[f,:3]=scipy.ndimage.center_of_mass(mask)
            
            #plt.imshow(mask.squeeze())
            #plt.plot(centerline[f,2],centerline[f,1], marker='v', color="white")
            #plt.plot(250,250, marker='o', color="red")
            #plt.show()
           
        centerline[:,0]=np.linspace(0,frames-1,frames)
        #plt.plot(centerline[:,0], centerline[:,1])
        # masks_full=np.array(data[0])
        # masks_full_center=np.zeros_like(masks_full)
        centerline_smoothed=np.zeros_like(centerline)
        
        for dim in range(3):
            centerline_smoothed[:,dim]=scipy.signal.savgol_filter(centerline[:,dim],31,2)
        self.OCT_centerline = centerline_smoothed

        for i in range(len(self.OCT_selected_indices_shift)):
 
            OCT_original_idx = self.OCT_selected_indices_shift[i]
            CT_original_idx = self.CT_selected_indices_shift[i]
            #took 6 frames per bifurcation
            if OCT_original_idx < range_size:
                OCT_bif = OCT_sdf_cpr[:, OCT_original_idx:OCT_original_idx+range_size+1, :, :]
            elif OCT_original_idx > OCT_sdf_cpr.shape[1] - range_size:
                OCT_bif = OCT_sdf_cpr[:, OCT_original_idx-range_size:OCT_original_idx, :, :]
            else:
                OCT_bif = OCT_sdf_cpr[:, OCT_original_idx-range_size:OCT_original_idx+range_size+1, :, :]
            
            if CT_original_idx < range_size:
                CT_bif = CT_sdf_cpr[:, CT_original_idx:CT_original_idx+range_size+1, :, :]
            elif CT_original_idx > CT_sdf_cpr.shape[1] - range_size:
                CT_bif = CT_sdf_cpr[:, CT_original_idx-range_size:CT_original_idx, :, :]
            else:
                CT_bif = CT_sdf_cpr[:, CT_original_idx-range_size:CT_original_idx+range_size+1, :, :]
            
            #pdb.set_trace()
            OCT_bif_smush = torch.sum(OCT_bif, dim=1, keepdim=True)
            OCT_bif_smush = OCT_bif_smush > 0
            CT_bif_smush = torch.sum(CT_bif, dim=1, keepdim=True)
            CT_bif_smush = CT_bif_smush > 0
            

            
            # find angle from center to bifurcation for smushed images
            OCT_contours, _ = cv2.findContours(OCT_bif_smush.numpy().squeeze().astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            CT_contours, _ = cv2.findContours(CT_bif_smush.numpy().squeeze().astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          
            # find contour with max area
            OCT_max_contour = max(OCT_contours, key=cv2.contourArea)
            CT_max_contour = max(CT_contours, key=cv2.contourArea)
           
            # calculate the center of the image for OCT
            frame = self.OCT_selected_indices_shift[i]
            center_OCT = (centerline_smoothed[frame, 2], centerline_smoothed[frame, 1])
            
            # calculate the center of the image for CT
            height, width = CT_bif_smush.numpy().squeeze().shape[:2]
            #center_CT = (int(width / 2), int(height / 2))# this is here because u dont have line via center 
            M = cv2.moments(CT_max_contour)
            if M["m00"] != 0:
                center_CT = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                center_CT = (int(width / 2), int(height / 2))
            
            OCT_distances = np.sqrt(np.sum((OCT_max_contour - center_OCT)**2, axis=2))
            OCT_outermost_index = np.argmax(OCT_distances)
            OCT_outermost_point = tuple(OCT_max_contour[OCT_outermost_index][0])
            print('OCT outermost point: ', OCT_outermost_point)

            CT_distances = np.sqrt(np.sum((CT_max_contour - center_CT)**2, axis=2))
            CT_outermost_index = np.argmax(CT_distances)
            CT_outermost_point = tuple(CT_max_contour[CT_outermost_index][0])
            print('CT outermost point: ', CT_outermost_point)
            
            # calculate angle between center and outermost points of contour
            OCT_delta_x = OCT_outermost_point[0] - center_OCT[0]
            OCT_delta_y = center_OCT[1] - OCT_outermost_point[1]
            OCT_angle_in_radians = math.atan2(OCT_delta_y, OCT_delta_x)
            OCT_angle_in_degrees = math.degrees(OCT_angle_in_radians)
            self.OCT_bif_angles.append(OCT_angle_in_radians)
            
            CT_delta_x = CT_outermost_point[0] - center_CT[0]
            CT_delta_y = center_CT[1] - CT_outermost_point[1]
            CT_angle_in_radians = math.atan2(CT_delta_y, CT_delta_x)
            CT_angle_in_degrees = math.degrees(CT_angle_in_radians)
            self.CT_bif_angles.append(CT_angle_in_radians)
            
            # display the angle
            print(OCT_angle_in_degrees)
            print(CT_angle_in_degrees)
"""            
