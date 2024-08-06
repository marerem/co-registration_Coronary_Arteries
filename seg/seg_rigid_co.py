#import
import torch
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from joblib import load
import cv2
# segmentation data
import nibabel as nib
from skimage import measure, morphology
from scipy.ndimage import distance_transform_edt
from torchcubicspline import NaturalCubicSpline
from torchcubicspline import natural_cubic_spline_coeffs
from scipy.interpolate import PchipInterpolator
import math

import numpy as np
from scipy.signal import savgol_filter
import numpy as np
from skimage import measure, morphology
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects 






def rotation(rot,center_want,center_have):
    sfift_rots = []
    for i in range(rot.shape[1]):

        shift_x = center_want[i][1] - center_have[i][1]
        shift_y = center_want[i][0] - center_have[i][0]

        # Create the translation matrix
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        # Apply the translation
        shifted_image = cv2.warpAffine(rot[0,i,...].detach().numpy(), translation_matrix, (rot.shape[-1], rot.shape[-1]))
        sfift_rots.append(shifted_image)
    return sfift_rots



def ridgit_register(images,angles):
    rot_agnl = []
    rotated_images = []
    for i in range(images.shape[0]):
        angle_in_radians = float(angles[i][0])
        #print(angle_in_radians)
        image = images[i,...].detach().numpy()
        # Convert angle from radians to degrees
        
        
        #sign = np.sign(angle_in_radians)
    
        angle_in_degrees = angle_in_radians*180/math.pi 
        #if abs(angle_in_degrees) > 180:
        #    angle_in_degrees = abs(angle_in_degrees) - 360

        #if angle_in_degrees > 0:
        #    angle_in_degrees *=-1
       
        rot_agnl.append(angle_in_degrees)
        #print(angle_in_degrees)
        #print(angle_in_degrees)
        # Get the image dimensions
      
        (h, w) = image.shape[:2]

        # Calculate the center of the image
        center = (w // 2, h // 2)

        #center = (float(centers[i][1]), float(centers[i][0]))
        #print(center)
        # Compute the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle_in_degrees, 1.0)

        # Perform the rotation
        rotated_image = cv2.warpAffine(image, M, (w, h))
        rotated_images.append(rotated_image)
    return torch.tensor(np.array(rotated_images))


def center_circle(binary_image):
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

    # Create a mask for the largest object
    mask = np.zeros_like(binary_image, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True

    # Compute the distance transform
    distance_map = distance_transform_edt(mask)

    # Find the maximum distance in the distance map
    max_dist_idx = np.unravel_index(np.argmax(distance_map), distance_map.shape)
    radius = distance_map[max_dist_idx]
    centroid = np.array(max_dist_idx)

    # Find the farthest point on the boundary to define the orientation
    distances = np.linalg.norm(coords - centroid, axis=1)
    direction_vertex = coords[np.argmax(distances)]

    # Calculate the orientation
    orientation = np.arctan2(direction_vertex[1] - centroid[1], direction_vertex[0] - centroid[0])

    return orientation,centroid
import numpy as np
import random

def duplicate_frames(sequence, num_duplicates=1, exclusion_fraction=0.2):
    """
    Duplicates a specified number of randomly chosen frames from the sequence, excluding the middle portion.
    
    Parameters:
    sequence (np.ndarray): The 3D array of frames with shape (height, width, number_of_frames).
    num_duplicates (int): The number of frames to duplicate. Default is 1.
    exclusion_fraction (float): The fraction of the sequence to exclude from the middle. Default is 0.2 (20%).

    Returns:
    np.ndarray: The new sequence with the duplicated frames.
    """
    if not 0 <= exclusion_fraction < 1:
        raise ValueError("Exclusion fraction must be between 0 and 1.")
    
    h, w, total_frames = sequence.shape
    exclusion_size = int(total_frames * exclusion_fraction)
    
    if exclusion_size == 0:
        exclusion_size = 1  # Ensure at least one frame is excluded from the middle
    
    middle_start = (total_frames - exclusion_size) // 2
    middle_end = middle_start + exclusion_size
    
    valid_indices = list(range(0, middle_start)) + list(range(middle_end, total_frames))
    
    if not valid_indices:
        raise ValueError("No valid indices available for duplication.")
    
    chosen_indices = random.choices(valid_indices, k=num_duplicates)
    new_sequence = sequence.copy()
    
    for idx in chosen_indices:
        frame_to_duplicate = new_sequence[:, :, idx]
        new_sequence = np.insert(new_sequence, idx + 1, frame_to_duplicate, axis=2)
    
    return new_sequence
import numpy as np
import random

def remove_frames(sequence, num_removals=1, exclusion_fraction=0.2):
    """
    Removes a specified number of randomly chosen frames from the sequence, excluding the middle portion.
    
    Parameters:
    sequence (np.ndarray): The 3D array of frames with shape (height, width, number_of_frames).
    num_removals (int): The number of frames to remove. Default is 1.
    exclusion_fraction (float): The fraction of the sequence to exclude from the middle. Default is 0.2 (20%).

    Returns:
    np.ndarray: The new sequence with the frames removed.
    """
    if not 0 <= exclusion_fraction < 1:
        raise ValueError("Exclusion fraction must be between 0 and 1.")
    
    h, w, total_frames = sequence.shape
    exclusion_size = int(total_frames * exclusion_fraction)
    
    if exclusion_size == 0:
        exclusion_size = 1  # Ensure at least one frame is excluded from the middle
    
    middle_start = (total_frames - exclusion_size) // 2
    middle_end = middle_start + exclusion_size
    
    valid_indices = list(range(0, middle_start)) + list(range(middle_end, total_frames))
    
    if len(valid_indices) < num_removals:
        raise ValueError("Number of removals exceeds the number of valid indices available.")
    
    chosen_indices = random.sample(valid_indices, k=num_removals)
    new_sequence = sequence.copy()
    
    for idx in sorted(chosen_indices, reverse=True):
        new_sequence = np.delete(new_sequence, idx, axis=2)
    
    return new_sequence
