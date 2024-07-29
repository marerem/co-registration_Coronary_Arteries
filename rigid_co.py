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
from GUI_up_v import *
from torchcubicspline import NaturalCubicSpline
from torchcubicspline import natural_cubic_spline_coeffs
from scipy.interpolate import PchipInterpolator
import math
import cv2
import numpy as np
from scipy.signal import savgol_filter
import numpy as np
from skimage import measure, morphology
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects 
from scipy.ndimage import distance_transform_edt


def co_registration(moving_path_mask='02008Final.nii',target_path_mask='02008Post.nii',moving_path_raw='02008Final_0000.nii',target_path_raw='02008Post_0000.nii'):
    # load segmentation data
    img = nib.load(target_path_mask)
    img_data_pre = img.get_fdata() 
    img = nib.load(moving_path_mask)
    img_data_post = img.get_fdata()

    # load original data 
    pre_data = nib.load(target_path_raw).get_fdata()
    post_data = nib.load(moving_path_raw).get_fdata()
    # sdf

    sdf_ct= distance_transform_edt(torch.tensor(img_data_post).permute(2, 0, 1))  
    sdf_oct= distance_transform_edt(torch.tensor(img_data_pre).permute(2, 0, 1))
    CT_sdf_cpr = torch.tensor(sdf_ct).unsqueeze(0)
    OCT_sdf_cpr = torch.tensor(sdf_oct).unsqueeze(0)
    """
    # Plot areas
    plt.figure(figsize=(10, 6))
    """
    Area_CT_ori = torch.sum((CT_sdf_cpr > 0), dim=(0, 2, 3))
    Area_OCT = torch.sum((OCT_sdf_cpr > 0), dim=(0, 2, 3))
    """
    plt.plot(Area_CT_ori.numpy(), label='Area_CT')
    plt.plot(Area_OCT.numpy(), label='Area_OCT')
    plt.xlabel('Frame')
    plt.ylabel('Area')
    plt.legend()
    plt.title('Original Area')
    plt.show()
    """
    # detect bifurcation
    root = tk.Tk()
    root.title("Peaks Matcher GUI")
    app = PeaksMatcherGUI(master=root, Area_CT=Area_CT_ori, Area_OCT=Area_OCT, CT_image=CT_sdf_cpr>0, OCT_image=OCT_sdf_cpr>0,or_ct=torch.tensor(post_data).permute(2, 0, 1).unsqueeze(0),or_oct=torch.tensor(pre_data).permute(2, 0, 1).unsqueeze(0))
    app.mainloop()
    # Capture organized pairs before closing
    #organized_pairs = organize_matched_pairs(app.highlighted_points)
    organized_pairs,ang_big = organize_matched_pairss(app.saving_orientation)
    print("Last organized pairs:", organized_pairs)
    # Show ang_big in the terminal and ask for confirmation
    """
    while True:
        print("Detected ang_big values:", ang_big)
        response = input("Are these values okay? (yes/no): ").strip().lower()
        if response != 'yes':
            ang_big = input("Please enter the new ang_big values as a comma-separated list: ")
            ang_big = [float(x) for x in ang_big.split(',')]
            print("Updated ang_big values:", ang_big)
    """
    CT_selected_indices = torch.tensor([t[0] for t in organized_pairs])
    OCT_selected_indices = torch.tensor([t[1] for t in organized_pairs])
    idx_OCT_shift = OCT_selected_indices[0]
    idx_CT_shift = CT_selected_indices[0]

    # indices following shift based on first bifurcation
    OCT_selected_indices_shift = OCT_selected_indices - idx_OCT_shift
    print('OCT_selected_indices_shift:must start with 0', OCT_selected_indices_shift)
    CT_selected_indices_shift = CT_selected_indices - idx_CT_shift
    print('CT_selected_indices_shift:must start with 0', CT_selected_indices_shift)
    CT_sdf_cpr = CT_sdf_cpr[:,CT_selected_indices[0]:,...]
    OCT_sdf_cpr = OCT_sdf_cpr[:,OCT_selected_indices[0]:,...]
    # cat same length
    if CT_sdf_cpr.shape[1] < OCT_sdf_cpr.shape[1]:
        OCT_sdf_cpr = OCT_sdf_cpr[:, :CT_sdf_cpr.shape[1], ...]
    else:
        CT_sdf_cpr = CT_sdf_cpr[:, :OCT_sdf_cpr.shape[1], ...]

    #angl of bifurcation
    #RULES
    # 1. Moving image is CT refernce to target OCT
    # 2. Clockwise is negative, counter clockwise is positive
    # 3. Maintain the same SIGN orientation of the moving image for ALL bifurcations (ALL positive or ALL negative)
    # 4. Maximum angle is 180 degrees or 3.14 radians
    print('ang_for each bifurcation:',ang_big)
    angl  = ang_big
    theta_shift = angl


    t = torch.linspace(0, 1,CT_sdf_cpr.shape[1])
    vector = torch.full((CT_sdf_cpr.shape[1],1), float('nan'))

    for i,b in zip(CT_selected_indices_shift,torch.tensor(theta_shift)):
        vector[i] = b
    # parametrized spline
    coeffs = natural_cubic_spline_coeffs(t, vector)
    splines = NaturalCubicSpline(coeffs)
    theta_vec_cubic = splines.evaluate(t)
    t_clean = t[~torch.isnan(vector)]
    vector_clean = vector[~torch.isnan(vector)]
    pchip = PchipInterpolator(t_clean, vector_clean)
    #plt.scatter(CT_selected_indices_shift, angl, label='Original', color='red')
    #plt.plot(theta_vec_cubic, label='Cubic', color='green')
    #plt.plot(pchip(t), label='Pchip', color='orange')
    #plt.legend()
    # add end point by cubic spline
    arr = pchip(t)  # Get the array
    arr[CT_selected_indices_shift[-1]:] = theta_vec_cubic[CT_selected_indices_shift[-1]:].reshape(-1)  # Modify the slice  

    #plt.scatter(CT_selected_indices_shift, angl, label='Original', color='red')
    #plt.plot(pchip(t), label='Pchip', color='orange')
    #plt.plot(theta_vec_cubic, label='Cubic', color='green')
    #plt.plot(arr, label='Final', color='blue')

    ct_data_or  = torch.tensor(post_data[:,:,idx_CT_shift:CT_sdf_cpr.shape[1]+idx_CT_shift]).permute(2, 0, 1).unsqueeze(0)
    oct_data_or = torch.tensor(pre_data[:,:,idx_OCT_shift:OCT_sdf_cpr.shape[1]+idx_OCT_shift]).permute(2, 0, 1).unsqueeze(0)

    ph_or = ridgit_register(ct_data_or[0], torch.tensor(np.array(arr).reshape(-1,1)))
    ph = ridgit_register(CT_sdf_cpr[0], torch.tensor(np.array(arr).reshape(-1,1)))
    #OCT_centers_smoothedmin, CT_centers_smoothedmin =  find_center_sdf_max(OCT_sdf_cpr,ph.unsqueeze(0))
    
    ct_circl = []
    for i in range(CT_sdf_cpr.shape[1]):
        orientation,centroid = center_circle(ph.unsqueeze(0)[0,i,...].detach().numpy()>0)
        ct_circl.append(centroid)

    oct_circl= []
    for i in range(OCT_sdf_cpr.shape[1]):
        orientation,centroid = center_circle(OCT_sdf_cpr[0,i,...].detach().numpy()>0)
        oct_circl.append(centroid)

    ph_or_circle = rotation(ph_or.unsqueeze(0),oct_circl,ct_circl)
    #ph_or_smooth_min = rotation(ph_or.unsqueeze(0),OCT_centers_smoothedmin,CT_centers_smoothedmin)
    #ph_translation_smooth_min = rotation(ph.unsqueeze(0),OCT_centers_smoothedmin,CT_centers_smoothedmin)
    ph_circle = rotation(ph.unsqueeze(0),oct_circl,ct_circl)
    
    # Interactive slider
    def show_images(i):
        fig, axs = plt.subplots(2, 3, figsize=(15,5))
        axs[0,0].set_title('Target Original')
        axs[0,0].imshow(OCT_sdf_cpr[0,i,...].detach().numpy()>0)
        axs[0,0].plot(oct_circl[i][1],oct_circl[i][0], marker='o', color="red")

        axs[0,1].set_title('Moving Original')
        axs[0,1].imshow(CT_sdf_cpr[0,i,...].detach().numpy()>0)
        axs[0,1].plot(ct_circl[i][1],ct_circl[i][0], marker='o', color="red")

        axs[0,2].set_title('Moving Co-register')
        axs[0,2].imshow(ph_circle[i])
        axs[0,0].plot(oct_circl[i][1],oct_circl[i][0], marker='o', color="red")
        axs[0,1].plot(ct_circl[i][1],ct_circl[i][0], marker='o', color="blue")

        axs[1,0].set_title('Target Original')
        axs[1,0].imshow(oct_data_or[0,i,...].detach().numpy())

        axs[1,1].set_title('Moving Original')
        axs[1,1].imshow(ct_data_or[0,i,...].detach().numpy())

        axs[1,2].set_title('Moving Co-register')
        axs[1,2].imshow(ph_or_circle[i])
    frame_slider = IntSlider(min=0, max=CT_sdf_cpr.shape[1]-1, step=1, value=0)
    interact(show_images, i=frame_slider)
    """
        response = input("Are okey with FINAL? (yes/no): ").strip().lower()
        if response != 'yes':
            ang_big = input("Please enter the new ang_big values as a comma-separated list: ")
            ang_big = [float(x) for x in ang_big.split(',')]
            print("Updated ang_big values:", ang_big)
            continue
        else:
    """
    return oct_data_or, ph_or_circle, ang_big, organized_pairs







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
"""
if __name__ == "__main__":
    patient = '03008'
    #d = torch.load('data_dict_bif_angl.pt')
    d = {}
    d[patient] = {}
    ############################ PRE-FINAL ############################
    oct_data_or, ph_or_circle, ang_big, organized_pairs = co_registration(moving_path_mask=patient+'Final.nii',target_path_mask=patient+'Pre.nii',moving_path_raw=patient+'Final_0000.nii',target_path_raw=patient+'Pre_0000.nii')


    d[patient]['bif_(PreTarget_StentMoving)'] = [organized_pairs,ang_big]
    d[patient]['bif_(PreTarget_StentMoving)'][0]
    # substract first tuple from all tuples in the list
    l = [(x[0]-d[patient]['bif_(PreTarget_StentMoving)'][0][0][0], x[1]-d[patient]['bif_(PreTarget_StentMoving)'][0][0][1]) for x in d[patient]['bif_(PreTarget_StentMoving)'][0]]
    fixed_image = oct_data_or.detach().numpy().reshape(oct_data_or.shape[-1], oct_data_or.shape[-1], oct_data_or.shape[1])
    moving_image = np.array(ph_or_circle)
    bil = []
    for i in range(len(l)-1):
        
        lengthf = fixed_image[:,:,l[i][1]:(l[i+1][1])].shape[-1]
        lengthm = moving_image[:,:,l[i][0]:(l[i+1][0])].shape[-1]

        frames = moving_image[:,:,l[i][0]:(l[i+1][0])]
        if lengthf>lengthm:
            num_duplicates = lengthf - lengthm 
            sq = duplicate_frames(frames, num_duplicates=num_duplicates, exclusion_fraction=0.2)
            bil.append(sq)
        else:
            num_removals = lengthm - lengthf
            sq = remove_frames(frames, num_removals=num_removals, exclusion_fraction=0.2)
            bil.append(sq)
        if i == len(l)-1:
            frames = moving_image[:,:,l[i+1][0]:]
            bil.append(frames)
    br = np.concatenate((bil),axis=2)

    def show_images(i):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].imshow(fixed_image[...,i])
        axs[0].set_title('Target(Pre)')
        
        axs[1].imshow(moving_image[...,i])
        axs[1].set_title('Moving(Stent)')
        
        axs[3].imshow(br[...,i])
        axs[3].set_title('Resample random removel\duplicate')

        plt.show()

    # Interactive slider
    frame_slider = IntSlider(min=0, max=moving_image.shape[1]-1, step=1, value=0)
    interact(show_images, i=frame_slider)

    br = nib.Nifti1Image(br, np.eye(4))
    nib.save(br, patient'Final_reg.nii.gz')

    fixed_image = nib.Nifti1Image(fixed_image, np.eye(4))       
    nib.save(fixed_image, 'Pre_target_Final.nii.gz')

"""
