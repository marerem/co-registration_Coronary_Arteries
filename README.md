# Spline Co-Registration of Coronary Artery Imaging

This Spline approuch, with a graphical user interface (GUI), assists in co-registering two sequences of images by aligning one sequence to another based on user-selected checkpoints.

## Motivation

Doctors use a catheter with a camera to examine coronary arteries. Before and after procedures like stent placements, two sets of images are obtained. These sets start and end at different points, have different angles, and contain a different number of frames. Manually aligning these images is challenging and time-consuming. This user-friendly GUI simplifies the alignment process, making it quicker and easier.

## Complexity and Runtime

The algorithm has a complexity of O(n), where n is the number of frames in a sequence. 

### Runtime Performance

- **Average Time per 250 Frames:** 5 seconds

## Getting Started

### Prerequisites

- Python 3.12
- Required Python packages (listed in `requirements.txt`)

### Installation

Steps to install the project.

```bash
# Clone the repository
git clone https://github.com/yourusername/coronary-artery-imaging.git
cd coronary-artery-imaging

# Create and activate a virtual environment
conda env create -f environment.yml
conda activate coronary-artery-imaging

# Install required packages
pip install -r requirements.txt

```

## Porject Structure

```bash
coronary-artery-imaging/
├── data/
│   ├── sequence_0.nii.gz/ - target raw frames sequnce with .nii format
│   ├── ssequence_1.nii.gz/ - moving raw frames sequnce with .nii format
│   ├── sequence_0_seg.nii.gz/ - target segmentation frames sequnce with .nii format (optinal)
│   ├── ssequence_1_seg.nii.gz/ - moving segmentation frames sequnce with .nii format (optinal)
│
│
├── seg/
│   ├── run_seg.py/ - main run file for re co-registration of segmentation frames based on previously defined checkpoints (requires dict.pt with the number of frames at each checkpoint and the corresponding angle).
│       ├── seg_rigid_co.py/ - file containing all functions needed for re co-registration of segmentation frames
│   ├── data_dict_bif_angl.pt/ - dict.pt contains data for each patient  with number of frame  at each checkpoint and the corresponding angle. (belongs to Pre_Post,Pre_Final,Post_Final data set)
│   ├── data_dict_bif_angl_pre_post.pt/ - dict.pt contains data for each patient  with number of frame  at each checkpoint and the corresponding angle. (belongs to P3_MIT data set with only Pre and Stent data)
│   ├── conver_seg.ipynb/ - it's ipynb file where you can run re co-registration segmentation using Jupyter Notebook
│
│
├── run_raw.py/ - main run file for co-registration, as input required 4 pathes(2 pathes of original raw sequences and 2 pathes of corresponding segmenntations(optional)) of data (exmaple in data/ folder).
│   ├── rigid_co.py/ - file containing all functions needed for co-registration.
│   ├── without_seg_fun.py/ - file containing all functions needed for co-registration without segmenation sequnces.
│
│
├── GUI_up_v.py/ - GUI for selecting checkpoint and angle.
│   ├── orientation_algorith.py/ - file containing functions to detect a angl of the checkpoint.
│
│
├── pre_post_flow.ipynb/ - it's ipynb file where you can run co-registration using Jupyter Notebook
│
├── requirements.txt 
├── environment.yml
├── README.md

'''
