# App for co-registering image sequences using a novel spline-based approach

This app enhances processing speed by 40x, allowing the co-registration of a sequence of 300 frames in just 5 seconds. It is compatible with Windows and MacOS and is powered by a novel spline-based algorithm developed at MIT’s Edelman Lab. Originally designed for medical hardware, it efficiently aligns videos with rotational, transverse, and displacement adjustments.

https://github.com/user-attachments/assets/892990fa-c3a7-40d5-af71-7840aaff4f9b


## Getting Started

### Prerequisites

- Python 3.12
- Required Python packages (listed in `requirements.txt`)
### Operating System

Depending on your operating system, we provide two separate branches:

- **macOS and Linux**: [macOS/Linux Branch](https://github.com/marerem/co-registration_Coronary_Arteries/tree/macOS-app)
- **Windows**: [Windows Branch](https://github.com/marerem/co-registration_Coronary_Arteries/tree/app_final_check)



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
