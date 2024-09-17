## Installation App on MacOS operation system

Steps to install the project.

```bash
1. Navigate to the directory where your app will be 
cd /Users/YourUserName/Desktop/

2. Create .command file there with script (see example file RAG.command)

#!/bin/bash

# Initialize conda for shell
eval "$(/opt/homebrew/Caskroom/miniforge/base/bin/conda shell.bash hook)"

# Activate the conda environment
conda activate /opt/homebrew/Caskroom/miniforge/base/envs/nameofmyenv

# Run your Python script
python /Users/YourUserName/Desktop/co-registration_Coronary_Arteries/runnish.py


3. Use the following command to make the .command file executable

chmod +x RAG.command

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
├── run_raw_only.py/ - main run file for co-registration, as input required 4 pathes(2 pathes of original raw sequences and 2 pathes of corresponding segmenntations(optional) if need coregister only raw, use only=True and provide only 2 pathes of raw sequences) of data (exmaple in data/ folder). Default show=True which auto represent result of co-registration as .mp4 fromat video.
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
├── runnish.py/ - application initialization, along with a script to enable the uploading of data for co-registration (limited to .nii.gz format).
├── requirements.txt 
├── environment.yml
├── README.md

'''



