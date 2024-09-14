@echo off
REM Set environment variable
set KMP_DUPLICATE_LIB_OK=TRUE

REM Activate the Conda environment
call C:\ProgramData\Anaconda3\condabin\conda.bat activate GNN

REM Navigate to the directory where your script is located
cd /d C:\Users\SUBJECT_6\Desktop\Mariia\co-reg_segmentation\co-reg

REM Run the Python script
python runnish.py

REM Keep the terminal open to view output
pause

