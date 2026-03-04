To run the code:
Step 1 :
cd ...\STMproject\video_stm_activity
Step 2:
.\.venv\Scripts\Activate.ps1
Step 3: 
# Single video inference (STM)
python -m src.run_infer --video .\data\videos\phone1.mp4 --memory stm

# Single video inference (NONE)
python -m src.run_infer --video .\data\videos\phone1.mp4 --memory none

# Whole experiment, all VLMs
python -m src.run_experiment --video main_activity.mp4 --modes stm none

# One video experiment, one VLM
python -m src.run_experiment --models llava15_7b --modes stm none --video phone1.mp4 --debug


=============Requirements=========
torch
torchvision
transformers>=4.41.0
accelerate
bitsandbytes
opencv-python
pandas
pyyaml
tqdm
scikit-learn
Pillow
matplotlib
seaborn
pandas
