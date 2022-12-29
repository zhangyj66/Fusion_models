### ENVIRONMENT ###
python 3.7
torch 1.8.1
torchvision
numpy
tqdm
timm
math
cv2
os
sys

### FUSION MODULES
In this code, we have introduced 3 different fusion models in the fusion_models/ folder.
The input and output forms, hyperparameters, etc. can be modified according to specific tasks.

### BACKBONE
In this code, we create a very simple FCN as the backbone to show the code, which has 4 layers of resnet34.
And, for simplicity, we use the same backbone network for RGB and Depth modalities. 

### MODEL
We briefly show how to plug the fusion model into the backbone network in model.py.
Users can decide where and how many fusion models to insert by modifying model.py.

### HOW TO USE
1. Config
   Edit config file in configs.py, including dataset and train/evaluate settings.
2. train:
   python train.py
3. test (predict the segmentation rsults):
   python test.py
