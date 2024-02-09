# COMPUTING - Membrane Property Extraction

## 1. Installation

```shell
# Download Repository
git clone https://codebase.helmholtz.cloud/hi-dkfz/applied-computer-vision-lab/collaborations/hereon_2023_computing.git
cd hereon_2023_computing
# (Optional) - Create a new conda environment
conda create --name computing python=3.10
conda activate computing
# Install all required packages
pip install -r requirements.txt
```

## 2. Membrane Property Extraction

### How to run
Follow the steps (2.1 - 2.3) to run the complete pipeline step by step. 
Alternatively adopt the path and execute runner.sh script (only fo ubuntu) and the scripts will be executed automatically.
On windows you have to do it step by step like shown below.

````shell
# For ubuntu (couldnt make it run on windows)
bash runner.sh 
# For Windows do it step by step
python preprocessing.py -i="C:\Users\l727r\Desktop\Computing\PS-P4VP" -o="C:\Users\l727r\Desktop\Computing\PS-P4VP_props"
python predict.py -i="C:\Users\l727r\Desktop\Computing\PS-P4VP_props"
python propertie_extraction.py -i="C:\Users\l727r\Desktop\Computing\PS-P4VP_props"
````


### 2.1 Preprocessing

The preprocessing can be executed as shown below. 
During preprocessing, the black bars at the bottom of the images are cut off, the files are renamed and converted to png images.
- **Why Renaming?** 
A lot of the files contain special chars (e.g. "µ", "^", "+", "%"), this can make problems when handling with file paths, therefore these chars get removed.
Additionally, a '_0000' postfix is added which is required for nnUNet.
The mapping from original to the new file names will be saved in 'name_mapping.csv'.
- **Why Conversion from .tif to .png?** 
nnUNet was trained with .png images and therefore requires the data to be in .png format.

```shell
python preprocessing.py -i=root_raw -o=root_props
```

- **root_raw:** 
Path to the folder which contains the images from which the parameters should be extracted. 
All .tif files in the directory + all subdirectories will be used (files named "!black.tif" will be excluded).
- **root_props:** 
Path to the folder in which all the processed data and later all the extracted membrane properties will be saved in.

### 2.2 Predicting

How to run the prediction script is shown below. During Predicting a segmentation mask is created for each image.
If a GPU is available the prediction is performed on the GPU, otherwise the CPU is used. 
Inference on CPU will take longer compared to the GPU.
When executing the script for the first time, the model weights of nnUNet will be downloaded (~2.7 GB) and saved into the 'nnUNetv2_trained_models' folder.
This has to be done only once.

```shell
python predict.py -i=root_props
```
- **root_props:** 
Path to the folder which contains the processed data from the previous stage. 
The predictions will be saved into this directorys


### 2.3 Property Extraction

How to run the script to extract the membrane properties is shown below. 
A 'membrane_properties.csv' will be created which contains all properties for each image.
Additionally some folders will be created which contain visualizations for the Classes, Mesh, Mesh_Regularity, Pore_Size, Pore_Distance, Pore_Circularity.

```shell
python propertie_extraction.py -i=root_props
```
- **root_props:** 
Path to the folder which contains the processed data from the previous stages. 
The membrane properties and all visualizations will be saved here.

## 3. Limitations
- Currently, segmentation and membrane extraction works only with images which have a magnification of 50.00KX (Image Pixel Size = 2.233 nm).


# Acknowledgements

<p align="left">
  <img src="Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="Logos/DKFZ_Logo.png" width="500"> 
</p>

This Repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/).