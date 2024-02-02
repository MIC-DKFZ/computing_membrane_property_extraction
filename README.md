Computing
0. Installation
1. Preprocessing
   2. crop the images
   3. rename them
   4. convert to png
4. Segmentation
   5. nnUNet
6. Extract Parameter + Visualize


```shell
python preprocessing.py -i input_dir -o outputdir
nnUNetv2_predict -i outputdir+/images -o outputdir+/masks
python propertie_extraction -i input_dir
python postprocessing -i outputdir
```

pip install git+https://github.com/MIC-DKFZ/nnUNet.git

