import glob
import os
import shutil
from os.path import join
from tqdm import tqdm
import nibabel as nib
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from PIL import Image
import numpy as np
from matplotlib import cm
import cv2
from skimage import io

def viz_color_encoding(labels: list[str], cmap):
    # litte helper function to visualize the color-class encoding
    width = 350
    height = 60
    num = len(labels)
    img = np.zeros((num * height, width, 3), np.uint8)

    for i, label in enumerate(labels):

        img[i * height : (height + 1) * height, :] = cmap[i]
        cv2.putText(
            img,
            str(i) + ". " + label,
            (10, (i) * height + int(height * 0.75)),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 255, 255),
            2,
        )
    for index in range(1, num):
        cv2.line(img, (0, index * height), (width, index * height), (255, 255, 255))
    return img


def viz_mask(img, cmap):
    w, h = img.shape
    mask_np = np.zeros((w, h, 3), dtype=np.uint8)
    for i in np.unique(img):
        x, y = np.where(img == i)
        # print(cm[int(i)])
        mask_np[x, y] = cmap[int(i)]
    return mask_np


if __name__ == "__main__":
    nnUNet_raw = "/media/l727r/data/nnUNet/nnUNetv2_raw"
    nnUNet_preprocessed = "/media/l727r/data/nnUNet/nnUNet_preprocessed"

    # Define Dataset properties
    '''
    #Iteration 0
    root = "/home/l727r/Desktop/HEREON_2022_COMPUTING/Data_labeled"
    imgs_folders = ["images", "images_partly_labeled"]
    masks_folder = ["masks", "masks_partly_labeled"]
    img_dtype = ".tif"
    mask_dtype = ".nii.gz"
    iteration = 0
    '''

    # Iteration 1
    root = "/home/l727r/Desktop/HEREON_2022_COMPUTING/Data_labeled_it3"
    imgs_folders = ["images"]
    masks_folder = ["masks"]
    img_dtype = ".tif"
    mask_dtype = ".png"#".nii.gz"
    iteration = 3

    # Class and CM definition
    # classes = [
    #     "background",
    #     "covered_areas",
    #     "open_pore",
    #     "dead-end_pore",
    #     "closed_pore",
    #     "artifact_crater",
    #     "artifact_blob",
    #     "artifact_dust",
    #     "ignore",
    # ]
    # num_classes = len(classes)
    # colomap = "viridis"
    # colormap = cm.get_cmap(colomap, num_classes - 1)
    # cmap = [[0, 0, 0]]
    # for i in range(0, num_classes):
    #     cmap.append(np.array(colormap(i))[0:3] * 255)
    classes = [
        "background",
        "covered_areas",
        "open_pore",
        "closed_pore",
        "internal_structure",
        "artifact_blob",
        "artifact_dust",
        "ignore",
    ]
    cmap=cmap_napari=[
        [0,0,0],         # 0 - background
        [120,37,7],     # 1 - covered_area
        [92,214,249],   # 2 - open_pore
        [146,137,233],  # 3 - closed_pore
        [156,97,48],    # 4 - internal_structure
        [72,58,160],    # 5 - artifact_blob
        [171,236,139],  # 6 - artifact_dust
        [180,125,189],  # 7 - ignore
    ]
    # cmap = [c[::-1] for c in cmap]

    # Define nnUNetv2 Folders
    TaskID = 250 + iteration
    DatasetName = f"Dataset{TaskID}_COMPUTING_it{iteration}"
    output_folder = os.path.join(nnUNet_raw, DatasetName)

    os.makedirs(join(output_folder, "imagesTr"), exist_ok=True)
    os.makedirs(join(output_folder, "labelsTr"), exist_ok=True)
    os.makedirs(join(output_folder, "labelsViz"), exist_ok=True)
    os.makedirs(join(output_folder, "labelsVizOver"), exist_ok=True)

    # Collect Images and Masks
    imgs = []
    for img_folder in imgs_folders:
        imgs += glob.glob(join(root, img_folder, "*" + img_dtype))
    imgs.sort()

    masks = []
    for mask_folder in masks_folder:
        masks += glob.glob(join(root, mask_folder, "*"))
    masks.sort()

    print("{}: {} Images and {} Masks are found".format(DatasetName, len(imgs), len(masks)))

    # Start Conversion
    for img_file, mask_file in tqdm(zip(imgs, masks)):
        # Open Image file and save as .png
        img_name = img_file.rsplit("/", 1)[1].replace(img_dtype, ".png").replace("(","").replace(")","")
        #img_pil = Image.open(img_file)#.convert('RGB')
        img=io.imread(img_file)
        img_pil=Image.fromarray(img).convert("L")
        img_pil.save(join(output_folder, "imagesTr", img_name).replace(".png", "_0000.png"))

        # Open Mask file and save as .png
        #if mask_dtype==".nii.gz":
        if mask_file[-7:]==".nii.gz":
            mask = nib.load(mask_file).get_fdata().T
            mask_pil = Image.fromarray(mask).convert("L")
            mask_pil.save(join(output_folder, "labelsTr", img_name))
        #elif mask_dtype == ".png":
        if mask_file[-4:] == ".png":
            mask = io.imread(mask_file)
            mask_pil = Image.fromarray(mask).convert("L")
            mask_pil.save(join(output_folder, "labelsTr", img_name))

        # Create and Save Mask Visualization
        mask_viz = viz_mask(mask, cmap)
        mask_pil = Image.fromarray(mask_viz)
        mask_pil.save(join(output_folder, "labelsViz", img_name))

        # Create and Save Overlay for Image and Mask
        img_ov = Image.blend(img_pil.convert("RGB"), mask_pil, 0.5)
        img_ov.save(join(output_folder, "labelsVizOver", img_name))

    # Create and Save Label Viz
    enc = viz_color_encoding(classes, cmap)
    Image.fromarray(enc).save(join(output_folder, "Labels.png"))

    # Generate Dataset Json
    generate_dataset_json(
        output_folder=output_folder,
        #channel_names={0: "zscore"},
        channel_names={0: "rgb_to_0_1"},
        labels={classes[i]: i for i in range(0, len(classes))},
        num_training_cases=len(glob.glob(join(output_folder, "imagesTr", "*.png"))),
        file_ending=".png",
        dataset_name=DatasetName,
    )
