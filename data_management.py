import glob
import os
from os.path import join, split
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import shutil

def extract_data_from_raw(input_dir: str, output_dir: str, crop_cutoff: int = 60):

    # Find all tifs inside the folders
    files = glob.glob(join(input_dir, "*", "S*.tif"))
    print(f"{len(files)} Files are found in total")

    # ignore these folders since images are duplicates
    ignore_folders = ["database10 new_files", "database10 renamed_files_to_add_new"]
    files = [file for file in files if not any([folder in file for folder in ignore_folders])]
    print(f"{len(files)} Files remain after ignoring some folders")

    os.makedirs(output_dir, exist_ok=True)
    for file in tqdm(files):
        file_name = split(file)[-1]

        # Read Image, crop it and save it as png in output folder
        img = cv2.imread(file, -1)[:-crop_cutoff, :,0]
        cv2.imwrite(join(output_dir, file_name.replace(".tif", ".png")), img)


def create_name_mapping(input_dir: str, output_dir: str):
    chars_to_replace = ["(", ")", ";", " ", "=", "µ", "^", "+", "%"]
    files = glob.glob(join(input_dir, "*.png"))
    file_names = [split(file)[-1] for file in files]

    file_renames = []
    for file_name in file_names:
        for r in chars_to_replace:
            file_name = file_name.replace(r, "")
        file_name = file_name.replace(".png", "_0000.png")
        file_renames.append(file_name)

    df = pd.DataFrame()
    df["original_name"] = file_names
    df["new_name"] = file_renames
    df.to_csv(join(output_dir, "name_mapping.csv"), index=False)


def rename_folder(input_folder: str, name_mapping_csv: str):

    df = pd.read_csv(name_mapping_csv)
    files = os.listdir(input_folder)

    if sum(df["original_name"] == files) > sum(df["new_name"] == files):
        source_names = "original_name"
        target_names = "new_name"
    else:
        source_names = "new_name"
        target_names = "original_name"
    for file in tqdm(files):
        new_name = list(df[df[source_names] == file][target_names])[0]
        os.rename(join(input_folder, file), join(input_folder, new_name))

def copy_predictions(prediction_path,image_path,output_path):
    '''
    Update Database, copy all predictions which still have an image in the new database
    :param prediction_path:
    :param image_path:
    :param output_path:
    :return:
    '''
    os.makedirs(output_path,exist_ok=True)
    files=os.listdir(image_path)
    for file in tqdm(files):
        if os.path.exists(join(prediction_path,file.replace("_0000",""))):
            shutil.copy(join(prediction_path,file.replace("_0000","")),join(output_path,file.replace("_0000","")))

def find_missing_predictions(prediction_path,image_path,output_path):
    '''
    Update Database, find all images in new database which dont have a prediction
    :param prediction_path:
    :param image_path:
    :param output_path:
    :return:
    '''
    os.makedirs(output_path,exist_ok=True)
    files=os.listdir(image_path)
    for file in tqdm(files):
        if not os.path.exists(join(prediction_path,file.replace("_0000",""))):
            shutil.copy(join(image_path,file),join(output_path,file))


if __name__ == "__main__":
    # Extracting the Raw Data
    # input_dir="/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/Raw/images"
    # output_dir="/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/images"
    # extract_data_from_raw(input_dir,output_dir)

    # Create Name Mapping
    # input_dir="/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/images"
    # output_dir="/home/l727r/Desktop/HEREON_2023_COMPUTING/Data"
    # create_name_mapping(input_dir,output_dir)

    # Rename images in Folder from original names to nnUNet conform names (or the other way around)
    name_mapping = "/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/name_mapping.csv"
    #input_folder = "/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/Membrane_Extraction_2023_03_08/images"
    input_folder = "/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/Membrane_Extraction_2023_03_08/predictions_it3_253_ensemble"
    rename_folder(input_folder, name_mapping)

    # Copy all Predictions which have an prediction in the updated dataset
    # image_path="/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/images"
    # prediction_path="/home/l727r/Documents/cluster-data/COMPUTING_imgs_all/it3_253_ensemble"
    # output_path="/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/predictions_it3_253_ensemble"
    # copy_predictions(prediction_path, image_path, output_path)

    # Copy all Images which have no prediction in the updated dataset
    # image_path="/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/images"
    # prediction_path="/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/predictions_it3_253_ensemble"
    # output_path="/home/l727r/Documents/cluster-data/COMPUTING_imgs_all/input_imgs_missing"
    # find_missing_predictions(prediction_path, image_path, output_path)