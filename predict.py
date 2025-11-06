import argparse
import os
import requests
from tqdm import tqdm
import zipfile
from os.path import join, split
import glob as glob
import sys

import torch
import numpy as np

sys.stdout = open(os.devnull, "w")
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

sys.stdout = sys.__stdout__


def download_file(url: str, save_filepath: str):
    """
    Downloading the file to the given path

    Parameters
    ----------
    url: str
        Web url which should be downloaded
    save_filepath:
        output file to save the download in
    """
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(save_filepath, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


def unzip_weights(file_path: str, target_dir: str):
    """
    Unzip the file into the target folder and delete the zip file
    Parameters
    ----------
    file_path: str
        zip file which should be unzipped
    target_dir
        folder where the output should be saved in
    """
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    os.remove(file_path)


def run_nnUNet(input_folder: str, output_folder: str, model_folder: str):
    if torch.cuda.is_available():
        print(f"{stage}: Cuda is available - run nnUNet inference on GPU")
        perform_everything_on_gpu = True
        device = torch.device("cuda")
    else:
        print(f"{stage}: Cuda is not available - run nnUNet inference on CPU")
        perform_everything_on_gpu = False
        device = torch.device("cpu")
    folds = np.arange(0, len(glob.glob(join(model_folder, "fold_*"))))

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=perform_everything_on_gpu,
        verbose=False,
        device=device,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=folds)
    predictor.predict_from_files(input_folder, output_folder)


if __name__ == "__main__":
    stage = "Predicting"

    #dataset_name = "Dataset254_COMPUTING_it4"
    dataset_name = "Dataset255_COMPUTING_it5"
    model_name = "nnUNetTrainerBN__nnUNetPlans__2d"
    model_folder = "nnUNetv2_trained_models"
    weights_url = "https://zenodo.org/records/17541943/files/Dataset255_COMPUTING_it5.zip?download=1"
    parser = argparse.ArgumentParser(
        description="Preprocess images in input directory (renaming + cropping)"
    )

    parser.add_argument("-i", "--input", required=True, help="Input Directory")

    args = parser.parse_args()
    input_dir = args.input
    
    print(f"{stage}: Started")
    # Download Model Weights if the not already exist
    if not os.path.exists(join(model_folder, dataset_name, model_name)):
        print(f"{stage}: Download Model Weights to: {model_folder}")
        os.makedirs(model_folder, exist_ok=True)
        download_file(weights_url, join(model_folder, f"{dataset_name}.zip"))
        print(f"{stage}: Unzip Model Weights")
        unzip_weights(join(model_folder, f"{dataset_name}.zip"), model_folder)
        print(f"{stage}: Weights successfully downloaded")
    else:
        print(f"{stage}: Model Weights already downloaded")

    # Run nnUNet
    run_nnUNet(join(input_dir, "images"), join(input_dir, "masks"), join(model_folder, dataset_name, model_name))

    print(f"{stage}: Done")

