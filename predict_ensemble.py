import argparse
import os
import requests
from tqdm import tqdm
import zipfile
from os.path import join, split
import glob as glob
import sys
from skimage import io
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
    predictor.predict_from_files(input_folder, output_folder,save_probabilities=True)


if __name__ == "__main__":
    stage = "Predicting"

    model_folder = "nnUNetv2_trained_models"
    dataset_name = "Dataset255_COMPUTING_it5"
    model_names = ["nnUNetTrainer__nnUNetPlans__2d",
                  "nnUNetTrainerBN__nnUNetPlans__2d",
                  "nnUNetTrainerDA5__nnUNetPlans__2d",
                  "nnUNetTrainerDA5BN__nnUNetPlans__2d"]

    weights_url = "https://syncandshare.desy.de/index.php/s/XXstCCX8Ln6tnCn/download/Dataset254_COMPUTING_it4.zip"
    weights_url = "https://syncandshare.desy.de/index.php/s/GSBioTqfsZp8dzx/download/Dataset255_COMPUTING_it5.zip"
    parser = argparse.ArgumentParser(
        description="Preprocess images in input directory (renaming + cropping)"
    )

    parser.add_argument("-i", "--input", required=True, help="Input Directory")

    args = parser.parse_args()
    input_dir = args.input
    for model_name in model_names:
        print(f"{stage}: Started with {dataset_name} and {model_name}")
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
        run_nnUNet(join(input_dir, "images"), join(input_dir, "ensemble",f"{dataset_name}_{model_name}"), join(model_folder, dataset_name, model_name))

    print(f"{stage}: Ensemble Predictions")
    output_folder = join(input_dir, "masks")
    os.makedirs(output_folder, exist_ok=True)

    folders=os.listdir(join(input_dir, "ensemble",))

    files=os.listdir(join(input_dir, "ensemble",folders[0]))
    files = [file for file in files if file.endswith(".npz")]
    print(f"{len(folders)} folders are used for ensemble prediction")
    for file in tqdm(files):
        output_file = join(output_folder,file.replace(".npz", ".png"))

        if os.path.exists(output_file):
            continue
        pred_SM = None
        for folder in folders:
            nn_sm = np.load(join(input_dir,"ensemble",folder, file))["probabilities"]
            if len(nn_sm.shape) == 4:
                nn_sm = nn_sm.squeeze(1)
            if pred_SM is not None:
                pred_SM += nn_sm
            else:
                pred_SM = nn_sm
        pred_mx = np.argmax(pred_SM, 0).astype(np.uint8)

        io.imsave(output_file, pred_mx,check_contrast=False)
        #cv2.imwrite(join(output_folder, file).replace(".npz", ".png"), pred_mx)
    print(f"{stage}: Done")

