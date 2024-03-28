import argparse
import os
from os.path import join, split
from tqdm import tqdm

import pandas as pd
from skimage import io
from scipy.ndimage import zoom

BASE_MAGNIFICATION = 50


def find_files(directory, dtype):
    files_dtype = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(dtype):
                files_dtype.append(join(root, file))
    return files_dtype


def replace_chars(
    string, delete=["(", ")", ";", " ", "=", "µ", "^", "+", "%"], replace=[(".tif", "_0000.png")]
):
    for d in delete:
        string = string.replace(d, "")
    for r in replace:
        string = string.replace(r[0], r[1])
    return string


def create_name_mapping(file_names: str):
    file_names = [split(file)[-1] for file in file_names]

    file_renames = []
    for file_name in file_names:
        file_renames.append(replace_chars(file_name))

    df = pd.DataFrame()
    df["original_name"] = file_names
    df["new_name"] = file_renames

    return df


def process_image_file(file, cutoff, magn):
    image = io.imread(file)[:-cutoff, :]
    scale_factor = BASE_MAGNIFICATION/magn
    image = zoom(image, (scale_factor, scale_factor), order=3)
    return image


if __name__ == "__main__":
    stage = "Preprocessing"

    parser = argparse.ArgumentParser(
        description="Preprocess images in input directory (renaming + cropping)"
    )

    parser.add_argument("-i", "--input", required=True, help="Input Directory")
    parser.add_argument("-o", "--output", required=True, help="Output Directory")
    parser.add_argument("-m", "--magnification", required=True, help="magnification")
    parser.add_argument("--cutoff", type=int, default=60, help="Cutoff Size (default: 60)")

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    magnification_data = args.magnification
    cutoff = args.cutoff

    df_raw = pd.read_excel(magnification_data, engine="openpyxl")
    df = df_raw[["Image Name (file name on fileserver)", "Magnification as number"]]
    df = df.rename(
        columns={
            "Image Name (file name on fileserver)": "file_name",
            "Magnification as number": "magnification",
        }
    )

    # Setup all needed folders
    print(f"{stage}: Started")
    print(f"{stage}: Data will be stored in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(join(output_dir, "images"), exist_ok=True)

    # Find all .tif files in the dir + subdirs
    print(f"{stage}: Search for files in: {input_dir}")
    image_files = find_files(input_dir, ".tif")
    image_files = [image_file for image_file in image_files if "!black.tif" not in image_files]
    print(f"{stage}: Found {len(image_files)} Files")

    # Create file mapping
    df_mapping = create_name_mapping(image_files)
    df_mapping.to_csv(join(output_dir, "name_mapping.csv"), index=False)
    print(f"{stage}: Save Name Mapping to: {join(output_dir, 'name_mapping.csv')}")

    # Process files
    for image_file in tqdm(image_files, desc=f"{stage}: Process image files"):
        file_name = os.path.split(image_file)[-1].replace(".tif", "")
        if df['file_name'].isin([file_name]).any():
            magn = df.loc[df["file_name"] == file_name, "magnification"].values[0]
        else:
            print(f"WARNING: File {file_name} does not exist in your maginifaction data. Default values are used")
            magn=BASE_MAGNIFICATION
        image = process_image_file(image_file, cutoff, magn)
        output_file = replace_chars(split(image_file)[-1])
        io.imsave(join(output_dir, "images", output_file), image)
    print(f"{stage}: Done")
