import requests
from tqdm import tqdm
import os
import zipfile

def unzip_file(file_path: str, target_dir: str):
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

def download_file(url,save_filepath,overwrite=False, iszip=True):


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


if __name__ == "__main__":
    print(os.getcwd())
    url="https://syncandshare.desy.de/index.php/s/XXstCCX8Ln6tnCn/download/Dataset254_COMPUTING_it4.zip"
    save_dir="test/"