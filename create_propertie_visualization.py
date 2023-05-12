from hereon_2023_computing.DataExtractor import DataExtractor
from os.path import join
import glob
from tqdm import tqdm
import os
import cv2
import numpy as np

from multiprocessing import Pool

import warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
warnings.filterwarnings(action='ignore', message='Degrees of freedom <= 0 for slice')

def process_img(img,mask,out_root):
    ex = DataExtractor(img, mask, True)
    file_name = ex.name

    viz__classes = alpha_blend(ex.img, ex.show_mask(), 0.5)
    viz__pore_size = alpha_blend(ex.img, ex.show_pore_size())
    vize__pore_circularity = alpha_blend(ex.img, ex.show_circularity())
    vize__pore_distance = alpha_blend(ex.img, ex.show_distances())
    vize__mesh = alpha_blend(ex.img, ex.show_mesh())
    vize__mesh_regularity = alpha_blend(ex.img, ex.show_mesh_regularity())

    cv2.imwrite(join(out_root, "Visualization__Classes", file_name), viz__classes)
    cv2.imwrite(join(out_root, "Visualization__Pore_Size", file_name), viz__pore_size)
    cv2.imwrite(join(out_root, "Visualization__Pore_Circularity", file_name),
                vize__pore_circularity)
    cv2.imwrite(join(out_root, "Visualization__Pore_Distance", file_name), vize__pore_distance)
    cv2.imwrite(join(out_root, "Visualization__Mesh", file_name), vize__mesh)
    cv2.imwrite(join(out_root, "Visualization__Mesh_Regularity", file_name), vize__mesh_regularity)

def alpha_blend(img,mask,alpha=1.0,to_RGB=True):
    fig=cv2.addWeighted(img, 1 - alpha, mask, alpha, 0.0)
    bg_map = np.all(mask == [0, 0, 0], axis=2) + np.all(
        mask == [255, 255, 255], axis=2
    )
    fig[bg_map] = img[bg_map]
    if to_RGB:
        fig=cv2.cvtColor(fig, cv2.COLOR_RGB2BGR)
    return fig

if __name__ == "__main__":
    img_root = "/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/Membrane_Extraction_2023_03_08/images"
    mask_root = "/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/Membrane_Extraction_2023_03_08/predictions_it3_253_ensemble"
    out_root = "/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/Membrane_Extraction_2023_03_08/"

    os.makedirs(join(out_root,"Visualization__Classes"),exist_ok=True)
    os.makedirs(join(out_root,"Visualization__Pore_Size"),exist_ok=True)
    os.makedirs(join(out_root,"Visualization__Pore_Circularity"),exist_ok=True)
    os.makedirs(join(out_root,"Visualization__Pore_Distance"),exist_ok=True)
    os.makedirs(join(out_root,"Visualization__Mesh"),exist_ok=True)
    os.makedirs(join(out_root,"Visualization__Mesh_Regularity"),exist_ok=True)

    imgs = glob.glob(join(img_root, "*.png"))
    masks = glob.glob(join(mask_root, "*.png"))
    imgs.sort()
    masks.sort()
    p=Pool(8)
    res=[]
    for img, mask in tqdm(zip(imgs, masks)):
        res.append(
            p.starmap_async(
                process_img,
                ((img, mask, out_root),),
            )
        )
    _ = [re.get() for re in res]
    p.close()
    p.join()


        #break