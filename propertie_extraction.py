import argparse
import os
from os.path import join
import glob as glob
import pandas as pd
from tqdm import tqdm

from utils.DataExtractor import DataExtractor,cmap_classes,class_dict
import cv2
import numpy as np
import itertools
from multiprocessing import Pool

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
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def get_propertie_names():
    # Class Properties
    class_names = ["covered_area", "internal_structure", "artifact_blob", "artifact_dust"]
    class_parameters = [
        "area__covered",
        "instances__count",
        "instances__mean_size",
        "instances__std_size",
    ]
    class_properties = [
        f"{cn}__{cp}" for cn, cp in list(itertools.product(class_names, class_parameters))
    ]

    # Mesh Properties
    mesh_properties = [
        # "vertices__count",
        "edges__count",
        "edges__mean_length",
        "edges__std_length",
        "triangles__mean_size",
        "triangles__std_size",
        "triangles__mean_regularity",
        "triangles__std_regularity",
    ]
    mesh_properties = [f"mesh__{mp}" for mp in mesh_properties]

    # Pore Properties
    pore_types = [
        "all_pores",
        "open_pores",
        "closed_pores",
    ]
    pore_parameters = class_parameters + [
        "instances__mean_diameter",
        "instances__std_diameter",
        "instances__mean_circularity",
        "instances__std_circularity",
    ]
    pore_properties = [
        f"{pt}__{pp}" for pt, pp in list(itertools.product(pore_types, pore_parameters))
    ]
    return mesh_properties + pore_properties + class_properties


def alpha_blend(img, mask, alpha=1.0, to_RGB=True):
    fig = cv2.addWeighted(img, 1 - alpha, mask, alpha, 0.0)
    bg_map = np.all(mask == [0, 0, 0], axis=2) + np.all(mask == [255, 255, 255], axis=2)
    fig[bg_map] = img[bg_map]
    if to_RGB:
        fig = cv2.cvtColor(fig, cv2.COLOR_RGB2BGR)
    return fig


def process_img(img, mask, root,name_mapping, save_viz=True):
    ex = DataExtractor(img, mask, True)
    file_name = ex.name

    if save_viz:
        viz__classes = alpha_blend(ex.img, ex.show_mask(), 0.7)
        viz__pore_size = alpha_blend(ex.img, ex.show_pore_size())
        vize__pore_circularity = alpha_blend(ex.img, ex.show_circularity())
        vize__pore_distance = alpha_blend(ex.img, ex.show_distances())
        vize__mesh = alpha_blend(ex.img, ex.show_mesh())
        vize__mesh_regularity = alpha_blend(ex.img, ex.show_mesh_regularity())

        cv2.imwrite(join(root, "Visualization__Classes", file_name), viz__classes)
        cv2.imwrite(join(root, "Visualization__Pore_Size", file_name), viz__pore_size)
        cv2.imwrite(join(root, "Visualization__Pore_Circularity", file_name), vize__pore_circularity)
        cv2.imwrite(join(root, "Visualization__Pore_Distance", file_name), vize__pore_distance)
        cv2.imwrite(join(root, "Visualization__Mesh", file_name), vize__mesh)
        cv2.imwrite(join(root, "Visualization__Mesh_Regularity", file_name), vize__mesh_regularity)

    file_name=ex.name
    org_file_name=name_mapping.loc[name_mapping["new_name"]==file_name,"original_name"].values[0]

    joint_props = {
        "file_name": file_name,
        "org_file_name":org_file_name,
        **ex.get__mesh__properties(),
        **ex.get__all_pores__properties(),
        **ex.get__open_pores__properties(),
        **ex.get__closed_pores__properties(),
        **ex.get__covered_area__properties(),
        **ex.get__internal_structure__properties(),
        **ex.get__artifact_blob__properties(),
        **ex.get__artifact_dust__properties(),
    }
    return joint_props


if __name__ == "__main__":
    stage = "Extraction"
    parser = argparse.ArgumentParser(description='Preprocess images in input directory (renaming + cropping)')

    parser.add_argument('-i', '--input', required=True, help='Input Directory')

    args = parser.parse_args()
    root=args.input

    print(f"{stage}: Started")

    save_viz = True
    img_root = join(root, "images")
    mask_root = join(root, "masks")

    imgs = glob.glob(join(img_root, "*.png"))
    masks = glob.glob(join(mask_root, "*.png"))
    imgs.sort()
    masks.sort()
    if save_viz:
        os.makedirs(join(root, "Visualization__Classes"), exist_ok=True)
        os.makedirs(join(root, "Visualization__Pore_Size"), exist_ok=True)
        os.makedirs(join(root, "Visualization__Pore_Circularity"), exist_ok=True)
        os.makedirs(join(root, "Visualization__Pore_Distance"), exist_ok=True)
        os.makedirs(join(root, "Visualization__Mesh"), exist_ok=True)
        os.makedirs(join(root, "Visualization__Mesh_Regularity"), exist_ok=True)

        img_col=viz_color_encoding(list(class_dict.values()),cmap_classes)
        cv2.imwrite(join(root,"Labels.png"), img_col)

    df_columns = ["file_name",'org_file_name'] + get_propertie_names()
    df = pd.DataFrame(columns=df_columns)
    name_mapping = pd.read_csv(join(root,"name_mapping.csv"))
    print(name_mapping.shape)
    p = Pool(8)
    res = []
    for img, mask in zip(imgs, masks):
        res.append(
            p.starmap_async(
                process_img,
                ((img, mask, root, name_mapping,save_viz),),
            )
        )
    data = [re.get() for re in tqdm(res)]

    for d in data:
        df = pd.concat([df, pd.DataFrame(d)], ignore_index=True)
    df.to_csv(join(root, "membrane_properties.csv"), index=False)  # with redundancies
    print(f"{stage}: Save Membrane Properties to: {join(root, 'membrane_properties.csv')}")
    print(f"{stage}: Done")