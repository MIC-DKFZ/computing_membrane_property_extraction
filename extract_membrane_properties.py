from hereon_2023_computing.DataExtractor import DataExtractor
from os.path import join
import glob
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide')
warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
warnings.filterwarnings(action='ignore', message='Degrees of freedom <= 0 for slice')

if __name__ == "__main__":
    img_root = "/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/Membrane_Extraction_2023_03_08/images"
    mask_root = "/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/Membrane_Extraction_2023_03_08/predictions_it3_253_ensemble"
    out_root = "/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/Membrane_Extraction_2023_03_08/"

    imgs = glob.glob(join(img_root, "*.png"))
    masks = glob.glob(join(mask_root, "*.png"))
    imgs.sort()
    masks.sort()

    import itertools

    # Class Properties
    class_names = ["covered_area", "internal_structure", "artifact_blob", "artifact_dust"]
    class_parameters = [
        "area__covered",
        #"pixel__pct",
        "instances__count",
        "instances__mean_size",
        "instances__std_size",
    ]
    class_properties = [
        f"{cn}__{cp}" for cn, cp in list(itertools.product(class_names, class_parameters))
    ]

    # Mesh Properties
    mesh_properties = [
        #"vertices__count",
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
    df_columns = ["file_name"]+mesh_properties+pore_properties + class_properties
    df = pd.DataFrame(columns=df_columns)

    for img, mask in tqdm(zip(imgs, masks)):
        ex = DataExtractor(img, mask, True)

        joint_props={
            "file_name":ex.name,
            **ex.get__mesh__properties(),
            **ex.get__all_pores__properties(),
            **ex.get__open_pores__properties(),
            **ex.get__closed_pores__properties(),
            **ex.get__covered_area__properties(),
            **ex.get__internal_structure__properties(),
            **ex.get__artifact_blob__properties(),
            **ex.get__artifact_dust__properties()
                     }
        df=pd.concat([df,pd.DataFrame([joint_props])],ignore_index=True)
    df.to_csv(join(out_root,"membrane_properties.csv"),index=False) # with redundancies
    #df.to_csv("membrane_properties.csv") # without redundancies