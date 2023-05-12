import cv2
import numpy as np
from hereon_2023_computing.DataExtractor import DataExtractor
import glob
from os.path import join
import os
class Visualizer:
    def __init__(self, imgs, masks, window_name="Window",verbose=False):
        self.imgs = imgs
        self.masks = masks
        self.verbose=verbose
        self.window_name = window_name

        self.modes = ["Mask", "Components","Classes", "Circles", "Circularity", "Mesh", "Distance","Triangle", "PoreSize"]
        self.mode_txt = "Mode:\n"
        for i, mode in enumerate(self.modes):
            self.mode_txt += f"{i}: {mode}      "
        self.mode_txt += "\n"

        # Define Window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1800, 1200)

        # Set Trackbars
        cv2.createTrackbar("Image ID", self.window_name, 0, len(self.imgs) - 1, self.update_image)
        cv2.createTrackbar("alpha", self.window_name, 0, 100, self.alpha_blend)
        cv2.createTrackbar(
            self.mode_txt, self.window_name, 0, len(self.modes) - 1, self.update_mode
        )

    def run(self):

        #cv2.setTrackbarPos("Image ID", "Window", 2)
        self.update_image()
        while True:
            k = cv2.waitKey()
            if k == 113:
                break
            elif k == 115:

                img_id = cv2.getTrackbarPos("Image ID", "Window")
                mode = self.modes[cv2.getTrackbarPos(self.mode_txt, "Window")]
                file_name = self.imgs[img_id].rsplit("/",1)[-1].replace(".png",f"_{mode}")
                #file_name = f"{cfg.DATASET.NAME}__ID{img_id}"
                os.makedirs("dataset_visualizations", exist_ok=True)

                print(f"Save {file_name}")

                img = cv2.cvtColor(self.left_fig_overlay, cv2.COLOR_RGB2BGR)
                mask = cv2.cvtColor(self.right_fig, cv2.COLOR_RGB2BGR)

                cv2.imwrite(join("dataset_visualizations", file_name + "__image.png"), img)
                cv2.imwrite(join("dataset_visualizations", file_name + "__mask.png"), mask)

        cv2.destroyAllWindows()

    def update_image(self, *arg, **kwargs):
        img_id = cv2.getTrackbarPos("Image ID", "Window")
        img_path = self.imgs[img_id]
        mask_path = self.masks[img_id]
        self.data_extractor = DataExtractor(img_path, mask_path,self.verbose)
        self.update_mode()

    def update_mode(self, *arg, **kwargs):
        mode = cv2.getTrackbarPos(self.mode_txt, "Window")

        self.left_fig = self.data_extractor.show_img()

        if mode == 0:  # Mask
            self.right_fig = self.data_extractor.show_mask()
        elif mode == 1:  # Components
            self.right_fig = self.data_extractor.show_components()
        elif mode == 2:  # Components
            self.right_fig = self.data_extractor.show_components_class()
        elif mode == 3:  # Circles
            self.right_fig = self.data_extractor.show_circles()
        elif mode == 4:  # Circularity
            self.right_fig = self.data_extractor.show_circularity()
        elif mode == 5:  # Mesh
            self.right_fig = self.data_extractor.show_mesh()
        elif mode == 6:  # Distance
            self.right_fig = self.data_extractor.show_distances()
        elif mode == 7:  # Mesh
            self.right_fig = self.data_extractor.show_mesh_regularity()
        elif mode == 8:  # Pore_Size
            self.right_fig = self.data_extractor.show_pore_size()

        self.alpha_blend()

    def alpha_blend(self, *arg, **kwargs):
        alpha = cv2.getTrackbarPos("alpha", "Window") / 100

        # Overlay the right_fig with the left_fig with excluding black areas
        self.left_fig_overlay = cv2.addWeighted(self.left_fig, 1 - alpha, self.right_fig, alpha, 0.0)

        bg_map = np.all(self.right_fig == [0, 0, 0], axis=2) + np.all(
            self.right_fig == [255, 255, 255], axis=2
        )
        self.left_fig_overlay[bg_map] = self.left_fig[bg_map]

        # concat left & right image
        fig = np.concatenate((self.left_fig_overlay, self.right_fig), 1)

        # Convert Fig form RGB to BGR
        fig = cv2.cvtColor(fig, cv2.COLOR_RGB2BGR)
        cv2.imshow("Window", fig)


if __name__ == "__main__":
    root = "/media/l727r/data/nnUNet/nnUNetv2_raw/Dataset253_COMPUTING_it3"
    imgs = glob.glob(join(root, "imagesTr", "*.png"))
    masks = glob.glob(join(root, "labelsTr", "*.png"))

    root = "/home/l727r/Desktop/HEREON_2023_COMPUTING/Data/Membrane_Extraction_2023_03_08"
    imgs = glob.glob(join(root, "images", "*.png"))
    masks = glob.glob(join(root, "predictions_it3_253_ensemble", "*.png"))
    imgs.sort()
    masks.sort()

    # files=["S19016_PS83P4VP17_100k_32_THF_DMF_5_5_S5s_No.1_013_0000.png",
    #        "S19127_PS80P4VP20_182k_23.5wt_THF_DMF_4_6_S5s_Machine_No3_023_0000.png",
    #        "S19124_SVP24_238k_18wt_THF_DMF_4_6_S5s_Machine_No.1_016_0000.png",
    #        ]
    # imgs = [join(root, "input_imgs", file) for file in files]
    # masks = [join(root, "it3_253_ensemble", file).replace("_0000","") for file in files]

    # Start Visualization
    viz = Visualizer(imgs, masks,verbose=False)
    viz.run()
