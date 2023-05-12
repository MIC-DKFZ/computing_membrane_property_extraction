import glob
import os
import shutil
from os.path import join
import cv2
import numpy as np
from skimage.measure import label, regionprops, regionprops_table, perimeter
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib import cm
import matplotlib as mpl
import itertools
import math
from tqdm import tqdm
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
warnings.filterwarnings(action='ignore', message='Degrees of freedom <= 0 for slice')

px_to_nm=100/45

class_dict = {
    0: "background",
    1: "covered_area",
    2: "open_pore",
    3: "closed_pore",
    4: "internal_structure",
    5: "artifact_blob",
    6: "artifact_dust",
}
cmap_napari = [
    [0, 0, 0],  # 0 - background
    [120, 37, 7],  # 1 - covered_area
    [92, 214, 249],  # 2 - open_pore
    [146, 137, 233],  # 3 - closed_pore
    [156, 97, 48],  # 4 - internal_structure
    [72, 58, 160],  # 5 - artifact_blob
    [171, 236, 139],  # 6 - artifact_dust
    [180, 125, 189],  # 7 - ignore
]

class ColorMapper():
    def __init__(self):
        self.cmap = mpl.colormaps["magma"]

    def __call__(self,val, min_val=0,max_val=1,*args, **kwargs):
        norm_value=(val-min_val)/(max_val-min_val)
        norm_value=max(0.01, min(norm_value,0.99))
        col = np.array(self.cmap(norm_value)[0:3]) * 255
        col=col.astype(dtype=np.uint8)
        return col

    def rand_col(self):
        return np.random.randint(0, 255, [3])



def point_distance(p0, p1):
    return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def in_percentages(a, b):
    return a / b * 100


def print_dict(dictionary):
    for k, v in dictionary.items():
        print(f"    {(k+':').ljust(50)}{v:.3f}")


class DataExtractor:
    def __init__(self, img_path, mask_path, verbose=False):
        self.name = img_path.rsplit("/", 1)[-1]
        self.cmap=ColorMapper()

        # Load Image and Mask
        self.img = cv2.imread(img_path)
        self.mask = cv2.imread(mask_path, -1)
        # Get Connected Components + Region Props from segmentation
        class_ids = [2, 3]  # 2: open pore; 3: closed pore
        # binary_mask = np.any([self.mask == open_pore, self.mask == closed_pore], axis=0)
        binary_mask = np.any([self.mask == class_id for class_id in class_ids], axis=0)
        self.regions = regionprops(label(binary_mask))

        # Create Pores, exlcude boreder cases
        self.all_pores=np.array([Pore(region, self.mask) for region in self.regions])
        self.pores = np.array([pore for pore in self.all_pores if not pore.border_case()])

        # Construct the Mesh from all center points of the all regions and then exclude border
        # cases afterwards
        if len(self.all_pores) >= 4:
            self.mesh = Delaunay(np.array([pore.properties.centroid for pore in self.all_pores]))
            self.triangles = []
            for edge, nn in zip(self.mesh.simplices, self.mesh.neighbors):

                pores = self.all_pores[edge]
                if not any([pore.border_case() for pore in pores]):

                    self.triangles.append(Triangle([pore.properties.centroid for pore in pores]))
                    #connections += itertools.combinations(simpices, 2)

        else:
            self.triangles = []
            self.edge_tuples = []
            self.mesh = None

        # Get all edges inside the Mesh
        connections = []
        if self.mesh:
            for simpices in self.mesh.simplices:
                if not any([pore.border_case() for pore in self.all_pores[simpices]]):
                    connections += itertools.combinations(simpices, 2)
            self.edge_tuples = list({*map(tuple, map(sorted, connections))})
        # Extract all Parameters
        if not verbose:
            self.print_stats()

    def print_stats(self):

        print(f"\nFile: {self.name}")

        print(f"--- mesh ---")
        print_dict(self.get__mesh__properties())

        print(f"--- all_pores ---")
        print_dict(self.get__all_pores__properties())
        print(f"--- open_pores ---")
        print_dict(self.get__open_pores__properties())
        print(f"--- closed_pores ---")
        print_dict(self.get__closed_pores__properties())

        print(f"--- covered_area ---")
        print_dict(self.get__covered_area__properties())
        print(f"--- internal_structure ---")
        print_dict(self.get__internal_structure__properties())
        print(f"--- artifact_blob ---")
        print_dict(self.get__artifact_blob__properties())
        print(f"--- artifact_dust ---")
        print_dict(self.get__artifact_dust__properties())

    def get__class__properties(self, class_ids,in_nm=True):

        binary_mask = np.any([self.mask == class_id for class_id in class_ids], axis=0)

        regions = regionprops(label(binary_mask))

        pixel__count = np.sum(binary_mask)
        if in_nm:
            pixel__count=pixel__count*px_to_nm*px_to_nm
        pixel__pct = pixel__count / (binary_mask.shape[0] * binary_mask.shape[1]) * 100

        instances__count = len(regions)

        instances__size = [region.area for region in regions]
        instances__mean_size = np.mean(instances__size)
        instances__std_size = np.std(instances__size)

        return {
            "area__covered": pixel__count, # in nm
            #"pixel__pct": pixel__pct,
            "instances__count": instances__count,
            "instances__mean_size": instances__mean_size, # in nm
            "instances__std_size": instances__std_size, # in nm
        }

    def get__pore__properties(self, class_ids, in_nm=True):

        pixel__count = self.get_total_pore_area(class_ids=class_ids,in_nm=in_nm)
        #pixel__pct = pixel__count / (self.mask.shape[0] * self.mask.shape[1]) * 100
        instances__count = self.get_pore_number(class_ids=class_ids)
        instances__size = self.get_pore_sizes(class_ids=class_ids,in_nm=in_nm)
        instances__diameter = self.get_pore_diameter(class_ids=class_ids,in_nm=in_nm)
        instances__circularity = self.get_pore_circularities(class_ids=class_ids)

        instances__mean_size = np.mean(instances__size)
        instances__std_size = np.std(instances__size)

        instances__mean_diameter = np.mean(instances__diameter)
        instances__std_diameter = np.std(instances__diameter)

        instances__mean_circularity = np.mean(instances__circularity)
        instances__std_circularity = np.std(instances__circularity)

        return {
            "area__covered": pixel__count,
            #"pixel__pct": pixel__pct,
            "instances__count": instances__count,
            "instances__mean_size": instances__mean_size,
            "instances__std_size": instances__std_size,
            "instances__mean_diameter": instances__mean_diameter,
            "instances__std_diameter": instances__std_diameter,
            "instances__mean_circularity": instances__mean_circularity,
            "instances__std_circularity": instances__std_circularity,
        }

    def get__all_pores__properties(self):
        class_id = [2, 3]
        class_name = "all_pores"

        properties = self.get__pore__properties(class_id)
        properties = {f"{class_name}__{k}": v for k, v in properties.items()}
        return properties

    def get__open_pores__properties(self):
        class_id = [2]
        class_name = "open_pores"

        properties = self.get__pore__properties(class_id)
        #properties = self.get__class__properties(class_id)
        properties = {f"{class_name}__{k}": v for k, v in properties.items()}
        return properties

    def get__closed_pores__properties(self):
        class_id = [3]
        class_name = "closed_pores"

        properties = self.get__pore__properties(class_id)
        #properties = self.get__class__properties(class_id)
        properties = {f"{class_name}__{k}": v for k, v in properties.items()}
        return properties

    def get__covered_area__properties(self):
        class_id = 1
        class_name = "covered_area"

        properties = self.get__class__properties([class_id])
        properties = {f"{class_name}__{k}": v for k, v in properties.items()}
        return properties

    def get__internal_structure__properties(self):
        class_id = 4
        class_name = "internal_structure"

        properties = self.get__class__properties([class_id])
        properties = {f"{class_name}__{k}": v for k, v in properties.items()}
        return properties

    def get__artifact_blob__properties(self):
        class_id = 5
        class_name = "artifact_blob"

        properties = self.get__class__properties([class_id])
        properties = {f"{class_name}__{k}": v for k, v in properties.items()}
        return properties

    def get__artifact_dust__properties(self):
        class_id = 6
        class_name = "artifact_dust"

        properties = self.get__class__properties([class_id])
        properties = {f"{class_name}__{k}": v for k, v in properties.items()}
        return properties

    def get__mesh__properties(self,in_nm=True):
        vertices__count = self.get_mesh_number_vertices()
        edges__count = self.get_mesh_number_edges()

        edges__length = self.get_mesh_edge_size(in_nm=in_nm)
        edges__mean_length = np.mean(edges__length)
        edges__std_length = np.std(edges__length)

        triangle__size = self.get_mesh_size(in_nm=in_nm)
        triangle__mean_size = np.mean(triangle__size)
        triangle__std_size = np.std(triangle__size)

        triangle__regularity = self.get_mesh_regularity()
        triangle__mean_regularity = np.mean(triangle__regularity)
        triangle__std_regularity = np.std(triangle__regularity)

        return {
            #"mesh__vertices__count": vertices__count,
            "mesh__edges__count": edges__count,
            "mesh__edges__mean_length": edges__mean_length,
            "mesh__edges__std_length": edges__std_length,
            "mesh__triangles__mean_size": triangle__mean_size,
            "mesh__triangles__std_size": triangle__std_size,
            "mesh__triangles__mean_regularity": triangle__mean_regularity,
            "mesh__triangles__std_regularity": triangle__std_regularity,
        }

    def get_pore_number(self, class_ids=[2, 3]):
        pores = [pore for pore in self.pores if pore.class_id in class_ids]
        return len(pores)

    def get_pore_sizes(self, class_ids=[2, 3],in_nm=True):
        areas = []
        pores = [pore for pore in self.pores if pore.class_id in class_ids]
        for pore in pores:
            areas.append(pore.properties.area)
        if in_nm:
            areas=[area*px_to_nm**px_to_nm for area in areas]
        return areas

    def get_total_pore_area(self, class_ids=[2, 3],in_nm=True):
        area = 0
        pores = [pore for pore in self.pores if pore.class_id in class_ids]
        for pore in pores:
            area += pore.properties.area
        if in_nm:
            area*=px_to_nm**px_to_nm
        return area

    def get_pore_diameter(self, class_ids=[2, 3],in_nm=True):
        diameters = []
        pores = [pore for pore in self.pores if pore.class_id in class_ids]
        for pore in pores:
            diameters.append(pore.properties.equivalent_diameter)
        if in_nm:
            diameters=[diameter*px_to_nm for diameter in diameters]
        return diameters

    def get_pore_circularities(self, class_ids=[2, 3]):
        circularity = []
        pores = [pore for pore in self.pores if pore.class_id in class_ids]
        for pore in pores:
            circularity.append(pore.get_circularity())
        return circularity

    def get_mesh_number_vertices(self):
        if self.mesh:
            return len(self.mesh.points)
        else:
            return 0

    def get_mesh_number_edges(self):
        if self.mesh:
            return len(self.edge_tuples)
        else:
            return 0

    def get_mesh_size(self,in_nm=True):
        areas = []
        for triangle in self.triangles:
            areas.append(triangle.get_area())
        if in_nm:
            areas=[area*px_to_nm*px_to_nm for area in areas]
        return areas

    def get_mesh_regularity(self):
        regularity = []
        for triangle in self.triangles:
            regularity.append(triangle.get_regularity())
        return regularity

    def get_mesh_edge_size(self,in_nm=True):
        distances = []
        for edge in self.edge_tuples:
            coords = [pore.properties.centroid for pore in self.all_pores[np.array(edge)]]
            distances.append(point_distance(coords[0], coords[1]))
        if in_nm:
            distances=[distance*px_to_nm for distance in distances]
        return distances

    def show_img(self):
        return self.img

    def show_mask(self):
        w, h = self.mask.shape
        fig = np.zeros((w, h, 3), dtype=np.uint8)
        for class_id in np.unique(self.mask):
            x, y = np.where(self.mask == class_id)
            if class_id >= len(cmap_napari):
                fig[x, y] = [0, 0, 0]
            else:
                fig[x, y, :] = cmap_napari[class_id]
        return fig

    def show_components(self):
        w, h, c = self.img.shape
        fig = np.ones((w, h, c), dtype=np.uint8) * 255
        for pore in self.all_pores:
            col = self.cmap.rand_col()

            bbox = pore.properties["bbox"]
            fig[bbox[0] : bbox[2], bbox[1] : bbox[3]][pore.properties["image"]] = col
        return fig

    def show_pore_size(self):
        w, h, c = self.img.shape
        fig = np.ones((w, h, c), dtype=np.uint8) * 255
        for pore in self.all_pores:
            col=self.cmap(pore.properties.area*px_to_nm*px_to_nm,min_val=15,max_val=2300)
            bbox = pore.properties["bbox"]
            fig[bbox[0] : bbox[2], bbox[1] : bbox[3]][pore.properties["image"]] = col
        return fig

    def show_components_class(self):

        w, h, c = self.img.shape
        fig = np.ones((w, h, c), dtype=np.uint8) * 255
        for pore in self.all_pores:
            class_id = pore.class_id
            if pore.border_case():
                col=[122,122,122]
            else:
                col = cmap_napari[class_id]
            bbox = pore.properties["bbox"]
            fig[bbox[0] : bbox[2], bbox[1] : bbox[3]][pore.properties["image"]] = col
        return fig

    def show_circles(self):
        w, h, c = self.img.shape
        fig = np.ones((w, h, c), dtype=np.uint8) * 255
        for pore in self.pores:
            col = self.cmap.rand_col()
            bbox = pore.properties["bbox"]
            x, y = pore.properties.centroid  # ["centroid"]
            d = pore.properties.equivalent_diameter  # ["equivalent_diameter"]
            #x=round(x)
            #y=round(y)

            x_min=int(min(bbox[0],np.trunc(x-d/2)))
            x_max=int(max(bbox[2],np.ceil(x+d/2)))

            y_min=int(min(bbox[1],np.trunc(y-d/2)))
            y_max=int(max(bbox[3],np.ceil(y+d/2)))
            #fig = cv2.rectangle(fig, (y_min, x_min), (y_max, x_max), (255, 0, 0), 1)

            pore_grid=np.zeros((x_max-x_min,y_max-y_min))
            pore_grid[bbox[0]-x_min:bbox[2]-x_min,bbox[1]-y_min:bbox[3]-y_min]=pore.properties["image"]

            coords = np.indices(pore_grid.shape)
            dist = np.sqrt(np.sum((coords - np.array([x-x_min,y-y_min]).reshape(-1, 1, 1)) ** 2, axis=0))
            #t=dist<=d/2
            tp=np.logical_and(dist<=d/2,pore_grid)
            fp=np.logical_and(dist>d/2,pore_grid)
            fn=np.logical_and(dist<=d/2,np.logical_not(pore_grid))

            iou=tp.sum()/(tp.sum()+fp.sum()+fn.sum())
            #t=t.astype(int)*255

            dist=dist/np.max(dist)*255
            #dist
            #print(fig.shape)
            #fig = cv2.circle(fig, (round(y), round(x)), round(d / 2.0), [1, 1, 1], 1)
            #print(np.min(dist),np.max(dist))
            #pore_grid*=255
            #iou*=255
            #print(np.unique(pore_grid))
            col = self.cmap(iou)
            #fig[x_min: x_max, y_min: y_max][pore_grid.astype(bool)] = col#np.dstack((iou,iou*0,255-iou))#fig[x_min: x_max, y_min: y_max]*0.5+0.5* col
            fig[x_min: x_max, y_min: y_max][tp] = [0,255,0] #np.dstack((t,t,t))#fig[x_min: x_max, y_min: y_max]*0.5+0.5* col
            fig[x_min: x_max, y_min: y_max][fp] = [0,0,255]
            fig[x_min: x_max, y_min: y_max][fn] = [255,0,0]
            # fig[bbox[0]: bbox[2], bbox[1]: bbox[3]][pore.properties["image"]] = \
            # fig[bbox[0]: bbox[2], bbox[1]: bbox[3]][pore.properties["image"]] * 0.5 + 0.5 * col

        return fig

    def show_circularity(self):
        w, h, c = self.img.shape
        fig = np.ones((w, h, c), dtype=np.uint8) * 255
        for pore in self.pores:
            col = self.cmap(pore.get_circularity())
            bbox = pore.properties["bbox"]
            fig[bbox[0] : bbox[2], bbox[1] : bbox[3]][pore.properties["image"]] = col
        return fig

    def show_mesh(self):
        fig = np.ones(self.img.shape, dtype=np.uint8) * 255
        for triangle in self.triangles:
            p0, p1, p2 = triangle.get_points()

            col_edge = [29, 56, 142]
            col_vertex = [203, 141, 17]

            fig = cv2.line(fig, p0, p1, col_edge, 1)
            fig = cv2.line(fig, p0, p2, col_edge, 1)
            fig = cv2.line(fig, p1, p2, col_edge, 1)

            fig = cv2.circle(fig, p0, 3, col_vertex, -1)
            fig = cv2.circle(fig, p1, 3, col_vertex, -1)
            fig = cv2.circle(fig, p2, 3, col_vertex, -1)
        return fig

    def show_distances(self):
        distances = self.get_mesh_edge_size()
        w, h, c = self.img.shape
        fig = np.ones((w, h, c), dtype=np.uint8) * 255

        for edge, dist in zip(self.edge_tuples, distances):
            col = self.cmap(dist,min_val=30,max_val=222)
            p1 = self.all_pores[edge[0]].properties.centroid
            p2 = self.all_pores[edge[1]].properties.centroid
            fig = cv2.line(
                fig,
                (int(p1[1]), int(p1[0])),
                (int(p2[1]), int(p2[0])),
                [int(col[0]), int(col[1]), int(col[2])],
                2,
            )
        return fig

    def show_mesh_regularity(self):
        fig = np.ones(self.img.shape, dtype=np.uint8) * 255
        for triangle in self.triangles:
            p0, p1, p2 = triangle.get_points()
            c = triangle.get_regularity()

            col=self.cmap(c)

            fig = cv2.drawContours(
                fig, [np.array([p0, p1, p2])], 0, [int(col[0]), int(col[1]), int(col[2])], -1
            )
            col_edge = [122, 122, 122]
            fig = cv2.line(fig, p0, p1, col_edge, 1)
            fig = cv2.line(fig, p0, p2, col_edge, 1)
            fig = cv2.line(fig, p1, p2, col_edge, 1)

        return fig


class Pore:
    def __init__(self, region_prop, mask):
        self.mask_shape = mask.shape
        self.properties = region_prop
        self.class_id = self.get_class(mask, self.properties)

    def border_case(self):
        #return False
        # Check if the bounding box of the pore touches the image border
        x_min, y_min, x_max, y_max = self.properties.bbox
        #print(x_min<=0 or y_min<=0 or x_max>=self.mask_shape[0] or y_max>=self.mask_shape[1])
        return x_min<=0 or y_min<=0 or x_max>=self.mask_shape[0] or y_max>=self.mask_shape[1]

    def get_class(self, mask, region):
        mask_region = mask[
            region["bbox"][0] : region["bbox"][2], region["bbox"][1] : region["bbox"][3]
        ]
        un, cou = np.unique(mask_region[region["image"]], return_counts=True)
        class_id = np.bincount(mask_region[region["image"]]).argmax()
        # print(class_id, un, cou)
        return class_id

    def get_circularity(self):

        # Get Properties
        bbox = self.properties["bbox"]
        x, y = self.properties.centroid
        d = self.properties.equivalent_diameter # diameter of the circle with the same sum of pixels
        # Rounding? Leads to different results
        # x=round(x)
        # y=round(y)

        # Get the coordinates of the bounding box which covers the complete circle AND the image
        x_min = int(min(bbox[0], np.trunc(x - d / 2)))
        x_max = int(max(bbox[2], np.ceil(x + d / 2)))

        y_min = int(min(bbox[1], np.trunc(y - d / 2)))
        y_max = int(max(bbox[3], np.ceil(y + d / 2)))

        # With the Coordinates create the new bounding box and fill in the image
        pore_grid = np.zeros((x_max - x_min, y_max - y_min))
        pore_grid[bbox[0] - x_min:bbox[2] - x_min, bbox[1] - y_min:bbox[3] - y_min] = self.properties["image"]

        # Get the distance from each point in the grid to the circles center (x,y)
        coords = np.indices(pore_grid.shape)
        dist = np.sqrt(
            np.sum((coords - np.array([x - x_min, y - y_min]).reshape(-1, 1, 1)) ** 2, axis=0))

        # Binarize the distance, by checking if the distance is higher or lower than the radius
        dist=dist <= d / 2

        # Compute the IoU between the binary cricle and the pore image
        intersection = np.logical_and(dist, pore_grid).sum()
        union = np.logical_or(dist, pore_grid).sum()
        iou = intersection / union

        return iou

        if self.properties.axis_major_length==0:
            circ=0
        else:
            circ=self.properties.axis_minor_length/self.properties.axis_major_length
        return circ

        circ = (4 * math.pi * self.properties.area) / (
                self.properties.perimeter * self.properties.perimeter
        )

        print("C",circ,self.properties.perimeter,self.properties.eccentricity)
        return circ

        return self.properties.eccentricity



class Triangle:
    def __init__(self, vertices):
        self.vertices = vertices

    def get_points(self, as_int=True):
        p0 = (self.vertices[0][1], self.vertices[0][0])
        p1 = (self.vertices[1][1], self.vertices[1][0])
        p2 = (self.vertices[2][1], self.vertices[2][0])

        if as_int:
            p0 = (round(p0[0]), round(p0[1]))
            p1 = (round(p1[0]), round(p1[1]))
            p2 = (round(p2[0]), round(p2[1]))

        return p0, p1, p2

    def get_edge_lenght(self):
        p0, p1, p2 = self.get_points(as_int=False)
        l0 = point_distance(p0, p1)
        l1 = point_distance(p0, p2)
        l2 = point_distance(p1, p2)
        return l0, l1, l2

    def get_area(self):
        l0, l1, l2 = self.get_edge_lenght()
        s = (l0 + l1 + l2) / 2.0
        area = (s * (s - l0) * (s - l1) * (s - l2)) ** 0.5
        return area

    def get_regularity(self):
        l0, l1, l2 = self.get_edge_lenght()
        reg = np.min([l0, l1, l2]) / np.max([l0, l1, l2])
        return reg


if __name__ == "__main__":
    img_root = "/home/l727r/Documents/cluster-data/COMPUTING_imgs_all/input_imgs"
    mask_root = "/home/l727r/Documents/cluster-data/COMPUTING_imgs_all/it3_253_ensemble"
    out_root = "/home/l727r/Desktop/HEREON_2022_COMPUTING/Data"

    imgs = glob.glob(join(img_root, "*.png"))
    masks = glob.glob(join(mask_root, "*.png"))
    imgs.sort()
    masks.sort()

    import itertools

    # Class__property__measure
    # Class Properties
    class_names = ["covered_area", "internal_structure", "artifact_blob", "artifact_dust"]
    class_parameters = [
        "pixel__count",
        "pixel__pct",
        "instances__count",
        "instances__mean_size",
        "instances__std_size",
    ]
    class_properties = [
        f"{cn}__{cp}" for cn, cp in list(itertools.product(class_names, class_parameters))
    ]

    # Mesh Properties
    mesh_properties = [
        "vertices__count",
        "edges__count",
        "edges__mean_length",
        "edges__std_length",
        "triangle__mean_size",
        "triangle__std_size",
        "triangle__mean_regularity",
        "triangle__std_regularity",
    ]
    mesh_properties = [f"mesh__{mp}" for mp in mesh_properties]

    # Pore Properties
    pore_types = [
        "all_pores",
        "open_pores",
        "closed_pores",
    ]
    pore_parameters = class_parameters + [
        "instances__mean_diameter"
        "instances__std_diameter"
        "instances__mean_circularity"
        "instances__std_circularity"
    ]
    pore_properties = [
        f"{pt}__{pp}" for pt, pp in list(itertools.product(pore_types, pore_parameters))
    ]
    df_columns = pore_properties + mesh_properties + class_properties
    df_columns = class_properties
    df = pd.DataFrame(columns=df_columns)

    for img, mask in tqdm(zip(imgs, masks)):
        ex = DataExtractor(img, mask, True)
        print(ex.get__covered_area__properties())
        print(ex.get__internal_structure__properties())
        print(ex.get__artifact_blob__properties())
        print(ex.get__artifact_dust__properties())
        print(ex.get__mesh__properties())
        # Mesh Information
        # reg = ex.get_mesh_regularity()
        # dist = ex.get_mesh_edge_size()
        # num = ex.get_mesh_number_vertices()

        break

        # if dist>100 or reg<0.5 or num<=10:
        #    print(dist, reg,num)
        #    shutil.copy(img, img.replace(img_root, join(out_root, "reg_dist_num_outliers")))
        # if reg>0.5:
        #     shutil.copy(img,img.replace(img_root,join(out_root,"regularity_inliers")))
        # else:
        #     shutil.copy(img, img.replace(img_root, join(out_root, "regularity_outliers")))

        # quit()

    quit()
"""


        # internal_structure=ex.get__internal_structure__properties()
        # artifact_blob=ex.get__artifact_blob__properties()
        # artifact_dust=ex.get__artifact_dust__properties()

        return
        covered_area=ex.get__covered_area__properties()
        internal_structure=ex.get__internal_structure__properties()
        artifact_blob=ex.get__artifact_blob__properties()
        artifact_dust=ex.get__artifact_dust__properties()
        print(f"-- Class Information")
        for i in [1, 2, 3, 4, 5, 6]:
            print(f"   -- {class_dict[i]}:")
            number_of_pixel, pixel_percent, number_of_regions, avg_size = self.get_class_properties(
                [i]
            )
            print(f"      -- Number Pixel: {number_of_pixel} ({pixel_percent:.1f}%)")
            print(f"      -- Number Instances: {number_of_regions}")
            print(f"      -- Average Size: {avg_size:.4f}")

        # Extract all information about the pores, first when joining open and closed pores, than look at each separately
        print(f"-- Pore Information")
        number_all_pores = self.get_pore_number(class_ids=[2, 3])
        all_pore_size = self.get_pore_sizes(class_ids=[2, 3])
        all_pore_total_area = self.get_total_pore_area(class_ids=[2, 3])
        all_pores_circularity = self.get_pore_circularities(class_ids=[2, 3])
        all_pores_diameter = self.get_pore_diameter(class_ids=[2, 3])
        print(f"   -- All Pores:")
        print(f"      -- Number:              {number_all_pores}")
        print(
            "      -- Pixel:              "
            f" {all_pore_total_area} ({all_pore_total_area/(self.mask.shape[0]*self.mask.shape[1])*100:.1f}%)"
        )
        print(
            "      -- Pore Size:          "
            f" mean={np.mean(all_pore_size):.4f} std={np.std(all_pore_size):.4f}"
        )
        print(
            "      -- Pore Diameter:      "
            f" mean={np.mean(all_pores_diameter):.4f} std={np.std(all_pores_diameter):.4f}"
        )
        print(
            "      -- Region Circularity: "
            f" mean={np.mean(all_pores_circularity):.4f} std={np.std(all_pores_circularity):.4f}"
        )

        number_open_pores = self.get_pore_number(class_ids=[2])
        open_pore_size = self.get_pore_sizes(class_ids=[2])
        open_pore_total_area = self.get_total_pore_area(class_ids=[2])
        open_pores_circularity = self.get_pore_circularities(class_ids=[2])
        open_pores_diameter = self.get_pore_diameter(class_ids=[2])
        print(f"   -- Open Pores:")
        print(
            "      -- Number:             "
            f" {number_open_pores} ({number_open_pores/number_all_pores*100 if number_all_pores>0 else 0:.1f}%)"
        )
        print(
            "      -- Pixel:              "
            f" {open_pore_total_area} ({open_pore_total_area/(self.mask.shape[0]*self.mask.shape[1])*100:.1f}%)"
        )
        print(
            "      -- Pore Size:          "
            f" mean={np.mean(open_pore_size):.4f} std={np.std(open_pore_size):.4f}"
        )
        print(
            "      -- Pore Diameter:      "
            f" mean={np.mean(open_pores_diameter):.4f} std={np.std(open_pores_diameter):.4f}"
        )
        print(
            "      -- Region Circularity: "
            f" mean={np.mean(open_pores_circularity):.4f} std={np.std(open_pores_circularity):.4f}"
        )

        number_closed_pores = self.get_pore_number(class_ids=[3])
        closed_pore_size = self.get_pore_sizes(class_ids=[3])
        closed_pore_total_area = self.get_total_pore_area(class_ids=[3])
        closed_pores_circularity = self.get_pore_circularities(class_ids=[3])
        closed_pores_diameter = self.get_pore_diameter(class_ids=[3])
        print(f"   -- Closed Pores:")
        print(
            "      -- Number:             "
            f" {number_closed_pores} ({number_closed_pores/number_all_pores*100 if number_all_pores>0 else 0:.1f}%)"
        )
        print(
            "      -- Pixel:              "
            f" {closed_pore_total_area} ({closed_pore_total_area/(self.mask.shape[0]*self.mask.shape[1])*100:.1f}%)"
        )
        print(
            "      -- Pore Size:          "
            f" mean={np.mean(closed_pore_size):.4f} std={np.std(closed_pore_size):.4f}"
        )
        print(
            "      -- Pore Diameter:      "
            f" mean={np.mean(closed_pores_diameter):.4f} std={np.std(closed_pores_diameter):.4f}"
        )
        print(
            "      -- Region Circularity: "
            f" mean={np.mean(closed_pores_circularity):.4f} std={np.std(closed_pores_circularity):.4f}"
        )

        # Extract all information about the mesh structure
        print(f"-- Mesh Information")
        number_vertices = self.get_mesh_number_vertices()
        number_edges = self.get_mesh_number_edges()
        mesh_size = self.get_mesh_size()
        mesh_reg = self.get_mesh_regularity()
        distances = self.get_mesh_edge_size()
        print(f"   -- Number of Vertices:     {number_vertices}")
        print(f"   -- Number of Edges:        {number_edges}")
        print(
            f"   -- Edge Length:            mean={np.mean(distances):.4f},"
            f" std={np.std(distances):.4f}"
        )
        print(
            f"   -- Triangle Size:          mean={np.mean(mesh_size):.4f},"
            f" std={np.std(mesh_size):.4f}"
        )
        print(
            f"   -- Triangle Regularity:    mean={np.mean(mesh_reg):.4f},"
            f" std={np.std(mesh_reg):.4f}"
        )

    # def mesh_remove_outliers(self):
"""
