from pycocotools import mask
from skimage import measure
import json
import shutil
import itertools
import numpy as np
from simplification.cutil import simplify_coords_vwp
import os, cv2, copy
from distinctipy import distinctipy
from collections import defaultdict

def init_coco(dataset_folder, image_names, categories, coco_json_path):
    coco_json = {
        "info": {
            "description": "SAM Dataset",
            "url": "",
            "version": "1.0",
            "year": 2023,
            "contributor": "Sam",
            "date_created": "2021/07/01",
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }
    for i, category in enumerate(categories):
        coco_json["categories"].append(
            {"id": i, "name": category, "supercategory": category}
        )
    for i, image_name in enumerate(image_names):
        im = cv2.imread(os.path.join(dataset_folder, image_name))
        coco_json["images"].append(
            {
                "id": i,
                "file_name": image_name,
                "width": im.shape[1],
                "height": im.shape[0],
            }
        )
    with open(coco_json_path, "w") as f:
        json.dump(coco_json, f)


def bunch_coords(coords):
    coords_trans = []
    for i in range(0, len(coords) // 2):
        coords_trans.append([coords[2 * i], coords[2 * i + 1]])
    return coords_trans


def unbunch_coords(coords):
    return list(itertools.chain(*coords))


def bounding_box_from_mask(mask):
    xyxyxyxy=rotated_bounding_box_from_mask(mask)
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = []
    for contour in contours:
        all_contours.extend(contour)
    convex_hull = cv2.convexHull(np.array(all_contours))
    x, y, w, h = cv2.boundingRect(convex_hull)
    return x, y, w, h,xyxyxyxy
def rotated_bounding_box_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([])
    
    all_points = np.vstack(contours)
    min_rect = cv2.minAreaRect(all_points)
    
    box_points = cv2.boxPoints(min_rect)
    #print(box_points)
    return np.int0(box_points)



def parse_mask_to_coco(image_id, anno_id, image_mask, category_id, poly=False):
    start_anno_id = anno_id
    x, y, width, height, xyxy = bounding_box_from_mask(image_mask)
    xy1,xy2,xy3,xy4=xyxy
    if poly == False:
        fortran_binary_mask = np.asfortranarray(image_mask)
        encoded_mask = mask.encode(fortran_binary_mask)
    if poly == True:
        contours = measure.find_contours(image_mask, 0.5)
    annotation = {
        "image_id": image_id,
        "id": start_anno_id,
        "category_id": category_id,
        "bbox": [int(x), int(y), int(width), int(height)],
        "rotated":[int(xy1[0]),int(xy1[1]),int(xy2[0]),int(xy2[1]),int(xy3[0]),int(xy3[1]),int(xy4[0]),int(xy4[1])],
    }

    return annotation


class DatasetExplorer:
    def __init__(self, dataset_folder, categories=None, coco_json_path=None):
        self.dataset_folder = dataset_folder
        self.image_names = os.listdir(os.path.join(self.dataset_folder, "images"))
        self.image_names = [
            os.path.split(name)[1] for name in self.image_names if name.endswith(".jpg") or name.endswith(".png")
        ]
        self.image_dir=dataset_folder
        self.coco_json_path = coco_json_path
        if not os.path.exists(coco_json_path):
            self.__init_coco_json(categories)
        with open(coco_json_path, "r") as f:
            self.coco_json = json.load(f)

        self.categories = [category["name"] for category in self.coco_json["categories"]]
        self.annotations_by_image_id = {}
        for annotation in self.coco_json["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in self.annotations_by_image_id:
                self.annotations_by_image_id[image_id] = []
            self.annotations_by_image_id[image_id].append(annotation)

        self.global_annotation_id = len(self.coco_json["annotations"])
        self.category_colors = distinctipy.get_colors(len(self.categories))
        self.category_colors = [
            tuple([int(255 * c) for c in color]) for color in self.category_colors
        ]

    def __init_coco_json(self, categories):
        appended_image_names = [
            os.path.join("images", name) for name in self.image_names
        ]
        init_coco(
            self.dataset_folder, appended_image_names, categories, self.coco_json_path
        )

    def get_colors(self, category_id):
        return self.category_colors[category_id]
    
    def get_categories(self):
        """返回当前所有有效类别列表"""
        return [category["name"] for category in self.coco_json["categories"]]

    def get_num_images(self):
        return len(self.image_names)

    def get_image_data(self, image_id):
        image_name = self.coco_json["images"][image_id]["file_name"]
        image_path = os.path.join(self.dataset_folder, image_name)
        embedding_path = os.path.join(
            self.dataset_folder,
            "embeddings",
            os.path.splitext(os.path.split(image_name)[1])[0] + ".npy",
        )
        image = cv2.imread(image_path)
        image_bgr = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_embedding = np.load(embedding_path)
        return image, image_bgr, image_embedding

    def __add_to_our_annotation_dict(self, annotation):
        image_id = annotation["image_id"]
        if image_id not in self.annotations_by_image_id:
            self.annotations_by_image_id[image_id] = []
        self.annotations_by_image_id[image_id].append(annotation)
    
    def __delet_to_our_annotation_dict(self, image_id):
        # 检查列表是否为空，避免 pop 空列表
        if image_id in self.annotations_by_image_id and len(self.annotations_by_image_id[image_id]) > 0:
            self.annotations_by_image_id[image_id].pop(-1)

    # 修改 delet_annotation 方法
    def delet_annotation(self, image_id):
        # 先检查是否有可删除的标注
        if image_id not in self.annotations_by_image_id or len(self.annotations_by_image_id[image_id]) == 0:
            return  # 没有标注可删除，直接返回
        self.__delet_to_our_annotation_dict(image_id)
        # 检查 annotations 列表是否为空
        if len(self.coco_json["annotations"]) > 0:
            self.coco_json["annotations"].pop(-1)
            self.global_annotation_id -= 1

    def get_annotations(self, image_id, return_colors=False):
        if image_id not in self.annotations_by_image_id:
            return [], []
        cats = [a["category_id"] for a in self.annotations_by_image_id[image_id]]

        try :
            colors = [self.category_colors[c] for c in cats]
        except Exception as e:
            colors =(200,200,200)
        if return_colors:
            return self.annotations_by_image_id[image_id], colors
        return self.annotations_by_image_id[image_id]

    def add_annotation(self, image_id, category_id, mask, poly=True):
        if mask is None:
            return
        annotation = parse_mask_to_coco(
            image_id, self.global_annotation_id, mask, category_id, poly=poly
        )
        self.__add_to_our_annotation_dict(annotation)
        self.coco_json["annotations"].append(annotation)
        self.global_annotation_id += 1


    def save_annotation(self):
        with open(self.coco_json_path, "w") as f:
            json.dump(self.coco_json, f)
    def add_category(self, category_name):
        # 检查类别是否已存在
        for cat in self.coco_json["categories"]:
            if cat["name"] == category_name:
                return
        
        # 添加新类别
        new_id = len(self.coco_json["categories"])  # 注意：原代码可能用了+1，需与category_id生成逻辑一致
        self.coco_json["categories"].append({
            "id": new_id,
            "name": category_name,
            "supercategory": "none"
        })
        
        # 同步更新类别列表和颜色列表
        self.categories = [category["name"] for category in self.coco_json["categories"]]
        self.category_colors = distinctipy.get_colors(len(self.categories))  # 重新生成颜色
        self.category_colors = [
            tuple([int(255 * c) for c in color]) for color in self.category_colors
        ]
        
        self.save_annotation()  # 保存更改


    def get_image_name(self, image_id):
        for img in self.coco_json["images"]:
            if img["id"] == image_id:
                return img["file_name"]
        return ""

    def get_image_path_by_id(self, image_id):
        """根据image_id返回图片完整路径"""
        for img in self.coco_json["images"]:
            if img["id"] == image_id:
                return os.path.join(self.image_dir, img["file_name"])  # self.image_dir为图片文件夹路径（确保已有此属性）
        return None


        #if cat_id is None:
            #return
        

    def remove_category(self, category_name):
        # 查找类别ID
        cat_id = None
        for i, cat in enumerate(self.coco_json["categories"]):
            if cat["name"] == category_name:
                cat_id = i
                break
        if cat_id is None:
            return
        
        # 删除该类别下的所有标注
        self.coco_json["annotations"] = [
            ann for ann in self.coco_json["annotations"] 
            if ann["category_id"] != cat_id
        ]
        
        # 从类别列表中删除
        self.coco_json["categories"].pop(cat_id)
        
        # 更新剩余类别的ID（保持连续）
        for i, cat in enumerate(self.coco_json["categories"]):
            cat["id"] = i
        
        # 更新内部缓存
        self.categories = [cat["name"] for cat in self.coco_json["categories"]]
        # 重新生成颜色列表（保持与类别数量一致）
        self.category_colors = distinctipy.get_colors(len(self.categories))
        self.category_colors = [
            tuple([int(255 * c) for c in color]) for color in self.category_colors
        ]
        
        # 更新按图像ID分组的标注
        self.annotations_by_image_id = defaultdict(list)
        for ann in self.coco_json["annotations"]:
            self.annotations_by_image_id[ann["image_id"]].append(ann)
        
        # 保存更改
        self.save_annotation()
