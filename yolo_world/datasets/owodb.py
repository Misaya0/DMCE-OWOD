# partly taken from  https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py
import functools
import json

import os
import copy
from mmdet.utils import ConfigType
from mmdet.datasets import BaseDetDataset
from tqdm import tqdm

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from mmyolo.registry import DATASETS

import xml.etree.ElementTree as ET
from mmengine.logging import MMLogger

from .owodb_const import *
import torch
from torchvision.ops import nms
from yolo_world.datasets.boxes import Boxes,pairwise_iou,BoxMode


@DATASETS.register_module()
class OWODDataset(BatchShapePolicyDataset, BaseDetDataset):
    """`OWOD in Pascal VOC format <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    METAINFO = {
        'classes': (),
        'palette': None,
    }

    def __init__(self,
                 data_root: str,
                 dataset: str = 'MOWODB',
                 image_set: str='train',
                 owod_cfg: ConfigType = None,
                 training_strategy: int = 0,
                 **kwargs):

        self.images = []
        self.annotations = []
        self.imgids = []
        self.imgid2annotations = {}
        self.image_set_fns = []

        self.image_set = image_set
        self.dataset=dataset
        self.CLASS_NAMES = VOC_COCO_CLASS_NAMES[dataset]
        self.task_num = owod_cfg.task_num
        self.owod_cfg = owod_cfg
        
        self._logger = MMLogger.get_current_instance()

        # training strategy
        self.training_strategy = training_strategy
        if "test" not in image_set:
            if training_strategy == 0:
                self._logger.info(f"Training strategy: OWOD")
            elif training_strategy == 1:
                self._logger.info(f"Training strategy: ORACLE")
            else:
                raise ValueError(f"Invalid training strategy: {training_strategy}")

        OWODDataset.METAINFO['classes'] = self.CLASS_NAMES
        
        self.data_root=str(data_root)
        annotation_dir = os.path.join(self.data_root, 'Annotations', dataset)
        image_dir = os.path.join(self.data_root, 'JPEGImages', dataset)
        # annotation_dir = os.path.join(self.data_root, 'Annotations', 'MOWODB')
        # image_dir = os.path.join(self.data_root, 'JPEGImages', 'MOWODB')

        file_names = self.extract_fns()
        self.image_set_fns.extend(file_names)
        self.images.extend([os.path.join(image_dir, x + ".jpg") for x in file_names])
        self.annotations.extend([os.path.join(annotation_dir, x + ".xml") for x in file_names])
        self.imgids.extend(x for x in file_names)            
        self.imgid2annotations.update(dict(zip(self.imgids, self.annotations)))

        assert (len(self.images) == len(self.annotations) == len(self.imgids))

        super().__init__(**kwargs)

    def extract_fns(self):
        splits_dir = os.path.join(self.data_root, 'ImageSets')
        splits_dir = os.path.join(splits_dir, self.dataset)
        image_sets = []
        file_names = []

        if 'test' in self.image_set: # for test
            image_sets.append(self.image_set)
        else: # owod or oracle
            image_sets.append(f"t{self.task_num}_{self.image_set}")

        self.image_set_list = image_sets
        for image_set in image_sets:
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
            with open(os.path.join(split_f), "r") as f:
                file_names.extend([x.strip() for x in f.readlines()])
        return file_names

    ### OWOD
    def remove_prev_class_and_unk_instances(self, target):
        # For training data. Removing earlier seen class objects and the unknown objects..
        prev_intro_cls = self.owod_cfg.PREV_INTRODUCED_CLS
        curr_intro_cls = self.owod_cfg.CUR_INTRODUCED_CLS
        valid_classes = range(prev_intro_cls, prev_intro_cls + curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["bbox_label"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def remove_unknown_instances(self, target):
        # For finetune data. Removing the unknown objects...
        prev_intro_cls = self.owod_cfg.PREV_INTRODUCED_CLS
        curr_intro_cls = self.owod_cfg.CUR_INTRODUCED_CLS
        valid_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["bbox_label"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def label_known_class_and_unknown(self, target):
        # For test and validation data.
        # Label known instances the corresponding label and unknown instances as unknown.
        prev_intro_cls = self.owod_cfg.PREV_INTRODUCED_CLS
        curr_intro_cls = self.owod_cfg.CUR_INTRODUCED_CLS
        total_num_class = self.owod_cfg.num_classes
        known_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
        # for annotation in entry:
            if annotation["bbox_label"] not in known_classes:
                annotation["bbox_label"] = total_num_class - 1
        return entry

    def load_data_list(self):
        data_list = []
        self._logger.info(f"Loading {self.dataset} from {self.image_set_list}...")
        # for i, img_id in tqdm(enumerate(self.imgids)):
        for i, img_id in enumerate(self.imgids):
            raw_data_info = dict(
                img_path=self.images[i],
                img_id=img_id,
            )
            parsed_data_info = self.parse_data_info(raw_data_info)
            data_list.append(parsed_data_info)

        self._logger.info(f"{self.dataset} Loaded, {len(data_list)} images in total")
        return data_list
    
    def parse_data_info(self, raw_data_info):
        data_info = copy.copy(raw_data_info)
        img_id = data_info["img_id"]
        tree = ET.parse(self.imgid2annotations[img_id])
        dir_name = os.path.join("data/OWOD")
        instances = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text

            if cls in VOC_CLASS_NAMES_COCOFIED:
                cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
            try:
                bbox_label = self.CLASS_NAMES.index(cls)
            except ValueError:
                continue # ignore 'ego' class in nu-OWODB
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instance = dict(
                bbox_label=bbox_label,
                bbox=bbox,
                area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                ignore_flag=0,
            )
            instances.append(instance)

        if 'train' in self.image_set:
            if self.training_strategy == 1: # oracle
                instances = self.label_known_class_and_unknown(instances)
                # instances = self.load_one_sam_data(instances,os.path.join(dir_name, 'SAM_H'), img_id,self.image_set_list[0], tr_sz=25, tr_iou=0.7)
            else: # owod
                instances = self.remove_prev_class_and_unk_instances(instances)
                # instances = self.load_one_sam_data(instances, os.path.join(dir_name, 'SAM_H'), img_id,self.image_set_list[0], tr_sz=25,
                #                                    tr_iou=0.7)
        elif 'test' in self.image_set:
            instances = self.label_known_class_and_unknown(instances)
        elif 'ft' in self.image_set:
            instances = self.remove_unknown_instances(instances)
            
        data_info.update(
            height=int(tree.findall("./size/height")[0].text),
            width=int(tree.findall("./size/width")[0].text),
            instances=instances,
        )

        return data_info

    def load_one_sam_data(self,instances, sam_file_root, image_id, image_set, tr_sz=5, tr_iou=0.9):
        unknown_label = 20
        if 't1' in image_set:
            unknown_label = 20
        elif 't2' in image_set:
            unknown_label = 40
        elif 't3' in image_set:
            unknown_label = 60
        elif 't4' in image_set:
            unknown_label = 80
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gt_boxes = Boxes(torch.tensor([item['bbox'] for item in instances], device=device))
        with open(os.path.join(sam_file_root, f'{image_id}.json'), 'r') as f:
            sam_data = json.load(f)['box_result']
        sam_data = [item for item in sam_data if item['bbox'][2] >= tr_sz and item['bbox'][3] >= tr_sz]  # 过滤掉小目标
        sam_boxes_list = [[
            item['bbox'][0], item['bbox'][1],
            item['bbox'][0] + item['bbox'][2],
            item['bbox'][1] + item['bbox'][3]]
            for item in sam_data]  # xyxy format
        sam_score_list = [item['score'] for item in sam_data]
        sam_boxes = torch.tensor(sam_boxes_list, device=device, dtype=torch.float)
        sam_scores = torch.tensor(sam_score_list, device=device, dtype=torch.float)

        # image = cv2.imread("/2T/gzj/OrthogonalDet-main/datasets/JPEGImages/" + image_id + ".jpg")
        # plt.figure(figsize=(20, 20))
        # plt.imshow(image)
        # show_anns_hou_numpy(sam_boxes)
        # plt.axis('off')
        # plt.show()

        # 筛选低得分框
        valid_indices = sam_scores >= 0.968
        sam_boxes = sam_boxes[valid_indices]
        sam_scores = sam_scores[valid_indices]
        if sam_boxes.numel() != 0:
            # 对 SAM 生成的框进行 NMS 去冗余
            keep_indices = nms(sam_boxes, sam_scores, 0.5)  # 只保留非冗余框
            sam_boxes = sam_boxes[keep_indices]
            sam_scores = sam_scores[keep_indices]
            # if image_id == '2008_002709':
            #     print("aaa")
            # if sam_boxes.shape[0] == 0:#如果SAM生成的框全部都被过滤掉
            #     sam_boxes = torch.tensor(sam_boxes_list[0], device=device, dtype=torch.float).view(1,4)
            #     sam_scores = torch.tensor(sam_score_list[0], device=device, dtype=torch.float).view(1)
            #     instances.append({
            #         "category_id": 80,
            #         "bbox": [sam_boxes_list[0][0], sam_boxes_list[0][1],
            #                 sam_boxes_list[0][2], sam_boxes_list[0][3]],  # xyxy
            #         "bbox_mode": BoxMode.XYXY_ABS,
            #         'score': sam_score_list[0]
            #     })
            #     return num, instances
            # if torch.sum(valid_indices == True) > 0:
            if sam_scores.shape[0] > 70:  # 进行再一次筛选
                valid_indices = sam_scores >= 0.98
                sam_boxes = sam_boxes[valid_indices]
                sam_scores = sam_scores[valid_indices]

        sam_boxes_obj = Boxes(sam_boxes)

        # plt.figure(figsize=(20, 20))
        # plt.imshow(image)
        # show_anns_hou_numpy(sam_boxes)
        # plt.axis('off')
        # plt.show()
        if sam_boxes.numel() != 0 and len(instances) != 0:
            ious, _ = pairwise_iou(sam_boxes_obj, gt_boxes).max(dim=1)
            for boxe, iou, score in zip(sam_boxes.tolist(), ious, sam_scores.tolist()):
                if (iou >= tr_iou):
                    continue
                instances.append({
                    "area": (boxe[2] - boxe[0]) * (boxe[3] - boxe[1]),
                    "bbox": [boxe[0], boxe[1],
                             boxe[2], boxe[3]],  # xyxy
                    "bbox_label": unknown_label,
                    "ignore_flag": 0,
                })

        # num_instances = 0
        # for instance in instances:
        #     if instance['category_id'] == 80:
        #         num_instances += 1
        # if num_instances == 0:
        #     instances.append({
        #         "category_id": 80,
        #         "bbox": [sam_boxes_list[0][0], sam_boxes_list[0][1],
        #                 sam_boxes_list[0][2], sam_boxes_list[0][3]],  # xyxy
        #         "bbox_mode": BoxMode.XYXY_ABS,
        #         'score': sam_score_list[0]
        #     })
        # num_unkown_instances = 0
        # for instance in instances:
        #     if instance['category_id'] == 80:
        #         num_unkown_instances += 1
        # print(image_id + ".json文件经过过滤后还有" + str(num_unkown_instances) + "个未知物体")
        # num.append(num_unkown_instances)
        # if num_unkown_instances > 80 or image_id == '000000076654':
        #     print(f"num_unkown_instances: {num_unkown_instances}"+"!!!!!!!!!!")
        #     image = cv2.imread("/2T/gzj/OrthogonalDet-main/datasets/JPEGImages/" + image_id + ".jpg")
        #
        #     plt.figure(figsize=(20, 20))
        #     plt.imshow(image)
        #     show_anns_hou_numpy(sam_boxes)
        #     plt.axis('off')
        #     plt.show()
        #     print(f"num_unkown_instances: {num_unkown_instances}"+"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return instances
    def filter_data(self):
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False
        min_size = self.filter_cfg.get('min_size', 0) \
            if self.filter_cfg is not None else 0

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos