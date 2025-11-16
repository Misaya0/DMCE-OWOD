import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from torchvision.ops import nms

import PIL.Image
import supervision as sv
import json
import os
import cv2

import sys

sys.path.append('/home/wl/code/opensource/open_detect/YOLO-World/yolo_world')

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)


def run_image(
        runner,
        input_image,
        max_num_boxes=100,
        score_thr=0.05,  ##src=0.05
        nms_thr=0.5,
        output_image="output.png",
):
    texts = [[t.strip()] for t in class_names.split(",")] + [[" "]]
    data_info = runner.pipeline(dict(img_id=0, img_path=input_image,
                                     texts=texts))

    data_batch = dict(
        inputs=data_info["inputs"].unsqueeze(0),
        data_samples=[data_info["data_samples"]],
    )

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        runner.model.class_names = texts
        pred_instances = output.pred_instances

    keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep_idxs]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]
    output.pred_instances = pred_instances

    pred_instances = pred_instances.cpu().numpy()
    infos = []
    for cls, box, conf in zip(pred_instances['labels'], pred_instances['bboxes'], pred_instances['scores']):
        info = '%d %.2f %.2f %.2f %.2f %.4f' % (cls, box[0], box[1], box[2], box[3], conf)
        infos.append(info)

    detections = sv.Detections(
        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores']
    )
    label_names = class_names.split(",")
    # print(label_names)

    labels = [
        f"{class_id}{label_names[class_id]}{confidence:0.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    image = PIL.Image.open(input_image)
    svimage = np.array(image)
    svimage = bounding_box_annotator.annotate(svimage, detections)
    svimage = label_annotator.annotate(svimage, detections, labels)
    return svimage[:, :, ::-1], infos


if __name__ == "__main__":

    cfg = Config.fromfile(
        "configs/owod_ft/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod.py"
    )
    cfg.load_from = "work_dirs/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod_mowodb_train_task1/best_owod_Both_epoch_20.pth"

    owod_root = "data/OWOD"
    owod_dataset = "MOWODB"
    owod_task = 1

    file_path = f"{owod_root}/ImageSets/{owod_dataset}/t{owod_task}_known.txt"

    # 读取文件并转换为字符串
    with open(file_path, 'r') as f:
        # 读取所有行并去除首尾空格/换行符
        class_list = [line.strip() for line in f.readlines() if line.strip()]

        # 合并为逗号分隔字符串
        class_names = ",".join(class_list)
    class_names = class_names + ",unknown"
    img_dir = 'demo/images'
    result_dir = 'uniow_s_ft_result'

    cfg.work_dir = "."
    runner = Runner.from_cfg(cfg)
    runner.call_hook("before_run")
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)
    runner.model.eval()

    for name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, name)
        resutl_path = os.path.join(result_dir, name)

        img, info = run_image(runner, img_path)
        cv2.imwrite(resutl_path, img)
    print("转换结果:", class_names)