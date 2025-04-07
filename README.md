# YOLO detector and SOTA Multi-object tracker Toolbox

## ❗❗Important Notes

Compared to the previous version, this is an ***entirely new version (branch v2)***!!!

**Please use this version directly, as I have almost rewritten all the code to ensure better readability and improved results, as well as to correct some errors in the past code.**

```bash 
git clone https://github.com/JackWoo0831/Yolov7-tracker.git
git checkout v2  # change to v2 branch !!
```

🙌 ***If you have any suggestions for adding trackers***, please leave a comment in the Issues section with the paper title or link! Everyone is welcome to contribute to making this repo better.

<div align="center">

**Language**: English | [简体中文](README_CN.md)

</div>

## 🗺️ Latest News

- ***2025.4.7*** Add more Re-ID modules (ShuffleNet, VehicleNet, MobileNet), fix some bugs (such as abandon bbox aspect ratio updating if the tracklet is not activated), and add some functions (customized low filter threshold, fuse detection score, etc.)
- ***2025.4.3*** Support the newest ultralytics version (YOLO v3 ~ v12) and fix some bugs of hybrid sort.

## ❤️ Introduction

This repo is a toolbox that implements the **tracking-by-detection paradigm multi-object tracker**. The detector supports:

- YOLOX 
- YOLO v7
- YOLO v3 ~ v12 by [ultralytics](https://docs.ultralytics.com/), 

and the tracker supports:

- SORT
- DeepSORT 
- ByteTrack ([ECCV2022](https://arxiv.org/pdf/2110.06864)) and ByetTrack-ReID
- Bot-SORT ([arxiv2206](https://arxiv.org/pdf/2206.14651.pdf)) and Bot-SORT-ReID
- OCSORT ([CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.pdf))
- DeepOCSORT ([ICIP2023](https://arxiv.org/abs/2302.11813))
- C_BIoU Track ([arxiv2211](https://arxiv.org/pdf/2211.14317v2.pdf))
- Strong SORT ([IEEE TMM 2023](https://arxiv.org/pdf/2202.13514))
- Sparse Track ([arxiv 2306](https://arxiv.org/pdf/2306.05238))
- UCMC Track ([AAAI 2024](http://arxiv.org/abs/2312.08952))
- Hybrid SORT([AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28471))

and the reid model supports:

Pedestrain Re-ID:
- OSNet
- Extractor from DeepSort
- ShuffleNet
- MobileNet

Vehicle Re-ID:
- VehicleNet ([AICIty-reID-2020](https://github.com/layumi/AICIty-reID-2020))

> **checkpoitns of some Re-ID models**: [Baidu Disk](https://pan.baidu.com/s/1QbVoBz4mPpf4Qsqq1PYXkQ) Code: c655


The highlights are:
- Supporting more trackers than MMTracking
- Rewrite multiple trackers with a ***unified code style***, without the need to configure multiple environments for each tracker 
- Modular design, which ***decouples*** the detector, tracker, reid model and Kalman filter for easy conducting experiments

![gif](figure/demo.gif)


##  🔨 Installation

The basic env is:
- Ubuntu 20.04
- Python：3.9, Pytorch: 1.12

Run following commond to install other packages:

```bash
pip3 install -r requirements.txt
```

### 🔍 Detector installation

1. YOLOX:

The version of YOLOX is **0.1.0 (same as ByteTrack)**. To install it, you can clone the ByteTrack repo somewhere, and run:

``` bash
https://github.com/ifzhang/ByteTrack.git

python3 setup.py develop
```

2. YOLO v7:

There is no need to execute addtional steps as the repo itself is based on YOLOv7.

3. YOLO series by ultralytics:

Please run:

```bash
pip3 install ultralytics
or 
pip3 install --upgrade ultralytics
```

### 📑 Data preparation

***If you do not want to test on the specific dataset, instead, you only want to run demos, please skip this section.***

***No matter what dataset you want to test, please organize it in the following way (YOLO style):***

```
dataset_name
     |---images
           |---train
                 |---sequence_name1
                             |---000001.jpg
                             |---000002.jpg ...
           |---val ...
           |---test ...

     |

```

You can refer to the codes in `./tools` to see how to organize the datasets.

***Then, you need to prepare a `yaml` file to indicate the path so that the code can find the images.***

Some examples are in `tracker/config_files`. The important keys are:

```
DATASET_ROOT: '/data/xxxx/datasets/MOT17'  # your dataset root
SPLIT: test  # train, test or val
CATEGORY_NAMES:  # same in YOLO training
  - 'pedestrian'

CATEGORY_DICT:
  0: 'pedestrian'
```



## 🚗 Practice 

### 🏃 Training 

Trackers generally do not require parameters to be trained. Please refer to the training methods of different detectors to train YOLOs.

Some references may help you:

- YOLOX: `tracker/yolox_utils/train_yolox.py`

- YOLO v7:

```shell
python train_aux.py --dataset visdrone --workers 8 --device <$GPU_id$> --batch-size 16 --data data/visdrone_all.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights <$YOLO v7 pretrained model path$> --name yolov7-w6-custom --hyp data/hyp.scratch.custom.yaml
```  

- YOLO series (YOLO v3 ~ v12) by ultralytics:: `tracker/yolo_ultralytics_utils/train_yolo_ultralytics.py`

```shell
python tracker/yolo_ultralytics_utils/train_yolo_ultralytics.py --model_weight weights/yolo11m.pt --data_cfg tracker/yolo_ultralytics_utils/data_cfgs/visdrone_det.yaml --epochs 30 --batch_size 8 --img_sz 1280 --device 0
```

> The training of Re-ID model please refer to its original paper or github repo. The pedestrain Re-ID model such as ShuffleNet, OSNet please refer to [torchreid](https://github.com/KaiyangZhou/deep-person-reid), the vehicle Re-ID model please refer to [AICIty-reID-2020](https://github.com/layumi/AICIty-reID-2020).

### 😊 Tracking ! 

**If you only want to run a demo**:

```bash
python tracker/track_demo.py --obj ${video path or images folder path} --detector ${yolox, yolov7 or yolo_ultra} --tracker ${tracker name} --kalman_format ${kalman format, sort, byte, ...} --detector_model_path ${detector weight path} --save_images
```

> ❗❗Important Notes
> 
> If you want to use the detector trained by **ultralytics**, the `--detector` argument **must include** the substring `ultra`, such as 
> `--detector yolo_ultra`, `--detector yolo_ultra_v8`, `--detector yolov11_ultra`, `--detector yolo12_ultralytics`, etc.

For example:

```bash
python tracker/track_demo.py --obj M0203.mp4 --detector yolo_ultra_v8 --tracker deepsort --kalman_format byte --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt --save_images
```

**If you want to run trackers on dataset**:

```bash
python tracker/track.py --dataset ${dataset name, related with the yaml file} --detector ${yolox, yolo_ultra_v8 or yolov7} --tracker ${tracker name} --kalman_format ${kalman format, sort, byte, ...} --detector_model_path ${detector weight path}
```

In addition, you can also specify

`--reid`: Enable the reid model (currently useful for ByteTrack, BoT-SORT, OCSORT)

`--reid_model`: Which model to use: Refer to `REID_MODEL_DICT` in `tracker/trackers/reid_models/engine.py` to select

`--reid_model_path`: Loaded re-identification model weight path

`--conf_thresh_low`: For two-stage association models (ByteTrack, BoT-SORT, etc.), the minimum confidence threshold (default 0.1)

`--fuse_detection_score`: If added, the IoU value and the detection confidence value are fused, for example, the source code of BoT-SORT does this

`--save_images`: Save the result image

***Examples of tracking algorithms***:

- SORT: `python tracker/track.py --dataset uavdt --detector yolo_ultra_v8 --tracker sort --kalman_format sort --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt `

- DeepSORT: `python tracker/track.py --dataset visdrone_part --detector yolov7 --tracker deepsort --kalman_format byte --detector_model_path weights/yolov8l_VisDroneDet_35epochs_20230605.pt`

- ByteTrack: `python tracker/track.py --dataset uavdt --detector yolo_ultra_v8 --tracker bytetrack --kalman_format byte --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt`

- OCSort: `python tracker/track.py --dataset mot17 --detector yolox --tracker ocsort --kalman_format ocsort --detector_model_path weights/bytetrack_m_mot17.pth.tar`

- C-BIoU Track: `python tracker/track.py --dataset uavdt --detector yolo_ultra_v8 --tracker c_bioutrack --kalman_format bot --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt`

- BoT-SORT: `python tracker/track.py --dataset uavdt --detector yolox --tracker botsort --kalman_format bot --detector_model_path weights/yolox_m_uavdt_50epochs.pth.tar`

- Strong SORT: `python tracker/track.py --dataset visdrone_part --detector yolo_ultra_v8 --tracker strongsort --kalman_format strongsort --detector_model_path weights/yolov8l_VisDrone_35epochs_20230509.pt`

- Sparse Track: `python tracker/track.py --dataset uavdt --detector yolo_ultra_v11 --tracker sparsetrack --kalman_format bot --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt`

- UCMC Track: `python tracker/track.py --dataset mot17 --detector yolox --tracker ucmctrack --kalman_format ucmc --detector_model_path weights/bytetrack_m_mot17.pth.tar --camera_parameter_folder ./tracker/cam_param_files`

- Hybrid SORT: `python tracker/track.py --dataset visdrone_part --detector yolo_ultra --tracker hybridsort --kalman_format hybridsort --detector_model_path weights/yolov8l_VisDrone_35epochs_20230509.pt --save_images`

> **Important notes for UCMC Track:**
> 
> 1. Camera parameters. The UCMC Track need the intrinsic and extrinsic parameter of camera. Please organize like the format of `tracker/cam_param_files/uavdt/M0101.txt`. One video sequence corresponds to one txt file. If you do not have the labelled parameters, you can refer to the estimating toolbox in original repo ([https://github.com/corfyi/UCMCTrack](https://github.com/corfyi/UCMCTrack)).
> 
> 2. The code does not contain the camera motion compensation part between every two frame, please refer to [https://github.com/corfyi/UCMCTrack/issues/12](https://github.com/corfyi/UCMCTrack/issues/12). From my perspective, since the algorithm name is 'uniform', the update of compensation between every two frames is not necessary.


### ✅ Evaluation 

Coming Soon. As an alternative, after obtaining the result txt file, you can use the [Easier to use TrackEval repo](https://github.com/JackWoo0831/Easier_To_Use_TrackEval).