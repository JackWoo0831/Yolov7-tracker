# YOLO检测器与SOTA多目标跟踪工具箱

## ❗❗重要提示

与之前的版本相比，这是一个***全新的版本（分支v2）***！！！

**请直接使用这个版本，因为我几乎重写了所有代码，以确保更好的可读性和改进的结果，并修正了以往代码中的一些错误。**

```bash 
git clone https://github.com/JackWoo0831/Yolov7-tracker.git
git checkout v2  # change to v2 branch !!
```

🙌 ***如果您有任何关于添加跟踪器的建议***，请在Issues部分留言并附上论文标题或链接！欢迎大家一起来让这个repo变得更好

## 🗺️ 最近更新

- ***2024.11.29*** 修复了C-BIoU Tracker中轨迹状态的更新和预测的错误
- ***2024.10.24*** 增加了 Hybrid SORT 并且修复了OC-SORT的一些bug和错误。


## ❤️ 介绍

这个仓库是一个实现了***检测后跟踪范式***多目标跟踪器的工具箱。检测器支持：

- YOLOX 
- YOLO v7
- YOLO v8, 

跟踪器支持:

- SORT
- DeepSORT 
- ByteTrack ([ECCV2022](https://arxiv.org/pdf/2110.06864))
- Bot-SORT ([arxiv2206](https://arxiv.org/pdf/2206.14651.pdf))
- OCSORT ([CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.pdf))
- C_BIoU Track ([arxiv2211](https://arxiv.org/pdf/2211.14317v2.pdf))
- Strong SORT ([IEEE TMM 2023](https://arxiv.org/pdf/2202.13514))
- Sparse Track ([arxiv 2306](https://arxiv.org/pdf/2306.05238))
- UCMC Track ([AAAI 2024](http://arxiv.org/abs/2312.08952))
- Hybrid SORT([AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28471))

REID模型支持：

- OSNet
- DeepSORT中的

亮点包括:
- 支持的跟踪器比MMTracking多
- 用***统一的代码风格***重写了多个跟踪器，无需为每个跟踪器配置多个环境 
- 模块化设计，将检测器、跟踪器、外观提取模块和卡尔曼滤波器**解耦**，便于进行实验

![gif](figure/demo.gif)

## 🗺️ 路线图

- [ x ] Add UCMC Track
- [] Add more ReID modules.

##  🔨 安装

基本环境是：
- Ubuntu 18.04
- Python：3.9, Pytorch: 1.12

运行以下命令安装其他包：

```bash
pip3 install -r requirements.txt
```

### 🔍 检测器安装

1. YOLOX:

YOLOX的版本是0.1.0（与ByteTrack相同）。要安装它，你可以在某处克隆ByteTrack仓库，然后运行：

``` bash
https://github.com/ifzhang/ByteTrack.git

python3 setup.py develop
```

2. YOLO v7:

由于仓库本身就是基于YOLOv7的，因此无需执行额外的步骤。

3. YOLO v8:

请运行：

```bash
pip3 install ultralytics==8.0.94
```

### 📑 数据准备

***如果你不想在特定数据集上测试，而只想运行演示，请跳过这一部分。***

***无论你想测试哪个数据集，请按以下方式（YOLO风格）组织：***

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

你可以参考`./tools`中的代码来了解如何组织数据集。

***然后，你需要准备一个yaml文件来指明路径，以便代码能够找到图像***

一些示例在tracker/config_files中。重要的键包括：

```
DATASET_ROOT: '/data/xxxx/datasets/MOT17'  # your dataset root
SPLIT: test  # train, test or val
CATEGORY_NAMES:  # same in YOLO training
  - 'pedestrian'

CATEGORY_DICT:
  0: 'pedestrian'
```



## 🚗 实践

### 🏃 训练

跟踪器通常不需要训练参数。请参考不同检测器的训练方法来训练YOLOs。

以下参考资料可能对你有帮助：

- YOLOX: `tracker/yolox_utils/train_yolox.py`

- YOLO v7:

```shell
python train_aux.py --dataset visdrone --workers 8 --device <$GPU_id$> --batch-size 16 --data data/visdrone_all.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights <$YOLO v7 pretrained model path$> --name yolov7-w6-custom --hyp data/hyp.scratch.custom.yaml
```  

- YOLO v8: `tracker/yolov8_utils/train_yolov8.py`



### 😊 跟踪！

如果你只是想运行一个demo:

```bash
python tracker/track_demo.py --obj ${video path or images folder path} --detector ${yolox, yolov8 or yolov7} --tracker ${tracker name} --kalman_format ${kalman format, sort, byte, ...} --detector_model_path ${detector weight path} --save_images
```

例如:

```bash
python tracker/track_demo.py --obj M0203.mp4 --detector yolov8 --tracker deepsort --kalman_format byte --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt --save_images
```

如果你想在数据集上测试:

```bash
python tracker/track.py --dataset ${dataset name, related with the yaml file} --detector ${yolox, yolov8 or yolov7} --tracker ${tracker name} --kalman_format ${kalman format, sort, byte, ...} --detector_model_path ${detector weight path}
```

例如:

- SORT: `python tracker/track.py --dataset uavdt --detector yolov8 --tracker sort --kalman_format sort --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt `

- DeepSORT: `python tracker/track.py --dataset uavdt --detector yolov7 --tracker deepsort --kalman_format byte --detector_model_path weights/yolov7_UAVDT_35epochs_20230507.pt`

- ByteTrack: `python tracker/track.py --dataset uavdt --detector yolov8 --tracker bytetrack --kalman_format byte --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt`

- OCSort: `python tracker/track.py --dataset uavdt --detector yolov8 --tracker ocsort --kalman_format ocsort --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt`

- C-BIoU Track: `python tracker/track.py --dataset uavdt --detector yolov8 --tracker c_bioutrack --kalman_format bot --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt`

- BoT-SORT: `python tracker/track.py --dataset uavdt --detector yolox --tracker botsort --kalman_format bot --detector_model_path weights/yolox_m_uavdt_50epochs.pth.tar`

- UCMC Track: `python tracker/track.py --dataset mot17 --detector yolox --tracker ucmctrack --kalman_format ucmc --detector_model_path weights/bytetrack_m_mot17.pth.tar --camera_parameter_folder ./tracker/cam_param_files`

- Hybrid SORT: `python tracker/track.py --dataset mot17 --detector yolox --tracker hybridsort --kalman_format hybridsort --detector_model_path weights/bytetrack_m_mot17.pth.tar --save_images`

>**UCMC Track的重要提示：**
> 
> 1. 相机参数. UCMC Track需要相机的内参和外参. 请按照`tracker/cam_ram_files/uavdt/M0101.txt`的格式组织. 一个视频序列对应一个txt文件. 如果您没有标记的参数, 可以参考原始仓库中的估算工具箱([https://github.com/corfyi/UCMCTrack](https://github.com/corfyi/UCMCTrack)).
> 
> 2. 该代码不包含每两帧之间的相机运动补偿部分, 请参阅[https://github.com/corfyi/UCMCTrack/issues/12](https://github.com/corfyi/UCMCTrack/issues/12). 在我看来, 既然算法叫"统一相机运动补偿", 因此不需要每两帧之间再更新补偿. 

### ✅ 评估

马上推出！作为备选项，你可以使用这个repo： [Easier to use TrackEval repo](https://github.com/JackWoo0831/Easier_To_Use_TrackEval).