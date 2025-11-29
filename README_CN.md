# YOLO检测器与SOTA多目标跟踪工具箱

## ❗❗重要提示

最近有一个重大更新, 为了准确性和可读性, 我**重新组织了**所有代码. 更重要的是, **添加了两个新的sota跟踪器**(ImpraAssoc和TrackTrack), 并支持**TensorRT engine**.

新版本发布于**branch v2.1**:

```bash 
git clone https://github.com/JackWoo0831/Yolov7-tracker.git
git checkout v2.1  # change to v2.1 branch !!
```

🙌 ***QQ交流群已建立, 欢迎加入!***，您可以在QQ群中提出bug、意见建议或者一起来做有趣的CV/AI项目！
然而，**Bug或问题还是优先在issue区提出，以便让更多人看到.**

<img src="figure/GroupQRcode.jpg" alt="group" style="width:40%;">

## 🗺️ 最近更新

- ***2025.11.28*** 增加FastTracker跟踪算法。修复CBIoU Tracker中的丢失轨迹的bug.
- ***2025.7.8*** 新版本2.1发布. 添加ImproAssoc, TrackTrack并支持TensorRT. 其他细节如下:

<details>
<summary>更新细节</summary>


1. 重新注释整理matching.py中所有函数
2. 对于相机运动补偿, 可自定义特征提取子的算法(SIFT, ORB, ECC), 运行`track.py`时指定`--cmc_method`参数.
3. 对于BoT-SORT, ByteTrack等方法, 原先的低置信度筛选阈值被固定设置为`0.1`. 现在可以手动设置, 运行`track.py`(或`track_demo.py`)时指定`--conf_thresh_low`参数.
4. 加入`init_thresh`参数作为初始化目标阈值, 弃用原本的`args.conf + 0.1`定值. 运行`track.py`时指定`--init_thresh`参数.
5. 在ReID特征提取中, 原本的裁剪-resize大小为定值`(h, w) = (128, 64)`, 现在可以手动设置, 运行`track.py`时指定`--reid_crop_size`参数, 例如`--reid_crop_size 32 64`.
6. 将所有Tracker继承BaseTracker类以实现良好的代码复用
7. 修复strongsort的reid相似度计算bug
8. 弃用cython_bbox以获得更好的numpy版本兼容
9. 弃用np.float等以获得更好的numpy版本兼容
10. 重新整理requirements.txt
</details>


## ❤️ 介绍

这个仓库是一个实现了***检测后跟踪范式***多目标跟踪器的工具箱。检测器支持：

- YOLOX 
- YOLO v7
- YOLO v3 ~ v12 by [ultralytics](https://docs.ultralytics.com/), 

跟踪器支持:

- SORT
- DeepSORT 
- ByteTrack ([ECCV2022](https://arxiv.org/pdf/2110.06864)) 以及 ByetTrack-ReID
- Bot-SORT ([arxiv2206](https://arxiv.org/pdf/2206.14651.pdf)) 以及 Bot-SORT-ReID
- OCSORT ([CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.pdf))
- DeepOCSORT ([ICIP2023](https://arxiv.org/abs/2302.11813))
- C_BIoU Track ([arxiv2211](https://arxiv.org/pdf/2211.14317v2.pdf))
- Strong SORT ([IEEE TMM 2023](https://arxiv.org/pdf/2202.13514))
- Sparse Track ([arxiv 2306](https://arxiv.org/pdf/2306.05238))
- UCMC Track ([AAAI 2024](http://arxiv.org/abs/2312.08952))
- Hybrid SORT([AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28471))
- ImproAssoc ([CVPRW 2023](https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/papers/Stadler_An_Improved_Association_Pipeline_for_Multi-Person_Tracking_CVPRW_2023_paper.pdf))
- TrackTrack ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html))
- FastTracker ([arxiv 2508](https://arxiv.org/pdf/2508.14370))

REID模型支持：

行人重识别模型:
- OSNet
- Extractor from DeepSort
- ShuffleNet
- MobileNet

车辆重识别模型:

- VehicleNet ([AICIty-reID-2020](https://github.com/layumi/AICIty-reID-2020))

> **部分重识别模型的权重**: [百度网盘](https://pan.baidu.com/s/1QbVoBz4mPpf4Qsqq1PYXkQ) 提取码: c655

亮点包括:
- 支持的跟踪器比MMTracking多
- 用***统一的代码风格***重写了多个跟踪器，无需为每个跟踪器配置多个环境 
- 模块化设计，将检测器、跟踪器、外观提取模块和卡尔曼滤波器**解耦**，便于进行实验

![gif](figure/demo.gif)


##  🔨 安装

基本环境是：
- Ubuntu 20.04
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

3. Ultralytics的YOLO系列模型：

请运行：

```bash
pip3 install ultralytics
or 
pip3 install --upgrade ultralytics
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

- Ultralytics的YOLO系列模型 (YOLO v3 ~ v12): `tracker/yolo_ultralytics_utils/train_yolo_ultralytics.py`

```shell
python tracker/yolo_ultralytics_utils/train_yolo_ultralytics.py --model_weight weights/yolo11m.pt --data_cfg tracker/yolo_ultralytics_utils/data_cfgs/visdrone_det.yaml --epochs 30 --batch_size 8 --img_sz 1280 --device 0
```

> 关于重识别模型的训练, 请先参照对应模型的原论文或代码. 行人重识别模型例如 ShuffleNet, OSNet 参考 [torchreid](https://github.com/KaiyangZhou/deep-person-reid), 车辆重识别模型参考 [AICIty-reID-2020](https://github.com/layumi/AICIty-reID-2020).

### 😊 跟踪！

**如果你只是想运行一个demo**:

```bash
python tracker/track_demo.py --obj ${video path or images folder path} --detector ${yolox, yolov7 or yolo_ultra} --tracker ${tracker name} --kalman_format ${kalman format, sort, byte, ...} --detector_model_path ${detector weight path} --save_images
```

> ❗❗重要提示
> 
> 如果你是通过 **ultralytics** 库训练检测模型, 命令里的`--detector`参数 **必须包含**`ultra`字段, 例如
> `--detector yolo_ultra`, `--detector yolo_ultra_v8`, `--detector yolov11_ultra`, `--detector yolo12_ultralytics`, 等等.

例如:

```bash
python tracker/track_demo.py --obj M0203.mp4 --detector yolov8 --tracker deepsort --kalman_format byte --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt --save_images
```

**如果你想在数据集上测试**:

```bash
python tracker/track.py --dataset ${dataset name, related with the yaml file} --detector ${yolox, yolo_ultra_v8 or yolov7} --tracker ${tracker name} --kalman_format ${kalman format, sort, byte, ...} --detector_model_path ${detector weight path}
```

此外, 还可以指定

`--reid`: 启用reid模型(目前对ByteTrack, BoT-SORT, OCSORT有用)

`--reid_model`: 采用那种模型: 参照`tracker/trackers/reid_models/engine.py`中的`REID_MODEL_DICT`选取

`--reid_model_path`: 加载的重识别模型权重路径

`--conf_thresh_low`: 对于两阶段关联模型(ByteTrack, BoT-SORT等), 最低置信度阈值(默认0.1)

`--fuse_detection_score`: 如果加上, 就融合IoU的值和检测置信度的值, 例如BoT-SORT的源码是这样做的

`--save_images`: 保存结果图片

***各种跟踪算法运行示例***:

- SORT: `python tracker/track.py --dataset uavdt --detector yolo_ultra_v8 --tracker sort --kalman_format sort --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt `

- DeepSORT: `python tracker/track.py --dataset visdrone_part --detector yolov7 --tracker deepsort --kalman_format byte --detector_model_path weights/yolov8l_VisDroneDet_35epochs_20230605.pt`

- ByteTrack: `python tracker/track.py --dataset uavdt --detector yolo_ultra_v8 --tracker bytetrack --kalman_format byte --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt`

- ByteTrack-ReID: `python tracker/track.py --dataset uavdt --detector yolo_ultra_v8 --tracker bytetrack --kalman_format byte --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt --reid --reid_model osnet_x0_25 --reid_model_path weights/osnet_x0_25.pth`

- OCSort: `python tracker/track.py --dataset mot17 --detector yolox --tracker ocsort --kalman_format ocsort --detector_model_path weights/bytetrack_m_mot17.pth.tar`

- DeepOCSORT: `python tracker/track.py --dataset mot17 --detector yolox --tracker ocsort --kalman_format ocsort --detector_model_path weights/bytetrack_m_mot17.pth.tar --reid --reid_model shufflenet_v2_x1_0 --reid_model_path shufflenetv2_x1-5666bf0f80.pth`

- C-BIoU Track: `python tracker/track.py --dataset uavdt --detector yolo_ultra_v8 --tracker c_bioutrack --kalman_format bot --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt`

- BoT-SORT: `python tracker/track.py --dataset uavdt --detector yolox --tracker botsort --kalman_format bot --detector_model_path weights/yolox_m_uavdt_50epochs.pth.tar`

- BoT-SORT-ReID: `python tracker/track.py --dataset uavdt --detector yolox --tracker botsort --kalman_format bot --detector_model_path weights/yolox_m_uavdt_50epochs.pth.tar --reid --reid_model vehiclenet --reid_model_path vehicle_net_resnet50.pth`

- Strong SORT: `python tracker/track.py --dataset visdrone_part --detector yolo_ultra_v8 --tracker strongsort --kalman_format strongsort --detector_model_path weights/yolov8l_VisDroneDet_35epochs_20230605.pt`

- Sparse Track: `python tracker/track.py --dataset uavdt --detector yolo_ultra_v11 --tracker sparsetrack --kalman_format bot --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt`

- UCMC Track: `python tracker/track.py --dataset mot17 --detector yolox --tracker ucmctrack --kalman_format ucmc --detector_model_path weights/bytetrack_m_mot17.pth.tar --camera_parameter_folder ./tracker/cam_param_files`

- Hybrid SORT: `python tracker/track.py --dataset visdrone_part --detector yolo_ultra --tracker hybridsort --kalman_format hybridsort --detector_model_path weights/yolov8l_VisDrone_35epochs_20230509.pt --save_images`

- ImproAssoc: `python tracker/track.py --dataset visdrone_part --detector yolo_ultra --tracker improassoc --kalman_format bot --detector_model_path weights/yolov8l_VisDrone_35epochs_20230509.pt --save_images`

- TrackTrack: `python tracker/track.py --dataset visdrone_part --detector yolo_ultra --tracker tracktrack --kalman_format bot --detector_model_path weights/yolov8l_VisDrone_35epochs_20230509.pt --save_images --nms_thresh 0.95 --reid`

- FastTracker: `python tracker/track.py --dataset uavdt --detector yolo_ultra_v8 --tracker fasttrack --kalman_format byte --detector_model_path weights/yolov8l_UAVDT_60epochs_20230509.pt`

>**UCMC Track的重要提示：**
> 
> 1. 相机参数. UCMC Track需要相机的内参和外参. 请按照`tracker/cam_ram_files/uavdt/M0101.txt`的格式组织. 一个视频序列对应一个txt文件. 如果您没有标记的参数, 可以参考原始仓库中的估算工具箱([https://github.com/corfyi/UCMCTrack](https://github.com/corfyi/UCMCTrack)).
> 
> 2. 该代码不包含每两帧之间的相机运动补偿部分, 请参阅[https://github.com/corfyi/UCMCTrack/issues/12](https://github.com/corfyi/UCMCTrack/issues/12). 在我看来, 既然算法叫"统一相机运动补偿", 因此不需要每两帧之间再更新补偿. 

>**Fast Tracker的重要提示：**
> 
> 在fast_tracker.py中，与跟踪器有关的配置在FAST_TRACKER_CONFIG全局变量中，包括对遮挡目标记录的相关阈值（速度阻尼、边界框放大等），以及融合道路结构的环境优化（"ROIs"键，具体数值以及含义请参照原论文）

### ✨ TensorRT的转换与推理

该代码支持**全自动**的Tensor RT engine的生成与推理, **既可以用于检测模型, 也可以用于ReID模型**. 如果您还没有转换Tensor RT engine, 只需在运行时加上`--trt`参数, 例如:

```bash
python tracker/track.py --dataset mot17 --detector yolox --tracker ocsort --kalman_format ocsort --detector_model_path weights/bytetrack_m_mot17.pth.tar --reid --reid_model shufflenet_v2_x1_0 --reid_model_path shufflenetv2_x1-5666bf0f80.pth --trt
```

如果已有engine, 则直接将相关路径写成engine, 参数`--trt`可以省略:

```bash
python tracker/track.py --dataset visdrone_part --detector b8_ultra --tracker deepsort --kalman_format byte --detector_model_path weights/yolov8l_VisDroneDet_35epochs_20230605.engine --reid deepsort --reid_model_path weights/ckpt.engine
```

### ✅ 评估

马上推出！作为备选项，你可以使用这个repo： [Easier to use TrackEval repo](https://github.com/JackWoo0831/Easier_To_Use_TrackEval).