# YOLO v7 + 各种tracker实现多目标跟踪

## 0. 更新记录

**2023.5.6[大更新]**: 对于v5, v7, 改变前处理和后处理方式(采用原有方式), ***解决了部分边界框近大远小的bug, 边界框更加精确***. 此外, 对于v8, 弃用了resize步骤, 直接推理.

**2023.3.14**解决了`DeepSORT`和`C_BIoUTracker`后面出现的目标不跟踪的bug.  

**2023.2.28**优化了`track_demo.py`, 减少了内存占用.  

**2023.2.24**加入了**推理单个视频或图片文件夹**以及**YOLO v8**的推理功能, 对应的代码为`tracker/track_demo.py`与`tracker/track_yolov8.py`.  推理单个视频或图片文件夹不需要指定数据集与真值, 也没有评测指标的功能, 只需要在命令行中指定`obj`即可, 例如:

```shell
python tracker/track_demo.py --obj demo.mp4 
```
YOLO v8 代码的参数与之前完全相同. 安装YOLO v8以及训练步骤请参照[YOLO v8](https://github.com/ultralytics/ultralytics)  


**2023.2.11**修复了TrackEval路径报错的问题, 详见[issue35](https://github.com/JackWoo0831/Yolov7-tracker/issues/35)  

**2023.2.10**修改了[DeepSORT](https://github.com/JackWoo0831/Yolov7-tracker/blob/master/tracker/deepsort.py)的代码与相关部分代码, 遵循了DeepSORT原论文**级联匹配和余弦距离计算**的原则, 并且解决了原有DeepSORT代码出现莫名漂移跟踪框的问题.  

**2023.1.15**加入了**MOT17数据集**的训练与测试功能, 增加了MOT17转yolo格式的代码(`./tools/convert_MOT17_to_yolo.py`), 转换的过程中强制使得坐标合法, 且忽略了遮挡率>=0.75的目标. 您可以采用此代码转换并训练, 具体请见后面的说明. 在使用tracker的时候, 注意将`data_format`设置为yolo, 这样可以直接根据txt文件的路径读取图片.  

**2023.1.14**加入了当前DanceTrack的SOTA[C_BIoUTracker](https://arxiv.org/pdf/2211.14317v2.pdf), 该论文提出了一种增广的IoU来避免目标的瞬间大范围移动, 且弃用了Kalman滤波. 该代码没有开源, 我是按照自己的理解进行了复现. **有错误非常欢迎指出**.  

**2022.11.26**加入了[TrackEval](https://github.com/JonathonLuiten/TrackEval)评测的方式, 支持MOT, VisDrone和UAVDT三种数据集. 此外将一些路径变量选择了按照`yaml`的方式读取, 尽量让代码可读性高一些. 如果您不想用TrackEval进行评测, 则可以将`track.py`或`track_yolov5.py`的命令配置代码`parser.add_argument('--track_eval', type=bool, default=True, help='Use TrackEval to evaluate')`改为`False`.

**2022.11.10**更新了如何设置数据集路径的说明, 请参见README的`track.py路径读取说明`部分.

**2022.11.09**修复了BoT-SORT中的一处错误[issue 16](https://github.com/JackWoo0831/Yolov7-tracker/issues/16), 加粗了边界框与字体.  

**2022.11.08**更新了track.py, track_yolov5.py, basetrack.py和tracker_dataloader.py, 修复了yolo格式读取数据以及保存视频功能的一些bug, 并增加了隔帧检测的功能(大多数时候用不到). 

**2022.10.22**本代码的匹配代码比较简单, 不一定会达到最好的效果(每次匹配只用一次linear assignment, 没有和历史帧的特征相匹配), 您可以使用cascade matching的方式(参见[StrongSORT](https://github.com/dyhBUPT/StrongSORT/blob/master/deep_sort/tracker.py)的line94-134)  

**2022.10.15**增加了对yolo v5的支持, 只需替换track.py, 将tracker文件夹放到v5的根目录(我测试的是官方的[repo](https://github.com/ultralytics/yolov5))下即可. 代码在[yolo v5](https://github.com/JackWoo0831/Yolov7-tracker/blob/master/tracker/track_yolov5.py). 

**2022.09.27[大更新]**修复了STrack类中update不更新外观的问题, 代码有较大更改, **您可能需要重新下载```./tracker```文件夹**. 
尝试加入StrongSORT, 但是目前还不work:(, 尽力调一调

## 1. 亮点  
1. 统一代码风格, 对多种tracker重新整理, 详细注释, 方便阅读, 适合初学者 
2. 多类多目标跟踪 
3. 各种tracker集成在一个文件夹"./tracker/"内, 方便移植到其他detector.  

## 2. 集成的tracker:
SORT,  
DeepSORT,  
ByteTrack([ECCV2022](https://arxiv.org/pdf/2110.06864)),  
DeepMOT([CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_How_to_Train_Your_Deep_Multi-Object_Tracker_CVPR_2020_paper.pdf)),  
BoT-SORT([arxiv2206](https://arxiv.org/pdf/2206.14651.pdf)),   
UAVMOT([CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Multi-Object_Tracking_Meets_Moving_UAV_CVPR_2022_paper.pdf))  
C_BIoUTracker([arxiv2211](https://arxiv.org/pdf/2211.14317v2.pdf))


## 3. TODO
- [x] 集成UAVMOT([CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Multi-Object_Tracking_Meets_Moving_UAV_CVPR_2022_paper.pdf))
- [ ] 达到更好的结果(缓解类别不平衡, 小目标等等)...
- [x] MOT challenge数据集
- [x] 更换Re-ID模型(更换了OSNet, 效果不好...)


## 4. 效果

### VisDrone

在VisDrone2019-MOT train训练约10 epochs, 采用YOLO v7 w6结构, COCO预训练模型基础上训练. GPU: single Tesla A100, 每个epoch约40min.  
在VisDrone2019-MOT test dev测试, 跟踪所有的类别. 

YOLO v7 VisDrone训练完模型: 
> 链接：https://pan.baidu.com/s/1m13Q8Lx_hrPVFZI6lLDrWQ 
> 提取码：ndkf

| Tracker       | MOTA   | IDF1 | IDS | fps |
|:--------------|:-------:|:------:|:------:|:------:|
|SORT       | **26.4**   | 36.4 |3264 |12.2 |
|DeepSORT  | 16.4   | 33.1 | 1387 | 12.51|
|ByteTrack  | 25.1   | 40.8| 1590 | 14.32|
|DeepMOT  | 15.0  | 24.8|3666 |7.64|
|BoT-SORT  | 23.0 | **41.4**|**1014** |5.41|
|UAVMOT   | 25.0 | 40.5 | 1644 |**18.56**|

> fps具有一定的随机性

![gif](https://github.com/JackWoo0831/Yolov7-tracker/blob/master/test2.gif)

### MOT17

在MOT17 train训练约15epochs, 采用YOLO v7 w6结构, COCO预训练模型基础上训练. GPU: single Tesla A100, 每个epoch约3.5min.

这是在测试集03序列的C_BIoU Tracker的效果:  

![gif](https://github.com/JackWoo0831/Yolov7-tracker/blob/master/C_BIoU2.gif)

## 5. 环境配置  
- python=3.7.0 pytorch=1.7.0 torchvision=0.8.0 cudatoolkit=11.0
- [py-motmetrics](https://github.com/cheind/py-motmetrics)  (`pip install motmetrics`)
- cython-bbox (`pip install cython_bbox`)
- opencv

## 6. 训练

训练遵循YOLO v7的训练方式, 数据集格式可以参照[YOLO v5 train custom data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)  
即数据集文件遵循
```shell
class x_center y_center width height
```  
其中x_center y_center width height必须是**归一化**的.  
如果您训练VisDrone数据集, 可以直接调用:  
```shell
python tools/convert_VisDrone_to_yolov2.py --split_name VisDrone2019-MOT-train --generate_imgs
```  
> 需要您修改一些路径变量.  

准备好数据集后, 假如训练YOLO v7-w6模型(single GPU):  
```shell
python train_aux.py --dataset visdrone --workers 8 --device <$GPU_id$> --batch-size 16 --data data/visdrone_all.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights <$YOLO v7 pretrained model path$> --name yolov7-w6-custom --hyp data/hyp.scratch.custom.yaml
```  
> 更多训练信息参考[YOLO v7](https://github.com/WongKinYiu/yolov7)

## 7. `track.py`路径读取说明  

通常来讲, 一个MOT数据集的目录结构大概遵循`划分(训练、测试等)---序列---图片与标注`的结构, 如下所示:

~~~
{DATASET ROOT}
|-- dataset name
|   |-- train
|   |   |-- sequence name
|   |   |    |--images
|   |   |-- ...
|   |-- val
|   |   |-- ...
|   |-- test
|   |   |-- ...
~~~

因为这个代码是基于YOLO检测器的, 所以您可以遵循数据集本来的格式(data_foramt = origin), 也可以遵循YOLO格式(data_foramt = yolo). 下面分别介绍. 

***1. origin***

origin意味着您直接使用数据集原本的路径, **而不是通过yolo要求的txt格式读取.**  `track.py` 或 `track_yolov5.py`中的**DATA_ROOT变量的值应为具体到序列下面的路径**. 以VisDrone为例, 如果要测试VisDrone2019测试集的视频, 目录为`/data/datasets/VisDrone2019/VisDrone2019-MOT-test-dev`, 在测试集目录下有annotations和sequences两个文件夹, 分别是标注和图片, 则您需要指定DATA_ROOT变量: 

```
DATA_ROOT的值应为/data/datasets/VisDrone2019/VisDrone2019-MOT-test-dev/sequences, 即DATA_ROOT目录下应该为各个视频序列的文件夹.
```

***2. yolo[推荐]***

yolo格式意味着您通过yolo训练时所要求的txt文件读取序列. 我们知道yolo要求txt文件记录图片的路径, 例如:

```
VisDrone2019/images/VisDrone2019-MOT-test-dev/uav0000120_04775_v/0000001.jpg
```

完整的路径是`/data/datasets/VisDrone2019/images/VisDrone2019-MOT-test-dev/uav0000120_04775_v/0000001.jpg`, 我们以`/`分割字符串, 则倒数第二个元素就是序列名称, 所以如果以yolo格式读取数据, 您需要指定以下两处:  

```
1. 读取哪个txt文件. 

2. 在tracker/tracker_dataloader.py的TrackerLoader类中, 指定self.DATA_ROOT属性, 保证和txt中连起来是图片的准确路径.
```

***总之, 可能需要根据不同的数据集调整数据的读取方式. 总的原则是, 要能读清楚有哪些序列, 并且让TrackerLoader的self.img_files变量读到每个图片的路径.***




## 8. 跟踪  



***在跟踪之前***, 您需要选择读取数据的方式, 即`opts.data_format`参数, 如果选择`yolo`格式, 您需要在工程根目录下按照`yolo`的方式(例如本仓库的`./visdrone/test.txt`), 您需要修改`track.py`中的`DATA_ROOT`等变量, 与`test.txt`中的路径配合起来. 如果使用数据集原本的路径, 要根据数据集本身的路径特点进行调整. 一是`track.py`中的路径变量, 二是`track_dataloder.py`中`TrackerLoader`类的初始化函数中的路径.  

> model_path 参数为训练后的detector model, 假设路径为 runs/train/yolov7-w6-custom4/weights/best.pt  

***SORT*** : 
```shell
python tracker/track.py --dataset visdrone --data_format origin --tracker sort --model_path runs/train/yolov7-w6-custom4/weights/best.pt
```  

***DeepSORT***:  
```shell
python tracker/track.py --dataset visdrone --data_format origin --tracker deepsort --model_path runs/train/yolov7-w6-custom4/weights/best.pt
```

***ByteTrack***:  
```shell
python tracker/track.py --dataset visdrone --data_format origin --tracker bytetrack --model_path runs/train/yolov7-w6-custom4/weights/best.pt 
```

***DeepMOT***:  
```shell
python tracker/track.py --dataset visdrone --data_format origin --tracker deepmot --model_path runs/train/yolov7-w6-custom4/weights/best.pt
```

***BoT-SORT***:  
```shell
python tracker/track.py --dataset visdrone --data_format origin --tracker botsort --model_path runs/train/yolov7-w6-custom4/weights/best.pt
```

***UAVMOT***
```shell
python tracker/track.py --dataset visdrone --data_format origin --tracker uavmot --model_path runs/train/yolov7-w6-custom4/weights/best.pt
```

***StrongSORT***(目前有问题 正在修复)
```shell
python tracker/track.py --dataset visdrone --data_format origin --tracker strongsort --model_path runs/train/yolov7-w6-custom4/weights/best.pt --reid_model_path weights/osnet_x1_0.pth
```

***C_BIoUTracker***
```shell
python tracker/track.py --dataset visdrone --data_format origin --tracker c_biou --model_path runs/train/yolov7-w6-custom4/weights/best.pt
```

***MOT17数据集***: 与下面的命令一致:

```shell
python tracker/track.py --dataset mot17 --data_format yolo --tracker ${TRACKER} --model_path ${MODEL_PATH}
```

***推理单个视频或图片序列***:

```shell
python tracker/track_demo.py --obj demo.mp4 
```


> StrongSORT中OSNet的下载地址, 请参照https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/blob/master/strong_sort/deep/reid_model_factory.py

> 您也可以通过增加
> ```shell
> --save_images --save_videos
> ```
> 来控制保存跟踪结果的图片与视频.  

## 9. 将./tracker应用于其他detector  

只需保证detector的输出格式为  
```shell
(batch_size, num_objects, x_center, y_center, width, height, obj_conf, category)
```
或经典的yolo格式
```shell
(batch_size, num_objects, x_center, y_center, width, height, obj_conf, category_conf0, category_conf1, category_conf2, ...)
```
> 注意: 推理的时候batch_size要求为1. 

## 更多运行命令参考 run_yolov7.txt文件
