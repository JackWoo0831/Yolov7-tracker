# YOLO v7 + 各种tracker实现多目标跟踪

## 注意 
20220927修复了STrack类中update不更新外观的问题, 代码有较大更, **您可能需要重新下载```./tracker```文件夹**. 
尝试加入StrongSORT, 但是目前还不work:(, 尽力调一调

## 亮点  
1. 统一代码风格, 对多种tracker重新整理, 详细注释, 方便阅读, 适合初学者 
2. 多类多目标跟踪 
3. 各种tracker集成在一个文件夹"./tracker/"内, 方便移植到其他detector.  

## 集成的tracker:
SORT,  
DeepSORT,  
ByteTrack([ECCV2022](https://arxiv.org/pdf/2110.06864)),  
DeepMOT([CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_How_to_Train_Your_Deep_Multi-Object_Tracker_CVPR_2020_paper.pdf)),  
BoT-SORT([arxiv2206](https://arxiv.org/pdf/2206.14651.pdf)),   
UAVMOT([CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Multi-Object_Tracking_Meets_Moving_UAV_CVPR_2022_paper.pdf))


## TODO
- [x] 集成UAVMOT([CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Multi-Object_Tracking_Meets_Moving_UAV_CVPR_2022_paper.pdf))
- [ ] 达到更好的结果(缓解类别不平衡, 小目标等等)...
- [ ] MOT challenge数据集
- [ ] 更换Re-ID模型


## 效果
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

## 环境配置  
- python=3.7.0 pytorch=1.7.0 torchvision=0.8.0 cudatoolkit=11.0
- [py-motmetrics](https://github.com/cheind/py-motmetrics)  (`pip install motmetrics`)
- cython-bbox (`pip install cython_bbox`)
- opencv

## 训练

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

## 跟踪  

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

> StrongSORT中OSNet的下载地址, 请参照https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/blob/master/strong_sort/deep/reid_model_factory.py

> 您也可以通过增加
> ```shell
> --save_images --save_videos
> ```
> 来控制保存跟踪结果的图片与视频.  

## 将./tracker应用于其他detector  

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
