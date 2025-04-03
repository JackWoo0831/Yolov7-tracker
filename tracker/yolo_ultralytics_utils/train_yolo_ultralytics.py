import torch 
from ultralytics import YOLO 
import numpy as np  

import argparse

def main(args):
    """ main func
    
    """

    model = YOLO(model=args.model_weight)
    model.train(
        data=args.data_cfg,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_sz,
        patience=50,  # epochs to wait for no observable improvement for early stopping of training
        device=args.device,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLO train parser")
    
    parser.add_argument('--model', type=str, default='yolov8s.yaml', help='yaml or pt file')
    parser.add_argument('--model_weight', type=str, default='yolov8s.pt', help='')
    parser.add_argument('--data_cfg', type=str, default='yolov8_utils/data_cfgs/visdrone.yaml', help='')
    parser.add_argument('--epochs', type=int, default=30, help='')
    parser.add_argument('--batch_size', type=int, default=8, help='')
    parser.add_argument('--img_sz', type=int, default=1280, help='')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    args = parser.parse_args()

    main(args)

# python tracker/yolo_ultralytics_utils/train_yolo_ultralytics.py --model_weight weights/yolo11m.pt --data_cfg tracker/yolo_ultralytics_utils/data_cfgs/visdrone_det.yaml --epochs 30 --batch_size 8 --img_sz 1280 --device 0