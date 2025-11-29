"""
main code for track
"""
import sys, os
import numpy as np
import torch
import cv2 
from PIL import Image
from tqdm import tqdm
import yaml 

from loguru import logger 
import argparse

from tracking_utils.envs import select_device
from tracking_utils.tools import *
from tracking_utils.visualization import plot_img, save_video

from tracker_dataloader import TestDataset, DemoDataset

# trackers 
from trackers.byte_tracker import ByteTracker
from trackers.sort_tracker import SortTracker
from trackers.botsort_tracker import BotTracker
from trackers.c_biou_tracker import C_BIoUTracker
from trackers.ocsort_tracker import OCSortTracker
from trackers.deepsort_tracker import DeepSortTracker
from trackers.strongsort_tracker import StrongSortTracker
from trackers.sparse_tracker import SparseTracker
from trackers.ucmc_tracker import UCMCTracker
from trackers.hybridsort_tracker import HybridSortTracker
from trackers.tracktrack_tracker import TrackTrackTracker
from trackers.improassoc_tracker import ImproAssocTracker
from trackers.fast_tracker import FastTracker

# YOLOX modules
try:
    from yolox.exp import get_exp 
    from yolox_utils.postprocess import postprocess_yolox
    from yolox.utils import fuse_model
except Exception as e:
    logger.warning(e)
    logger.warning('Load yolox fail. If you want to use yolox, please check the installation.')
    pass 

# YOLOv7 modules
try:
    sys.path.append(os.getcwd())
    from models.experimental import attempt_load
    from utils.torch_utils import select_device, time_synchronized, TracedModel
    from utils.general import non_max_suppression, scale_coords, check_img_size
    from yolov7_utils.postprocess import postprocess as postprocess_yolov7

except Exception as e:
    logger.warning(e)
    logger.warning('Load yolov7 fail. If you want to use yolov7, please check the installation.')
    pass

# Ultralytics YOLO modules (support YOLOv3 ~ YOLOv12)
try:
    from ultralytics import YOLO
    from yolo_ultralytics_utils.postprocess import postprocess as postprocess_ultralytics

except Exception as e:
    logger.warning(e)
    logger.warning('Load yolov8 fail. If you want to use yolov8, please check the installation.')
    pass

# TensorRT
try:
    from accelerations.tensorrt_tools import TensorRTConverter, TensorRTInference
except Exception as e:
    logger.warning(e)
    logger.warning('Load TensorRT fail. If you want to convert model to TensorRT, please install the packages.')

TRACKER_DICT = {
    'sort': SortTracker, 
    'bytetrack': ByteTracker, 
    'botsort': BotTracker, 
    'c_bioutrack': C_BIoUTracker, 
    'ocsort': OCSortTracker, 
    'deepsort': DeepSortTracker, 
    'strongsort': StrongSortTracker, 
    'sparsetrack': SparseTracker, 
    'ucmctrack': UCMCTracker, 
    'hybridsort': HybridSortTracker, 
    'tracktrack': TrackTrackTracker, 
    'improassoc': ImproAssocTracker, 
    'fasttrack': FastTracker, 
}

def get_args():
    
    parser = argparse.ArgumentParser()

    """general"""
    parser.add_argument('--obj', type=str, required=True, default='demo.mp4', help='video or images folder PATH')

    parser.add_argument('--detector', type=str, default='yolo_ultralytics_v8', help='yolov7, yolox, etc.')
    parser.add_argument('--tracker', type=str, default='sort', help='sort, deepsort, etc')
    parser.add_argument('--reid', action='store_true', help='enable reid model, work in bot, byte, ocsort and hybridsort')
    parser.add_argument('--reid_model', type=str, default='osnet_x0_25', help='osnet or deppsort')

    parser.add_argument('--kalman_format', type=str, default='default', help='use what kind of Kalman, sort, deepsort, byte, etc.')
    parser.add_argument('--img_size', type=int, default=1280, help='image size, [h, w]')
    parser.add_argument('--reid_crop_size', type=int, default=[128, 64], nargs='+', help='crop size in reid model, [h, w]')

    parser.add_argument('--conf_thresh', type=float, default=0.2, help='filter tracks')
    parser.add_argument('--conf_thresh_low', type=float, default=0.1, help='filter low conf detections, used in two-stage association')
    parser.add_argument('--init_thresh', type=float, default=0.3, help='filter new detections, larger than this thresh consider as new tracklet')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='thresh for NMS')

    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    """yolox"""
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--yolox_exp_file', type=str, default='./tracker/yolox_utils/yolox_m.py')

    """model path"""
    parser.add_argument('--detector_model_path', type=str, default='./weights/best.pt', help='model path')
    parser.add_argument('--trace', type=bool, default=False, help='traced model of YOLO v7')
    # other model path
    parser.add_argument('--reid_model_path', type=str, default='./weights/osnet_x0_25.pth', help='path for reid model path')

   
    """other options"""
    parser.add_argument('--fuse_detection_score', action='store_true', help='fuse detection conf with iou score')
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--gamma', type=float, default=0.1, help='param to control fusing motion and apperance dist')
    parser.add_argument('--min_area', type=float, default=150, help='use to filter small bboxs')

    parser.add_argument('--save_dir', type=str, default='track_demo_results')
    parser.add_argument('--save_images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save_videos', action='store_true', help='save tracking results (video)')
    
    parser.add_argument('--track_eval', type=bool, default=True, help='Use TrackEval to evaluate')

    parser.add_argument('--cmc_method', type=str, default='orb', help='feature discriptor in camera motion compensation')

    """camera parameter"""
    parser.add_argument('--camera_parameter_folder', type=str, default='./tracker/cam_param_files', help='folder path of camera parameter files')
    
    """tensorrt options"""
    parser.add_argument('--trt', action='store_true', help='use tensorrt engine to detect and reid')
    
    return parser.parse_args()

def main(args):
    
    """1. set some params"""

    # NOTE: if save video, you must save image
    if args.save_videos:
        args.save_images = True

    save_dir = args.save_dir
    save_obj_name = args.obj.replace('/', '_')  # save seq name w.r.t. the obj name, but replace '/'
    logger.info(f'demo result will be saved in {os.path.join(save_dir, save_obj_name)}.txt')
    logger.info(f"images and videos (if you enable it) will be saved in {os.path.join(save_dir, save_obj_name, 'vis_results')}")

    """2. load detector"""
    device = select_device(args.device)

    if args.detector == 'yolox':

        exp = get_exp(args.yolox_exp_file, None)  # TODO: modify num_classes etc. for specific dataset
        model_img_size = exp.input_size
        model = exp.get_model()
        model.to(device)
        model.eval()

        if args.trt:  # convert trt
            # check if need to convert
            if not args.detector_model_path.endswith('.engine'):
                trt_converter = TensorRTConverter(model, input_shape=[3, *model_img_size], ckpt_path=args.detector_model_path, 
                                                min_opt_max_batch=[1, 1, 1], device=device, load_ckpt=True, ckpt_key='model')
                trt_converter.export()
                model = TensorRTInference(engine_path=trt_converter.trt_model, min_opt_max_batch=[1, 1, 1], device=device)
            else:   
                model = TensorRTInference(engine_path=args.detector_model_path, min_opt_max_batch=[1, 1, 1], device=device)

        else:  # normal load
            logger.info(f"loading detector {args.detector} checkpoint {args.detector_model_path}")
            ckpt = torch.load(args.detector_model_path, map_location=device)
            model.load_state_dict(ckpt['model'])
            logger.info("loaded checkpoint done")
            model = fuse_model(model)
            logger.info(f'Now detector is on device {next(model.parameters()).device}')

        stride = None  # match with yolo v7        

    elif args.detector == 'yolov7':

        if args.trt:
            # check if need to convert
            stride = 32
            model_img_size = check_img_size(args.img_size, s=32)
            if not args.detector_model_path.endswith('.engine'):
                model = attempt_load(args.detector_model_path, map_location=device)
                trt_converter = TensorRTConverter(model, input_shape=[3, *model_img_size], ckpt_path=args.detector_model_path, 
                                                min_opt_max_batch=[1, 1, 1], device=device, load_ckpt=False)
                trt_converter.export()
                model = TensorRTInference(engine_path=trt_converter.trt_model, min_opt_max_batch=[1, 1, 1], device=device)
            else:
                model = TensorRTInference(engine_path=args.detector_model_path, min_opt_max_batch=[1, 1, 1], device=device)

        else:
            logger.info(f"loading detector {args.detector} checkpoint {args.detector_model_path}")
            model = attempt_load(args.detector_model_path, map_location=device)

            # Traced model
            model = TracedModel(model, device=device, img_size=args.img_size)
            # model.half()

            logger.info("loaded checkpoint done")
            logger.info(f'Now detector is on device {next(model.parameters()).device}')

            # get inference img size
            stride = int(model.stride.max())  # model stride
            model_img_size = check_img_size(args.img_size, s=stride)  # check img_size

    elif 'ultra' in args.detector:

        if args.trt:
            # for ultralytics, we use the api provided by official ultralytics
            # check if need to convert
            if not args.detector_model_path.endswith('.engine'):
                model = YOLO(args.detector_model_path)
                model = YOLO(model.export(format="engine"))
            else:
                model = YOLO(args.detector_model_path)

        else:
            logger.info(f"loading detector {args.detector} checkpoint {args.detector_model_path}")
            model = YOLO(args.detector_model_path)

            logger.info("loaded checkpoint done")

        model_img_size = [None, None]  
        stride = None 

    else:
        logger.error(f"detector {args.detector} is not supprted")
        logger.error("If you want to use the yolo v8 by ultralytics, please specify the `--detector` \
                     as the string including the substring `ultra`, \
                     such as `yolo_ultra_v8` or `yolo11_ultralytics`")
        exit(0)

    """3. load sequences"""

    dataset = DemoDataset(file_name=args.obj, img_size=model_img_size, model=args.detector, stride=stride, )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    tracker = TRACKER_DICT[args.tracker](args, )

    process_bar = enumerate(data_loader)
    process_bar = tqdm(process_bar, total=len(data_loader), ncols=150)

    results = []

    """4. Tracking"""

    for frame_idx, (ori_img, img) in process_bar:
        if 'ultra' in args.detector:
            img = img.squeeze(0).cpu().numpy()

        else:
            img = img.to(device)  # (1, C, H, W)
            img = img.float() 

        ori_img = ori_img.squeeze(0)

        # get detector output 
        with torch.no_grad():
            if 'ultra' in args.detector:
                    output = model.predict(img, conf=args.conf_thresh, iou=args.nms_thresh, verbose=False)
            else:
                output = model(img)

        # postprocess output to original scales
        if args.detector == 'yolox':
            output = postprocess_yolox(output, args.num_classes, conf_thresh=args.conf_thresh, 
                                        img=img, ori_img=ori_img)

        elif args.detector == 'yolov7':
            output = postprocess_yolov7(output, args.conf_thresh, args.nms_thresh, img.shape[2:], ori_img.shape)

        elif 'ultra' in args.detector:
            output = postprocess_ultralytics(output)
        
        else: raise NotImplementedError

        # output: (tlbr, conf, cls)
        # convert tlbr to tlwh
        if isinstance(output, torch.Tensor): 
            output = output.detach().cpu().numpy()
        output[:, 2] -= output[:, 0]
        output[:, 3] -= output[:, 1]
        current_tracks = tracker.update(output, img, ori_img.cpu().numpy())
    
        # save results
        cur_tlwh, cur_id, cur_cls, cur_score = [], [], [], []
        for trk in current_tracks:
            bbox = trk.tlwh
            id = trk.track_id
            cls = trk.category
            score = trk.score

            # filter low area bbox
            if bbox[2] * bbox[3] > args.min_area:
                cur_tlwh.append(bbox)
                cur_id.append(id)
                cur_cls.append(cls)
                cur_score.append(score)
                # results.append((frame_id + 1, id, bbox, cls))

        results.append((frame_idx + 1, cur_id, cur_tlwh, cur_cls, cur_score))

        if args.save_images:
            plot_img(img=ori_img, frame_id=frame_idx + 1, results=[cur_tlwh, cur_id, cur_cls], 
                        save_dir=os.path.join(save_dir, save_obj_name, 'vis_results'))

    save_results(save_dir=save_dir, 
                seq_name=save_obj_name,  
                results=results)
    
    if args.save_videos:
        save_video(images_path=os.path.join(save_dir, save_obj_name, 'vis_results'))
        logger.info(f'save video done')

if __name__ == '__main__':

    args = get_args()
        
    main(args)
