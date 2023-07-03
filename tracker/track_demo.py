"""
Only track a video or image seqs, without evaluate
"""

import numpy as np
import torch
import cv2 
from PIL import Image
import tqdm

import argparse
import os
from time import gmtime, strftime
from timer import Timer
import yaml

from basetrack import BaseTracker  # for framework
from deepsort import DeepSORT
from bytetrack import ByteTrack
from deepmot import DeepMOT
from botsort import BoTSORT
from uavmot import UAVMOT
from strongsort import StrongSORT
from c_biou_tracker import C_BIoUTracker

try:  # import package that outside the tracker folder  For yolo v7
    import sys 
    sys.path.append(os.getcwd())
    
    from models.experimental import attempt_load
    from evaluate import evaluate
    from utils.torch_utils import select_device, time_synchronized, TracedModel
    from utils.general import non_max_suppression, scale_coords, check_img_size
    print('Note: running yolo v7 detector')

except:
    pass

SAVE_FOLDER = 'demo_result'  # NOTE: set your save path here
CATEGORY_DICT = {0: 'car'}  # NOTE: set the categories in your videos here, 
# format: class_id(start from 0): class_name

timer = Timer()
seq_fps = []  # list to store time used for every seq

def main(opts):
    TRACKER_DICT = {
        'sort': BaseTracker,
        'deepsort': DeepSORT,
        'bytetrack': ByteTrack,
        'deepmot': DeepMOT,
        'botsort': BoTSORT,
        'uavmot': UAVMOT, 
        'strongsort': StrongSORT, 
        'c_biou': C_BIoUTracker,
    }  # dict for trackers, key: str, value: class(BaseTracker)

    # NOTE: ATTENTION: make kalman and tracker compatible
    if opts.tracker == 'botsort':
        opts.kalman_format = 'botsort'
    elif opts.tracker == 'strongsort':
        opts.kalman_format = 'strongsort'

    """
    1. load model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(opts.model_path, map_location=device)
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()  # for yolo v7
    stride = int(model.stride.max())  # model stride
    opts.img_size = check_img_size(opts.img_size, s=stride)  # check img_size

    if opts.trace:
        print(opts.img_size)
        model = TracedModel(model, device, opts.img_size)
    else:
        model.to(device)
    model.eval()

    """
    2. load videos or images
    """
    obj_name = opts.obj
    # if read video, then put every frame into a queue
    # if read image seqs, the same as video
    resized_images_queue = []  # List[torch.Tensor] store resized images
    images_queue = []  # List[torch.Tensor] store origin images

    # check path
    assert os.path.exists(obj_name), 'the path does not exist! '
    obj, get_next_frame = None, None  # init obj
    if 'mp4' in opts.obj or 'MP4' in opts.obj:  # if it is a video
        obj = cv2.VideoCapture(obj_name) 
        get_next_frame = lambda _ : obj.read()

        if os.path.isabs(obj_name): obj_name = obj_name.split('/')[-1][:-4]
        else: obj_name = obj_name[:-4]
    
    else:  
        obj = my_queue(os.listdir(obj_name))
        get_next_frame = lambda _ : obj.pop_front()

        if os.path.isabs(obj_name): obj_name = obj_name.split('/')[-1]


    """
    3. start tracking
    """
    tracker = TRACKER_DICT[opts.tracker](opts, frame_rate=30, gamma=opts.gamma)  # instantiate tracker  TODO: finish init params
    results = []  # store current seq results
    frame_id = 0

    while True:
        print(f'----------processing frame {frame_id}----------')

        # end condition
        is_valid, img0 = get_next_frame(None)  # img0: (H, W, C)

        if not is_valid: 
            break  # end of reading 

        img, img0 = preprocess_v7(ori_img=img0, model_size=(opts.img_size, opts.img_size), model_stride=stride)

        timer.tic()  # start timing this img
        img = img.unsqueeze(0)  # （C, H, W) -> (bs == 1, C, H, W)
        out = model(img.to(device))  # model forward             
        out = out[0]  # NOTE: for yolo v7

        out = post_process_v7(out, img_size=img.shape[2:], ori_img_size=img0.shape)

        if len(out.shape) == 3:  # case (bs, num_obj, ...)
            # out = out.squeeze()
            # NOTE: assert batch size == 1
            out = out.squeeze(0)
        # remove some low conf detections
        out = out[out[:, 4] > 0.001]
        
    
        # NOTE: yolo v7 origin out format: [xc, yc, w, h, conf, cls0_conf, cls1_conf, ..., clsn_conf]
        cls_conf, cls_idx = torch.max(out[:, 5:], dim=1)
        # out[:, 4] *= cls_conf  # fuse object and cls conf
        out[:, 5] = cls_idx
        out = out[:, :6]
    
        current_tracks = tracker.update(out, img0)  # List[class(STracks)]      


        # save results
        cur_tlwh, cur_id, cur_cls = [], [], []
        for trk in current_tracks:
            bbox = trk.tlwh
            id = trk.track_id
            cls = trk.cls

            # filter low area bbox
            if bbox[2] * bbox[3] > opts.min_area:
                cur_tlwh.append(bbox)
                cur_id.append(id)
                cur_cls.append(cls)
                # results.append((frame_id + 1, id, bbox, cls))

        results.append((frame_id + 1, cur_id, cur_tlwh, cur_cls))
        timer.toc()  # end timing this image
        
        plot_img(img0, frame_id, [cur_tlwh, cur_id, cur_cls], save_dir=os.path.join(SAVE_FOLDER, 'result_images', obj_name))
    
        frame_id += 1

    seq_fps.append(frame_id / timer.total_time)  # cal fps for current seq
    timer.clear()  # clear for next seq
    # thirdly, save results
    # every time assign a different name
    if opts.save_txt: save_results(obj_name, results)

    ## finally, save videos
    save_videos(obj_name)


class my_queue:
    """
    implement a queue for image seq reading
    """
    def __init__(self, arr: list) -> None:
        self.arr = arr 
        self.start_idx = 0

    def push_back(self, item):
        self.arr.append(item)
    
    def pop_front(self):
        ret = cv2.imread(self.arr[self.start_idx])
        self.start_idx += 1
        return not self.is_empty(), ret
    
    def is_empty(self):
        return self.start_idx == len(self.arr)


def post_process_v7(out, img_size, ori_img_size):
    """ post process for v5 and v7
    
    """

    out = non_max_suppression(out, conf_thres=0.01, )[0]
    out[:, :4] = scale_coords(img_size, out[:, :4], ori_img_size, ratio_pad=None).round()

    # out: tlbr, conf, cls

    return out

def preprocess_v7(ori_img, model_size, model_stride):
    """ simple preprocess for a single image
    
    """
    img_resized = _letterbox(ori_img, new_shape=model_size, stride=model_stride)[0]

    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img_resized = np.ascontiguousarray(img_resized)

    img_resized = torch.from_numpy(img_resized).float()
    img_resized /= 255.0

    return img_resized, ori_img

def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def save_results(obj_name, results, data_type='default'):
    """
    write results to txt file

    results: list  row format: frame id, target id, box coordinate, class(optional)
    to_file: file path(optional)
    data_type: write data format
    """
    assert len(results)
    if not data_type == 'default':
        raise NotImplementedError  # TODO

    with open(os.path.join(SAVE_FOLDER, obj_name + '.txt'), 'w') as f:
        for frame_id, target_ids, tlwhs, clses in results:
            if data_type == 'default':

                # f.write(f'{frame_id},{target_id},{tlwh[0]},{tlwh[1]},\
                #             {tlwh[2]},{tlwh[3]},{cls}\n')
                for id, tlwh, cls in zip(target_ids, tlwhs, clses):
                    f.write(f'{frame_id},{id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{int(cls)}\n')
    f.close()

def plot_img(img, frame_id, results, save_dir):
    """
    img: np.ndarray: (H, W, C)
    frame_id: int
    results: [tlwhs, ids, clses]
    save_dir: sr

    plot images with bboxes of a seq
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_ = np.ascontiguousarray(np.copy(img))

    tlwhs, ids, clses = results[0], results[1], results[2]
    for tlwh, id, cls in zip(tlwhs, ids, clses):

        # convert tlwh to tlbr
        tlbr = tuple([int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])])
        # draw a rect
        cv2.rectangle(img_, tlbr[:2], tlbr[2:], get_color(id), thickness=3, )
        # note the id and cls
        text = f'{CATEGORY_DICT[cls]}-{id}'
        cv2.putText(img_, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                        color=(255, 164, 0), thickness=2)

    cv2.imwrite(filename=os.path.join(save_dir, f'{frame_id:05d}.jpg'), img=img_)


def save_videos(obj_name):
    """
    convert imgs to a video

    seq_names: List[str] or str, seqs that will be generated
    """

    if not isinstance(obj_name, list):
        obj_name = [obj_name]

    for seq in obj_name:
        if 'mp4' in seq: seq = seq[:-4]
        images_path = os.path.join(SAVE_FOLDER, 'result_images', seq)
        images_name = sorted(os.listdir(images_path))

        to_video_path = os.path.join(images_path, '../', seq + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        img0 = Image.open(os.path.join(images_path, images_name[0]))
        vw = cv2.VideoWriter(to_video_path, fourcc, 15, img0.size)

        for img in images_name:
            if img.endswith('.jpg'):
                frame = cv2.imread(os.path.join(images_path, img))
                vw.write(frame)
    
    print('Save videos Done!!')


def get_color(idx):
    """
    aux func for plot_seq
    get a unique color for each id
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--obj', type=str, default='demo.mp4', help='video NAME or images FOLDER NAME')

    parser.add_argument('--save_txt', type=bool, default=False, help='whether save txt')

    parser.add_argument('--tracker', type=str, default='sort', help='sort, deepsort, etc')
    parser.add_argument('--model_path', type=str, default='./weights/yolov7_UAVDT_35epochs_20230507.pt', help='model path')
    parser.add_argument('--trace', type=bool, default=False, help='traced model of YOLO v7')

    parser.add_argument('--img_size', type=int, default=1280, help='[train, test] image sizes')

    """For tracker"""
    # model path
    parser.add_argument('--reid_model_path', type=str, default='./weights/ckpt.t7', help='path for reid model path')
    parser.add_argument('--dhn_path', type=str, default='./weights/DHN.pth', help='path of DHN path for DeepMOT')

    # threshs
    parser.add_argument('--conf_thresh', type=float, default=0.1, help='filter tracks')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='thresh for NMS')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IOU thresh to filter tracks')

    # other options
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--gamma', type=float, default=0.1, help='param to control fusing motion and apperance dist')
    parser.add_argument('--kalman_format', type=str, default='default', help='use what kind of Kalman, default, naive, strongsort or bot-sort like')
    parser.add_argument('--min_area', type=float, default=150, help='use to filter small bboxs')

    opts = parser.parse_args()

    if not os.path.exists(SAVE_FOLDER):  # demo save to a particular folder
        os.makedirs(SAVE_FOLDER)
        os.makedirs(os.path.join(SAVE_FOLDER, 'result_images'))
    main(opts)