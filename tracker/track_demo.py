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
    print('Note: running yolo v7 detector')

except:
    pass

SAVE_FOLDER = 'demo_result'
CATEGORY_DICT = {0: 'car'}

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

    if opts.trace:
        print(opts.img_size)
        model = TracedModel(model, device, opts.img_size)
    else:
        model.to(device)

    """
    2. load videos or images
    """
    obj_name = opts.obj
    # if read video, then put every frame into a queue
    # if read image seqs, the same as video
    resized_images_queue = []  # List[torch.Tensor] store resized images
    images_queue = []  # List[torch.Tensor] store origin images

    """
    func: resize a frame to target size
    """
    def resize_a_frame(frame, target_size):
        # resize to input to the YOLO net
        frame_resized = cv2.resize(frame, (target_size[0], target_size[1]))  # (H', W', C)
        # convert BGR to RGB and to (C, H, W)
        frame_resized = frame_resized[:, :, ::-1].transpose(2, 0, 1)

        frame_resized = np.ascontiguousarray(frame_resized, dtype=np.float32)
        frame_resized /= 255.0

        frame_resized = torch.from_numpy(frame_resized)

        return frame_resized

    if 'mp4' in opts.obj:  # if it is a video
        assert os.path.exists(obj_name), 'the path does not exist! '
        
        video = cv2.VideoCapture(obj_name) 
        while True:
            result, frame = video.read()  # frame: np.ndarray, shape (H, W, C)
            if not result: break  # end to the video
            frame_resized = resize_a_frame(frame, [opts.img_size, opts.img_size])

            resized_images_queue.append(frame_resized)
            images_queue.append(frame)
    else:  
        assert os.path.exists(obj_name), 'the path does not exist! '
        frames = os.listdir(obj_name)
        for item in frames:
            frame = cv2.imread(item)
            frame_resized = resize_a_frame(frame, [opts.img_size, opts.img_size])

            resized_images_queue.append(frame_resized)
            images_queue.append(frame)


    """
    3. start tracking
    """
    tracker = TRACKER_DICT[opts.tracker](opts, frame_rate=30, gamma=opts.gamma)  # instantiate tracker  TODO: finish init params
    results = []  # store current seq results
    frame_id = 0
    pbar = tqdm.tqdm(desc="demo--", ncols=80)
    for i, (img, img0) in enumerate(zip(resized_images_queue, images_queue)):
        pbar.update()
        timer.tic()  # start timing this img
        img = img.unsqueeze(0)  # （C, H, W) -> (bs == 1, C, H, W)
        out = model(img.to(device))  # model forward             
        out = out[0]  # NOTE: for yolo v7
    
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
        
        plot_img(img0, frame_id, [cur_tlwh, cur_id, cur_cls], save_dir=os.path.join(SAVE_FOLDER, 'reuslt_images', obj_name))
    
        frame_id += 1

    seq_fps.append(i / timer.total_time)  # cal fps for current seq
    timer.clear()  # clear for next seq
    pbar.close()
    # thirdly, save results
    # every time assign a different name
    if opts.save_txt: save_results(obj_name, '', results)

    ## finally, save videos
    save_videos(obj_name)

    

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
        images_path = os.path.join(SAVE_FOLDER, 'reuslt_images', seq)
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

    parser.add_argument('--obj', type=str, default='M1305.mp4', help='video NAME or images FOLDER NAME')

    parser.add_argument('--save_txt', type=bool, default=False, help='whether save txt')

    parser.add_argument('--tracker', type=str, default='sort', help='sort, deepsort, etc')
    parser.add_argument('--model_path', type=str, default='./weights/best.pt', help='model path')
    parser.add_argument('--trace', type=bool, default=False, help='traced model of YOLO v7')

    parser.add_argument('--img_size', nargs='+', type=int, default=1280, help='[train, test] image sizes')

    """For tracker"""
    # model path
    parser.add_argument('--reid_model_path', type=str, default='./weights/ckpt.t7', help='path for reid model path')
    parser.add_argument('--dhn_path', type=str, default='./weights/DHN.pth', help='path of DHN path for DeepMOT')

    # threshs
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='filter tracks')
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