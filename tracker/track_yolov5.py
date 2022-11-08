"""
main code for track
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

from basetrack import BaseTracker  # for framework
from deepsort import DeepSORT
from bytetrack import ByteTrack
from deepmot import DeepMOT
from botsort import BoTSORT
from uavmot import UAVMOT
from strongsort import StrongSORT

try:  # import package that outside the tracker folder  For yolo v7
    import sys 
    sys.path.append(os.getcwd())
    
    from models.common import DetectMultiBackend
    from evaluate import evaluate
    print('Note: running yolo v5 detector')

except:
    pass

import tracker_dataloader

DATASET_ROOT = '/data/wujiapeng/datasets/VisDrone2019/VisDrone2019'  # your dataset root
# DATASET_ROOT = '/data/wujiapeng/datasets/' 

# CATEGORY_NAMES = ['car']
CATEGORY_NAMES = ['car', 'van', 'truck', 'bus']
# CATEGORY_NAMES = ['pedestrain', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
CATEGORY_DICT = {i: CATEGORY_NAMES[i] for i in range(len(CATEGORY_NAMES))}  # show class

# IGNORE_SEQS = []
IGNORE_SEQS = ['uav0000073_00600_v', 'uav0000088_00290_v', 'uav0000073_04464_v']  # ignore seqs

# NOTE: ONLY for yolo v5 model loader(func DetectMultiBackend)
YAML_DICT = {'visdrone': './data/Visdrone_car.yaml', 
             'uavdt': './data/UAVDT.yaml'}  

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
    }  # dict for trackers, key: str, value: class(BaseTracker)

    # NOTE: ATTENTION: make kalman and tracker compatible
    if opts.tracker == 'botsort':
        opts.kalman_format = 'botsort'
    elif opts.tracker == 'strongsort':
        opts.kalman_format = 'strongsort'

    # NOTE: if save video, you must save image
    if opts.save_videos:
        opts.save_images = True
        
    """
    1. load model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(opts.model_path, device=device, dnn=False, data=YAML_DICT[opts.dataset], fp16=False)
    model.eval()
    # warm up
    model.warmup(imgsz=(1, 3, 640, 640))
    """
    2. load dataset and track
    """
    # track per seq
    # firstly, create seq list
    seqs = []
    if opts.data_format == 'yolo':
        with open(f'./{opts.dataset}/test.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                elems = line.split('/')  # devide path by / in order to get sequence name(elems[-2])
                if elems[-2] not in seqs:
                    seqs.append(elems[-2])

    elif opts.data_format == 'origin':
        DATA_ROOT = os.path.join(DATASET_ROOT, 'VisDrone2019-MOT-test-dev/sequences')
        seqs = os.listdir(DATA_ROOT)
    else:
        raise NotImplementedError
    seqs = sorted(seqs)
    seqs = [seq for seq in seqs if seq not in IGNORE_SEQS]
    print(f'Seqs will be evalueated, total{len(seqs)}:')
    print(seqs)

    # secondly, for each seq, instantiate dataloader class and track
    # every time assign a different folder to store results
    folder_name = strftime("%Y-%d-%m %H:%M:%S", gmtime())
    folder_name = folder_name[5:-3].replace('-', '_')
    folder_name = folder_name.replace(' ', '_')
    folder_name = folder_name.replace(':', '_')
    folder_name = opts.tracker + '_' + folder_name

    for seq in seqs:
        print(f'--------------tracking seq {seq}--------------')

        path = os.path.join(DATA_ROOT, seq) if opts.data_format == 'origin' else os.path.join('./', f'{opts.dataset}', 'test.txt')

        loader = tracker_dataloader.TrackerLoader(path, opts.img_size, opts.data_format, seq)

        data_loader = torch.utils.data.DataLoader(loader, batch_size=1)

        tracker = TRACKER_DICT[opts.tracker](opts, frame_rate=30, gamma=opts.gamma)  # instantiate tracker  TODO: finish init params

        results = []  # store current seq results
        frame_id = 0

        pbar = tqdm.tqdm(desc=f"{seq}", ncols=80)
        for i, (img, img0) in enumerate(data_loader):
            pbar.update()
            timer.tic()  # start timing this img

            if not i % opts.detect_per_frame:  # if it's time to detect

                out = model(img.to(device))  # model forward             
                out = out[0]  # NOTE: for yolo v7
            
                if len(out.shape) == 3:  # case (bs, num_obj, ...)
                    # out = out.squeeze()
                    # NOTE: assert batch size == 1
                    out = out.squeeze(0)
                    img0 = img0.squeeze(0)
                # remove some low conf detections
                out = out[out[:, 4] > 0.001]
                
            
                # NOTE: yolo v7 origin out format: [xc, yc, w, h, conf, cls0_conf, cls1_conf, ..., clsn_conf]
                if opts.det_output_format == 'yolo':
                    cls_conf, cls_idx = torch.max(out[:, 5:], dim=1)
                    # out[:, 4] *= cls_conf  # fuse object and cls conf
                    out[:, 5] = cls_idx
                    out = out[:, :6]
            
                current_tracks = tracker.update(out, img0)  # List[class(STracks)]
            else:  # otherwize
                # make the img shape (bs, C, H, W) as (C, H, W)
                if len(img0.shape) == 4:
                    img0 = img0.squeeze(0)
                current_tracks = tracker.update_without_detection(None, img0)
            
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
            
            if opts.save_images:
                plot_img(img0, frame_id, [cur_tlwh, cur_id, cur_cls], save_dir=os.path.join(DATASET_ROOT, 'reuslt_images', seq))
        
            frame_id += 1

        seq_fps.append(i / timer.total_time)  # cal fps for current seq
        timer.clear()  # clear for next seq
        pbar.close()
        # thirdly, save results
        # every time assign a different name
        save_results(folder_name, seq, results)

        ## finally, save videos
        if opts.save_images and opts.save_videos:
            save_videos(seq_names=seq)

    """
    3. evaluate results
    """
    print(f'average fps: {np.mean(seq_fps)}')
    evaluate(sorted(os.listdir(f'./tracker/results/{folder_name}')), 
                sorted([seq + '.txt' for seq in seqs]), data_type='visdrone', result_folder=folder_name)  



def save_results(folder_name, seq_name, results, data_type='default'):
    """
    write results to txt file

    results: list  row format: frame id, target id, box coordinate, class(optional)
    to_file: file path(optional)
    data_type: write data format
    """
    assert len(results)
    if not data_type == 'default':
        raise NotImplementedError  # TODO

    if not os.path.exists(f'./tracker/results/{folder_name}'):

        os.makedirs(f'./tracker/results/{folder_name}')

    with open(os.path.join('./tracker/results', folder_name, seq_name + '.txt'), 'w') as f:
        for frame_id, target_ids, tlwhs, clses in results:
            if data_type == 'default':

                # f.write(f'{frame_id},{target_id},{tlwh[0]},{tlwh[1]},\
                #             {tlwh[2]},{tlwh[3]},{cls}\n')
                for id, tlwh, cls in zip(target_ids, tlwhs, clses):
                    f.write(f'{frame_id},{id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{int(cls)}\n')
    f.close()

    return folder_name

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
        cv2.rectangle(img_, tlbr[:2], tlbr[2:], get_color(id), thickness=1, )
        # note the id and cls
        text = f'{CATEGORY_DICT[cls]}-{id}'
        cv2.putText(img_, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                        color=(255, 164, 0), thickness=1)

    cv2.imwrite(os.path.join(save_dir, f'{frame_id:05d}.jpg'), img_)


def save_videos(seq_names):
    """
    convert imgs to a video

    seq_names: List[str] or str, seqs that will be generated
    """
    if not isinstance(seq_names, list):
        seq_names = [seq_names]

    for seq in seq_names:
        images_path = os.path.join(DATASET_ROOT, 'reuslt_images', seq)
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

    parser.add_argument('--dataset', type=str, default='visdrone', help='visdrone, or mot')
    parser.add_argument('--data_format', type=str, default='origin', help='format of reading dataset')
    parser.add_argument('--det_output_format', type=str, default='yolo', help='data format of output of detector, yolo or other')

    parser.add_argument('--tracker', type=str, default='bytetrack', help='sort, deepsort, etc')

    parser.add_argument('--model_path', type=str, default=None, help='model path')

    parser.add_argument('--img_size', nargs='+', type=int, default=[1280, 1280], help='[train, test] image sizes')

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

    parser.add_argument('--save_images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save_videos', action='store_true', help='save tracking results (video)')

    # detect per several frames
    parser.add_argument('--detect_per_frame', type=int, default=1, help='choose how many frames per detect')
    

    opts = parser.parse_args()
    
    # for debug
    # evaluate(sorted(os.listdir('./tracker/results/deepmot_17_08_02_38')), 
    #             sorted(os.listdir('./tracker/results/deepmot_17_08_02_38')), data_type='visdrone', result_folder='deepmot_17_08_02_38')  
    main(opts)