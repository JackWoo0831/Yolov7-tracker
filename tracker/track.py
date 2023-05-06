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

import tracker_dataloader
import trackeval

def set_basic_params(cfgs):
    global CATEGORY_DICT, DATASET_ROOT, CERTAIN_SEQS, IGNORE_SEQS, YAML_DICT
    CATEGORY_DICT = cfgs['CATEGORY_DICT']
    DATASET_ROOT = cfgs['DATASET_ROOT']
    CERTAIN_SEQS = cfgs['CERTAIN_SEQS']
    IGNORE_SEQS = cfgs['IGNORE_SEQS']
    YAML_DICT = cfgs['YAML_DICT']


timer = Timer()
seq_fps = []  # list to store time used for every seq
def main(opts, cfgs):
    set_basic_params(cfgs)  # NOTE: set basic path and seqs params first

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

    # NOTE: if save video, you must save image
    if opts.save_videos:
        opts.save_images = True

    """
    1. load model for yolo v7
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = attempt_load(opts.model_path, map_location=device)  # for yolo v7
    stride = int(model.stride.max())  # model stride
    opts.img_size = check_img_size(opts.img_size, s=stride)  # check img_size

    if opts.trace:
        model = TracedModel(model, device, opts.img_size)

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

    if not None in CERTAIN_SEQS: seqs = CERTAIN_SEQS  # if only track some certain seqs
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

        loader = tracker_dataloader.TrackerLoader(path, opts.img_size, opts.data_format, seq, pre_process_method='v7', model_stride=stride)

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
                img0 = img0.squeeze(0)

                # post process
                out = post_process_v7(out, img_size=img.shape[2:], ori_img_size=img0.shape)
            
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
                plot_img(img0, frame_id, [cur_tlwh, cur_id, cur_cls], save_dir=os.path.join(DATASET_ROOT, 'result_images', seq))
        
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
    if opts.track_eval:
        default_eval_config = trackeval.Evaluator.get_default_eval_config()
        default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        yaml_dataset_config = cfgs['TRACK_EVAL']  # read yaml file to read TrackEval configs
        # make sure that seqs is same as 'SEQ_INFO' in yaml
        # delete key in 'SEQ_INFO' which is not in seqs
        seqs_in_cfgs = list(yaml_dataset_config['SEQ_INFO'].keys())
        for k in seqs_in_cfgs:
            if k not in seqs:
                yaml_dataset_config['SEQ_INFO'].pop(k)
        assert len(yaml_dataset_config['SEQ_INFO'].keys()) == len(seqs)
        
        for k in default_dataset_config.keys():
            if k in yaml_dataset_config.keys():  # if the key need to be modified
                default_dataset_config[k] = yaml_dataset_config[k]                

        default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
        config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
        eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
        dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
        metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

        # Run code
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)] if opts.dataset in ['mot', 'uavdt'] else [trackeval.datasets.VisDrone2DBox(dataset_config)]
        metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
            if metric.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric(metrics_config))
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')
        evaluator.evaluate(dataset_list, metrics_list)  
    else:
        evaluate(sorted(os.listdir(f'./tracker/results/{folder_name}')), 
                    sorted([seq + '.txt' for seq in seqs]), data_type='visdrone', result_folder=folder_name)  



def post_process_v7(out, img_size, ori_img_size):
    """ post process for v5 and v7
    
    """

    out = non_max_suppression(out, conf_thres=0.01, )[0]
    out[:, :4] = scale_coords(img_size, out[:, :4], ori_img_size, ratio_pad=None).round()

    # out: tlbr, conf, cls

    return out


def save_results(folder_name, seq_name, results, data_type='mot17'):
    """
    write results to txt file

    results: list  row format: frame id, target id, box coordinate, class(optional)
    to_file: file path(optional)
    data_type: write data format, default or mot17 format.
    """
    assert len(results)

    if not os.path.exists(f'./tracker/results/{folder_name}'):

        os.makedirs(f'./tracker/results/{folder_name}')

    with open(os.path.join('./tracker/results', folder_name, seq_name + '.txt'), 'w') as f:
        for frame_id, target_ids, tlwhs, clses in results:
            if data_type == 'default':
                
                for id, tlwh, cls in zip(target_ids, tlwhs, clses):
                    f.write(f'{frame_id},{id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{int(cls)}\n')
            
            elif data_type == 'mot17':
                for id, tlwh, cls in zip(target_ids, tlwhs, clses):
                    f.write(f'{frame_id},{id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n')
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
        cv2.rectangle(img_, tlbr[:2], tlbr[2:], get_color(id), thickness=3, )
        # note the id and cls
        text = f'{CATEGORY_DICT[cls]}-{id}'
        cv2.putText(img_, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                        color=(255, 164, 0), thickness=2)

    cv2.imwrite(filename=os.path.join(save_dir, f'{frame_id:05d}.jpg'), img=img_)


def save_videos(seq_names):
    """
    convert imgs to a video

    seq_names: List[str] or str, seqs that will be generated
    """
    if not isinstance(seq_names, list):
        seq_names = [seq_names]

    for seq in seq_names:
        images_path = os.path.join(DATASET_ROOT, 'result_images', seq)
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

    parser.add_argument('--tracker', type=str, default='sort', help='sort, deepsort, etc')

    parser.add_argument('--model_path', type=str, default='./weights/best.pt', help='model path')
    parser.add_argument('--trace', type=bool, default=False, help='traced model of YOLO v7')

    parser.add_argument('--img_size', nargs='+', type=int, default=1280, help='[train, test] image sizes')

    """For tracker"""
    # model path
    parser.add_argument('--reid_model_path', type=str, default='./weights/ckpt.t7', help='path for reid model path')
    parser.add_argument('--dhn_path', type=str, default='./weights/DHN.pth', help='path of DHN path for DeepMOT')

    # threshs
    parser.add_argument('--conf_thresh', type=float, default=0.2, help='filter tracks')
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
    
    parser.add_argument('--track_eval', type=bool, default=True, help='Use TrackEval to evaluate')

    opts = parser.parse_args()

    # NOTE: read path of datasets, sequences and TrackEval configs
    with open(f'./tracker/config_files/{opts.dataset}.yaml', 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    
    main(opts, cfgs)
