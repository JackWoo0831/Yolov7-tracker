"""
load reid model according to model name and checkpoint
"""

import os 
import os.path as osp

import pickle 
from functools import partial
import torch 
import torch.nn as nn 

from collections import OrderedDict
from loguru import logger

import cv2
import numpy as np
from .OSNet import *
from .DeepsortReID import Extractor
from .ShuffleNetv2 import *
from .MobileNetv2 import *
from .VehicleNet import ft_net

# All reid models
REID_MODEL_DICT = {
    'osnet_x1_0': osnet_x1_0, 
    'osnet_x0_75': osnet_x0_75, 
    'osnet_x0_5': osnet_x0_5, 
    'osnet_x0_25': osnet_x0_25,
    'shufflenet_v2_x0_5': shufflenet_v2_x0_5, 
    'shufflenet_v2_x1_0': shufflenet_v2_x1_0, 
    'shufflenet_v2_x1_5': shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': shufflenet_v2_x2_0,  
    'mobilenetv2_x1_0': mobilenetv2_x1_0, 
    'mobilenetv2_x1_4': mobilenetv2_x1_4, 
    'vehiclenet': ft_net, 
    'deepsort': Extractor
}


def load_reid_model(reid_model, reid_model_path, device=None):
    """
    load reid model according to model name and checkpoint
    """

    device = select_device(device)

    if not reid_model in REID_MODEL_DICT.keys():
        raise NotImplementedError        
        
    if 'deepsort' in reid_model:
        model = REID_MODEL_DICT[reid_model](reid_model_path, device=device)

    else:
        func = REID_MODEL_DICT[reid_model]
        model = func(num_classes=1, pretrained=False, )
        load_pretrained_weights(model, reid_model_path)
        model.to(device).eval()
    
    return model

def crop_and_resize(bboxes, ori_img, input_format='tlwh', sz=(128, 256), 
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    crop the bounding boxes from original image

    Arguments:
        bboxes: np.ndarray: (n, 4) 
        ori_img: np.ndarray: (h, w, c)
        sz: tuple: (w, h)

    Returns:
        cropped_img: torch.Tensor: (n, c, h, w)
    """
    # clone the bboxes to avoid modifying the original one
    bboxes = bboxes.copy()

    # convert bbox to xyxy first
    if not input_format == 'tlbr':
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    
    img_h, img_w = ori_img.shape[:2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean_array = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_array = torch.tensor(std, device=device).view(1, 3, 1, 1)
    
    num_crops = len(bboxes)
    crops = torch.empty((num_crops, 3, sz[1], sz[0]), 
                        dtype=torch.float, device=device)
    
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = box.round().astype('int')
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
        crop = ori_img[y1:y2, x1:x2]
        
        # Resize and convert color in one step
        crop = cv2.resize(crop, sz, interpolation=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize (convert to [0, 1] by dividing by 255 in batch later)
        crop = torch.from_numpy(crop).to(device, dtype=torch.float)
        crops[i] = torch.permute(crop, (2, 0, 1))  # Change to (C, H, W)
    
    crops = crops / 255.0

    # Normalize the crops as experience
    crops = (crops - mean_array) / std_array
    
    return crops



# auxiliary functions
def load_checkpoint(fpath):
    """
    loads checkpoint
    copy from https://github.com/KaiyangZhou/deep-person-reid
    """
    if fpath is None:
        raise ValueError('File path is None')
    fpath = osp.abspath(osp.expanduser(fpath))
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint

def load_pretrained_weights(model, weight_path):
    """
    load pretrained weights for OSNet
    """
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        logger.warning(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )

def select_device(device):
    """ set device 
    same as the function in tracker/tracking_utils/envs.py
    Args:
        device: str, 'cpu' or '0' or '1,2,3'-like

    Return:
        torch.device
    
    """

    if device == 'cpu':
        logger.info('Use CPU for training')

    elif ',' in device:  # multi-gpu
        logger.error('Multi-GPU currently not supported')
    
    else:
        logger.info(f'set gpu {device}')
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available()

    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    return device