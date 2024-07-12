"""
set gpus and ramdom seeds
"""

import os
import random
import numpy as np
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

def select_device(device):
    """ set device 
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