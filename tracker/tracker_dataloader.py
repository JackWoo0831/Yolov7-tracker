import numpy as np
import torch 
import cv2 
import os 
import os.path as osp

from torch.utils.data import Dataset


class TestDataset(Dataset):
    """ This class generate origin image, preprocessed image for inference
        NOTE: for every sequence, initialize a TestDataset class

    """

    def __init__(self, data_root, split, seq_name, img_size=[640, 640], legacy_yolox=True, model='yolox', **kwargs) -> None:
        """
        Args:
            data_root: path for entire dataset
            seq_name: name of sequence
            img_size: List[int, int] | Tuple[int, int] image size for detection model 
            legacy_yolox: bool, to be compatible with older versions of yolox
            model: detection model, currently support x, v7, v8
        """
        super().__init__()

        self.model = model

        self.data_root = data_root
        self.seq_name = seq_name
        self.img_size = img_size 
        self.split = split 

        self.seq_path = osp.join(self.data_root, 'images', self.split, self.seq_name)
        self.imgs_in_seq = sorted(os.listdir(self.seq_path))
        
        self.legacy = legacy_yolox

        self.other_param = kwargs

    def __getitem__(self, idx):
        
        if self.model == 'yolox':
            return self._getitem_yolox(idx)
        elif self.model == 'yolov7':
            return self._getitem_yolov7(idx)
        elif self.model == 'yolov8':
            return self._getitem_yolov8(idx)
    
    def _getitem_yolox(self, idx):

        img = cv2.imread(osp.join(self.seq_path, self.imgs_in_seq[idx])) 
        img_resized, _ = self._preprocess_yolox(img, self.img_size, )
        if self.legacy:
            img_resized = img_resized[::-1, :, :].copy()  # BGR -> RGB
            img_resized /= 255.0
            img_resized -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img_resized /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        return torch.from_numpy(img), torch.from_numpy(img_resized)

    def _getitem_yolov7(self, idx):

        img = cv2.imread(osp.join(self.seq_path, self.imgs_in_seq[idx])) 

        img_resized = self._preprocess_yolov7(img, )  # torch.Tensor

        return torch.from_numpy(img), img_resized
    
    def _getitem_yolov8(self, idx):

        img = cv2.imread(osp.join(self.seq_path, self.imgs_in_seq[idx]))  # (h, w, c)
        # img = self._preprocess_yolov8(img)

        return torch.from_numpy(img), torch.from_numpy(img)


    def _preprocess_yolox(self, img, size, swap=(2, 0, 1)):
        """ convert origin image to resized image, YOLOX-manner

        Args:
            img: np.ndarray
            size: List[int, int] | Tuple[int, int]
            swap: (H, W, C) -> (C, H, W)

        Returns:
            np.ndarray, float
        
        """
        if len(img.shape) == 3:
            padded_img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(size, dtype=np.uint8) * 114

        r = min(size[0] / img.shape[0], size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def _preprocess_yolov7(self, img, ):
        
        img_resized = self._letterbox(img, new_shape=self.img_size, stride=self.other_param['stride'], )[0]
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img_resized = np.ascontiguousarray(img_resized)

        img_resized = torch.from_numpy(img_resized).float()
        img_resized /= 255.0

        return img_resized
    
    def _preprocess_yolov8(self, img, ):

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img) 

        return img


    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
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

    def __len__(self, ):
        return len(self.imgs_in_seq)
    

class DemoDataset(TestDataset):
    """
    dataset for demo
    """
    def __init__(self, file_name, img_size=[640, 640], model='yolox', legacy_yolox=True, **kwargs) -> None:

        self.file_name = file_name
        self.model = model 
        self.img_size = img_size

        self.is_video = '.mp4' in file_name or '.avi' in file_name 

        if not self.is_video:
            self.imgs_in_seq = sorted(os.listdir(file_name))
        else:
            self.imgs_in_seq = []
            self.cap = cv2.VideoCapture(file_name)

            while True:
                ret, frame = self.cap.read()
                if not ret: break

                self.imgs_in_seq.append(frame)

        self.legacy = legacy_yolox

    def __getitem__(self, idx):

        if not self.is_video:
            img = cv2.imread(osp.join(self.file_name, self.imgs_in_seq[idx]))
        else:
            img = self.imgs_in_seq[idx]
        
        if self.model == 'yolox':
            return self._getitem_yolox(img)
        elif self.model == 'yolov7':
            return self._getitem_yolov7(img)
        elif self.model == 'yolov8':
            return self._getitem_yolov8(img)

    def _getitem_yolox(self, img):

        img_resized, _ = self._preprocess_yolox(img, self.img_size, )
        if self.legacy:
            img_resized = img_resized[::-1, :, :].copy()  # BGR -> RGB
            img_resized /= 255.0
            img_resized -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img_resized /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        return torch.from_numpy(img), torch.from_numpy(img_resized)

    def _getitem_yolov7(self, img):

        img_resized = self._preprocess_yolov7(img, )  # torch.Tensor

        return torch.from_numpy(img), img_resized
    
    def _getitem_yolov8(self, img):

        # img = self._preprocess_yolov8(img)

        return torch.from_numpy(img), torch.from_numpy(img)