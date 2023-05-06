import torch 
import os 
import cv2 
import numpy as np 


def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular 
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height)/shape[0], float(width)/shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio)) # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


class TrackerLoader(torch.utils.data.Dataset):
    def __init__(self, path, img_size=1280, format='origin', seq=None, pre_process_method='v5',
                 model_stride=32) -> None:
        """
        Load images for EACH SEQUENCE

        path: file for img paths(format == 'yolo') or dataset path(format == 'origin')
        img_size: image size for model, tuple or int
        format: 'origin' or 'yolo'. origin for direct read imgs under seqs, yolo for read imgs by train.txt
        pre_process_method: how to resize origin image
        model_stride: stride of the model, only valid for v5 or v7
        """
        super().__init__()
        self.DATA_ROOT = '/data/wujiapeng/datasets/' if format == 'yolo' else path  # to get image
        self.img_files = []
        self.format = format
        self.pre_process_method = pre_process_method
        self.model_stride = model_stride

        if format == 'origin':
            assert os.path.isdir(path), f'your path is {path}, path must be your dataset path'
           
            self.img_files = sorted(os.listdir(path))  # add relative path

        elif format == 'yolo':  
            assert os.path.isfile(path), f'your path is {path}, path must be your path file'
            with open(path, 'r') as f:
                lines = f.readlines()
            
                for line in lines:
                    line = line.strip()
                    elems = line.split('/')
                    if elems[-2] in seq:  # 
                        self.img_files.append(os.path.join(self.DATA_ROOT, line))  # add abs path

                    
        assert self.img_files is not None
        
        if type(img_size) == int:
            self.width, self.height = img_size, img_size
        elif type(img_size) == list or type(img_size) == tuple:
            self.width, self.height = img_size[0], img_size[1]


    def __getitem__(self, index):
        """
        return: img after resize and origin image, class(torch.Tensor)
        """

        current_img_path = self.img_files[index]  # current image path
        if self.format == 'origin':
            current_img_path = os.path.join(self.DATA_ROOT, current_img_path)
              
        ori_img = cv2.imread(current_img_path)  # (H, W, C)

        assert ori_img is not None, f'Fail to load image{current_img_path}'
        
        if self.pre_process_method in ['v5', 'v7']:

            img_resized = self._letterbox(ori_img, new_shape=(self.height, self.width), stride=self.model_stride)[0]

            img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img_resized = np.ascontiguousarray(img_resized)

            img_resized = torch.from_numpy(img_resized).float()
            img_resized /= 255.0

        elif self.pre_process_method in ['v8']:
            # NOTE: abort resize step
            # img_resized = cv2.resize(ori_img, (self.height, self.width))
            img_resized = torch.from_numpy(ori_img)

        else:
            raise NotImplementedError


        return img_resized, torch.from_numpy(ori_img)
    

    
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

    def __len__(self):
        return len(self.img_files)

