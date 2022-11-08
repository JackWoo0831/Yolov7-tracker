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
    def __init__(self, path, img_size=1280, format='origin', seq=None) -> None:
        """
        Load images for EACH SEQUENCE
        path: file for img paths(format == 'yolo') or dataset path(format == 'origin')
        img_size: image size for model, tuple or int
        format: 'origin' or 'yolo'. origin for direct read imgs under seqs, yolo for read imgs by train.txt
        """
        super().__init__()
        self.DATA_ROOT = '/data/wujiapeng/datasets/' if format == 'yolo' else path  # to get image
        self.img_files = []
        self.format = format
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
              
        img = cv2.imread(current_img_path)  # (H, W, C)

        assert img is not None, f'Fail to load image{current_img_path}'

        # img_resized, *_ = letterbox(img, self.height, self.width)
        img_resized = cv2.resize(img, (self.width, self.height))

        # convert BGR to RGB and to (C, H, W)
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)

        img_resized = np.ascontiguousarray(img_resized, dtype=np.float32)
        img_resized /= 255.0

        return torch.from_numpy(img_resized), torch.from_numpy(img)


    def __len__(self):
        return len(self.img_files)

