"""
将UAVDT转换为yolo v5格式
class_id, xc_norm, yc_norm, w_norm, h_norm
"""

import os
import os.path as osp
import argparse
import cv2
import glob
import numpy as np
import random

DATA_ROOT = '/data/wujiapeng/datasets/UAVDT/'

image_wh_dict = {}  # seq->(w,h) 字典 用于归一化

def generate_imgs_and_labels(opts):
    """
    产生图片路径的txt文件以及yolo格式真值文件
    """
    seq_list = os.listdir(osp.join(DATA_ROOT, 'UAV-benchmark-M'))
    print('--------------------------')
    print(f'Total {len(seq_list)} seqs!!')
    # 划分train test
    if opts.random: 
        random.shuffle(seq_list)

    bound = int(opts.ratio * len(seq_list))
    train_seq_list = seq_list[: bound]
    test_seq_list = seq_list[bound:]
    del bound
    print(f'train dataset: {train_seq_list}')
    print(f'test dataset: {test_seq_list}')
    print('--------------------------')
    
    if not osp.exists('./uavdt/'):
        os.makedirs('./uavdt/')

    # 定义类别 UAVDT只有一类
    CATEGOTY_ID = 0  # car

    # 定义帧数范围
    frame_range = {'start': 0.0, 'end': 1.0}
    if opts.half:  # half 截取一半
        frame_range['end'] = 0.5

    # 分别处理train与test
    process_train_test(train_seq_list, frame_range, CATEGOTY_ID, 'train')
    process_train_test(test_seq_list, {'start': 0.0, 'end': 1.0}, CATEGOTY_ID, 'test')
    print('All Done!!')
                

def process_train_test(seqs: list, frame_range: dict, cat_id: int = 0, split: str = 'trian') -> None:
    """
    处理UAVDT的train 或 test
    由于操作相似 故另写函数

    """   

    for seq in seqs:
        print('Dealing with train dataset...')

        img_dir = osp.join(DATA_ROOT, 'UAV-benchmark-M', seq, 'img1')  # 图片路径
        imgs = sorted(os.listdir(img_dir))  # 所有图片的相对路径
        seq_length = len(imgs)  # 序列长度

        # 求解图片高宽
        img_eg = cv2.imread(osp.join(img_dir, imgs[0]))
        w0, h0 = img_eg.shape[1], img_eg.shape[0]  # 原始高宽

        ann_of_seq_path = os.path.join(img_dir, '../', 'gt', 'gt.txt') # GT文件路径
        ann_of_seq = np.loadtxt(ann_of_seq_path, dtype=np.float32, delimiter=',')  # GT内容

        gt_to_path = osp.join(DATA_ROOT, 'labels', split, seq)  # 要写入的真值文件夹
        # 如果不存在就创建
        if not osp.exists(gt_to_path):
            os.makedirs(gt_to_path)

        exist_gts = []  # 初始化该列表 每个元素对应该seq的frame中有无真值框
        # 如果没有 就在train.txt产生图片路径

        for idx, img in enumerate(imgs):
            # img 形如: img000001.jpg
            if idx < int(seq_length * frame_range['start']) or idx > int(seq_length * frame_range['end']):
                continue
            
            # 第一步 产生图片软链接
            # print('step1, creating imgs symlink...')
            if opts.generate_imgs:
                img_to_path = osp.join(DATA_ROOT, 'images', split, seq)  # 该序列图片存储位置

                if not osp.exists(img_to_path):
                    os.makedirs(img_to_path)

                os.symlink(osp.join(img_dir, img),
                                osp.join(img_to_path, img))  # 创建软链接
            
            # 第二步 产生真值文件
            # print('step2, generating gt files...')
            ann_of_current_frame = ann_of_seq[ann_of_seq[:, 0] == float(idx + 1), :]  # 筛选真值文件里本帧的目标信息
            exist_gts.append(True if ann_of_current_frame.shape[0] != 0 else False)

            gt_to_file = osp.join(gt_to_path, img[:-4] + '.txt')

            with open(gt_to_file, 'w') as f_gt:
                for i in range(ann_of_current_frame.shape[0]):    
                    if int(ann_of_current_frame[i][6]) == 1:
                        # bbox xywh 
                        x0, y0 = int(ann_of_current_frame[i][2]), int(ann_of_current_frame[i][3])
                        w, h = int(ann_of_current_frame[i][4]), int(ann_of_current_frame[i][5])

                        xc, yc = x0 + w // 2, y0 + h // 2  # 中心点 x y

                        # 归一化
                        xc, yc = xc / w0, yc / h0
                        w, h = w / w0, h / h0
                        category_id = cat_id

                        write_line = '{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                            category_id, xc, yc, w, h)

                        f_gt.write(write_line)

            f_gt.close()

        # 第三步 产生图片索引train.txt等
        print(f'generating img index file of {seq}')        
        to_file = os.path.join('./uavdt/', split + '.txt')
        with open(to_file, 'a') as f:
            for idx, img in enumerate(imgs):
                if idx < int(seq_length * frame_range['start']) or idx > int(seq_length * frame_range['end']):
                    continue
                
                if exist_gts[idx]:
                    f.write('UAVDT/' + 'images/' + split + '/' \
                            + seq + '/' + img + '\n')

            f.close()

    

if __name__ == '__main__':
    if not osp.exists('./uavdt'):
        os.system('mkdir ./uavdt')
    else:
        os.system('rm -rf ./uavdt/*')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_imgs', action='store_true', help='whether generate soft link of imgs')
    parser.add_argument('--certain_seqs', action='store_true', help='for debug')
    parser.add_argument('--half', action='store_true', help='half frames')
    parser.add_argument('--ratio', type=float, default=0.8, help='ratio of test dataset devide train dataset')
    parser.add_argument('--random', action='store_true', help='random split train and test')

    opts = parser.parse_args()

    generate_imgs_and_labels(opts)
    # python tools/convert_UAVDT_to_yolo.py --generate_imgs --half --random