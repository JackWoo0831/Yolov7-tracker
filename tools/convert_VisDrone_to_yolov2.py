"""
将VisDrone转换为yolo v5格式
class_id, xc_norm, yc_norm, w_norm, h_norm

改动:
1. 将产生img和label函数合成一个
2. 增加如果无label就不产生当前img路径的功能
3. 增加half选项 每个视频截取一半
"""
import os
import os.path as osp
import argparse
import cv2
import glob
import numpy as np

DATA_ROOT = '/data/wujiapeng/datasets/VisDrone2019/VisDrone2019'


# 以下两个seqs只跟踪车的时候有用
certain_seqs = ['uav0000071_03240_v', 'uav0000072_04488_v','uav0000072_05448_v', 'uav0000072_06432_v','uav0000124_00944_v','uav0000126_00001_v','uav0000138_00000_v','uav0000145_00000_v','uav0000150_02310_v','uav0000222_03150_v','uav0000239_12336_v','uav0000243_00001_v',
'uav0000248_00001_v','uav0000263_03289_v','uav0000266_03598_v','uav0000273_00001_v','uav0000279_00001_v','uav0000281_00460_v','uav0000289_00001_v','uav0000289_06922_v','uav0000307_00000_v',
'uav0000308_00000_v','uav0000308_01380_v','uav0000326_01035_v','uav0000329_04715_v','uav0000361_02323_v','uav0000366_00001_v']

ignored_seqs = ['uav0000013_00000_v', 'uav0000013_01073_v', 'uav0000013_01392_v',
                'uav0000020_00406_v',  'uav0000079_00480_v',
                'uav0000084_00000_v', 'uav0000099_02109_v', 'uav0000086_00000_v',
                'uav0000073_00600_v', 'uav0000073_04464_v', 'uav0000088_00290_v']

image_wh_dict = {}  # seq->(w,h) 字典 用于归一化

def generate_imgs_and_labels(opts):
    """
    产生图片路径的txt文件以及yolo格式真值文件
    """
    if not opts.certain_seqs:
        seq_list = os.listdir(osp.join(DATA_ROOT, opts.split_name, 'sequences'))  # 所有序列名称
    else:
        seq_list = certain_seqs
    
    if opts.car_only:  # 只跟踪车就忽略行人多的视频
        seq_list = [seq for seq in seq_list if seq not in ignored_seqs]
        category_list = [4, 5, 6, 9]  # 感兴趣的类别编号 List[int]
    else:
        category_list = [i for i in range(1, 11)]

    print(f'Total {len(seq_list)} seqs!!')
    if not osp.exists('./visdrone/'):
        os.makedirs('./visdrone/')

    # 类别ID 从0开始
    category_dict = {category_list[idx]: idx for idx in range(len(category_list))}

    txt_name_dict = {'VisDrone2019-MOT-train': 'train',
                        'VisDrone2019-MOT-val': 'val', 
                        'VisDrone2019-MOT-test-dev': 'test'}  # 产生txt文件名称对应关系

    # 如果已经存在就不写了
    write_txt = False if os.path.isfile(os.path.join('./visdrone', txt_name_dict[opts.split_name] + '.txt')) else True
    print(f'write txt is {write_txt}')

    frame_range = {'start': 0.0, 'end': 1.0}
    if opts.half:  # VisDrone-half 截取一半
        frame_range['end'] = 0.5

    # 以序列为单位进行处理
    for seq in seq_list:
        img_dir = osp.join(DATA_ROOT, opts.split_name, 'sequences', seq)  # 该序列下所有图片路径 

        imgs = sorted(os.listdir(img_dir))  # 所有图片
        seq_length = len(imgs)  # 序列长度

        img_eg = cv2.imread(os.path.join(img_dir, imgs[0]))  # 序列的第一张图 用以计算高宽
        w0, h0 = img_eg.shape[1], img_eg.shape[0]  # 原始高宽

        ann_of_seq_path = os.path.join(DATA_ROOT, opts.split_name, 'annotations', seq + '.txt') # GT文件路径
        ann_of_seq = np.loadtxt(ann_of_seq_path, dtype=np.float32, delimiter=',')  # GT内容

        gt_to_path = osp.join(DATA_ROOT, 'labels', opts.split_name, seq)  # 要写入的真值文件夹
        # 如果不存在就创建
        if not osp.exists(gt_to_path):
            os.makedirs(gt_to_path)

        exist_gts = []  # 初始化该列表 每个元素对应该seq的frame中有无真值框
        # 如果没有 就在train.txt产生图片路径

        for idx, img in enumerate(imgs):
            # img: 相对路径 即 图片名称 0000001.jpg
            if idx < int(seq_length * frame_range['start']) or idx > int(seq_length * frame_range['end']):
                continue

            # 第一步 产生图片软链接
            # print('step1, creating imgs symlink...')
            if opts.generate_imgs:
                img_to_path = osp.join(DATA_ROOT, 'images', opts.split_name, seq)  # 该序列图片存储位置

                if not osp.exists(img_to_path):
                    os.makedirs(img_to_path)

                os.symlink(osp.join(img_dir, img),
                                osp.join(img_to_path, img))  # 创建软链接
            # print('Done!\n')

            # 第二步 产生真值文件
            # print('step2, generating gt files...')

            # 根据本序列的真值文件读取
            # ann_idx = int(ann_of_seq[:, 0]) == idx + 1
            ann_of_current_frame = ann_of_seq[ann_of_seq[:, 0] == float(idx + 1), :]  # 筛选真值文件里本帧的目标信息
            exist_gts.append(True if ann_of_current_frame.shape[0] != 0 else False)

            gt_to_file = osp.join(gt_to_path, img[:-4] + '.txt')

            with open(gt_to_file, 'a') as f_gt:
                for i in range(ann_of_current_frame.shape[0]):
                    
                    category = int(ann_of_current_frame[i][7])
                    if int(ann_of_current_frame[i][6]) == 1 and category in category_list:

                        # bbox xywh 
                        x0, y0 = int(ann_of_current_frame[i][2]), int(ann_of_current_frame[i][3])
                        w, h = int(ann_of_current_frame[i][4]), int(ann_of_current_frame[i][5])

                        xc, yc = x0 + w // 2, y0 + h // 2  # 中心点 x y

                        # 归一化
                        xc, yc = xc / w0, yc / h0
                        w, h = w / w0, h / h0

                        category_id = category_dict[category]

                        write_line = '{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                            category_id, xc, yc, w, h)

                        f_gt.write(write_line)

            f_gt.close()
            # print('Done!\n')
        print(f'img symlink and gt files of seq {seq} Done!')
        # 第三步 产生图片索引train.txt等
        print(f'generating img index file of {seq}')
        if write_txt:
            to_file = os.path.join('./visdrone', txt_name_dict[opts.split_name] + '.txt')
            with open(to_file, 'a') as f:
                for idx, img in enumerate(imgs):
                    if idx < int(seq_length * frame_range['start']) or idx > int(seq_length * frame_range['end']):
                        continue
                    
                    if exist_gts[idx]:
                        f.write('VisDrone2019/' + 'VisDrone2019/' + 'images/' + opts.split_name + '/' \
                                + seq + '/' + img + '\n')

            f.close()

    print('All done!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_name', type=str, default='VisDrone2019-MOT-train', help='train or test')
    parser.add_argument('--generate_imgs', action='store_true', help='whether generate soft link of imgs')
    parser.add_argument('--car_only', action='store_true', help='only cars')
    parser.add_argument('--certain_seqs', action='store_true', help='for debug')
    parser.add_argument('--half', action='store_true', help='half frames')

    opts = parser.parse_args()

    generate_imgs_and_labels(opts)
    # python tools/convert_VisDrone_to_yolov2.py --split_name VisDrone2019-MOT-train --generate_imgs --car_only --half