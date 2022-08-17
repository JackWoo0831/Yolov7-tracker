"""
将VisDrone转换为yolo v5格式
class_id, xc_norm, yc_norm, w_norm, h_norm
"""
import os
import os.path as osp
import argparse
import cv2
import glob

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

def generate_imgs(split_name='VisDrone2019-MOT-train', generate_imgs=True, if_certain_seqs=False, car_only=False):
    """
    产生图片文件夹 例如 VisDrone/images/VisDrone2019-MOT-train/uav0000076_00720_v/000010.jpg
    同时产生序列->高,宽的字典 便于后续

    split: str, 'VisDrone2019-MOT-train', 'VisDrone2019-MOT-val' or 'VisDrone2019-MOT-test-dev'
    if_certain_seqs: bool, use for debug. 
    """

    if not if_certain_seqs:
        seq_list = os.listdir(osp.join(DATA_ROOT, split_name, 'sequences'))  # 所有序列名称
    else:
        seq_list = certain_seqs
    
    if car_only:  # 只跟踪车就忽略行人多的视频
        seq_list = [seq for seq in seq_list if seq not in ignored_seqs]

    # 遍历所有序列 给图片创建软链接 同时更新seq->(w,h)字典
    if_write_txt = True if glob.glob('./visdrone/*.txt') else False
    # if_write_txt = True if not osp.exists(f'./visdrone/.txt') else False  # 是否需要写txt 用于生成visdrone.train

    if not if_write_txt:
        for seq in seq_list:
            img_dir = osp.join(DATA_ROOT, split_name, 'sequences', seq)  # 该序列下所有图片路径 

            imgs = sorted(os.listdir(img_dir))  # 所有图片

            if generate_imgs:
                to_path = osp.join(DATA_ROOT, 'images', split_name, seq)  # 该序列图片存储位置
                if not osp.exists(to_path):
                    os.makedirs(to_path)

                for img in imgs:  # 遍历该序列下的图片
                    os.symlink(osp.join(img_dir, img),
                                osp.join(to_path, img))  # 创建软链接

            img_sample = cv2.imread(osp.join(img_dir, imgs[0]))  # 每个序列第一张图片 用于获取w, h
            w, h = img_sample.shape[1], img_sample.shape[0]  # w, h

            image_wh_dict[seq] = (w, h)  # 更新seq->(w,h) 字典

        # print(image_wh_dict)
        # return
    else:
        with open('./visdrone.txt', 'a') as f:
            for seq in seq_list:
                img_dir = osp.join(DATA_ROOT, split_name, 'sequences', seq)  # 该序列下所有图片路径 

                imgs = sorted(os.listdir(img_dir))  # 所有图片

                if generate_imgs:
                    to_path = osp.join(DATA_ROOT, 'images', split_name, seq)  # 该序列图片存储位置
                    if not osp.exists(to_path):
                        os.makedirs(to_path)

                    for img in imgs:  # 遍历该序列下的图片

                        f.write('VisDrone2019/' + 'VisDrone2019/' + 'images/' + split_name + '/' \
                                + seq + '/' + img + '\n')

                        os.symlink(osp.join(img_dir, img),
                                    osp.join(to_path, img))  # 创建软链接

                img_sample = cv2.imread(osp.join(img_dir, imgs[0]))  # 每个序列第一张图片 用于获取w, h
                w, h = img_sample.shape[1], img_sample.shape[0]  # w, h

                image_wh_dict[seq] = (w, h)  # 更新seq->(w,h) 字典
        f.close()
    if if_certain_seqs:  # for debug
        print(image_wh_dict) 


def generate_labels(split='VisDrone2019-MOT-train', if_certain_seqs=False, car_only=False):
    """
    split: str, 'train', 'val' or 'test'
    if_certain_seqs: bool, use for debug. 
    """
    # from choose_anchors import image_wh_dict
    # print(image_wh_dict)
    if not if_certain_seqs:
        seq_list = os.listdir(osp.join(DATA_ROOT, split, 'sequences'))  # 序列列表
    else:
        seq_list = certain_seqs
    
    if car_only:  # 只跟踪车就忽略行人多的视频
        seq_list = [seq for seq in seq_list if seq not in ignored_seqs]
        category_list = ['4', '5', '6', '9']
    else:
        category_list = [str(i) for i in range(1, 11)]

    # 类别ID 从0开始
    category_dict = {category_list[idx]: idx for idx in range(len(category_list))}
    # 每张图片分配一个txt
    # 要从sequence的txt里分出来
    for seq in seq_list:
        seq_dir = osp.join(DATA_ROOT, split, 'annotations', seq + '.txt')  # 真值文件
        with open(seq_dir, 'r') as f:
            lines = f.readlines()

            for row in lines:
                
                current_line = row.split(',') 

                frame = current_line[0]  # 第几帧
                if current_line[6] == '0' or current_line[7] not in category_list:
                    continue

                to_file = osp.join(DATA_ROOT, 'labels', split, seq)  # 要写入的文件名
                # 如果不存在就创建
                if not osp.exists(to_file):
                    os.makedirs(to_file)
                
                to_file = osp.join(to_file, frame.zfill(7) + '.txt')

                category_id = category_dict[current_line[7]]
                x0, y0 = int(current_line[2]), int(current_line[3])  # 左上角 x y
                w, h = int(current_line[4]), int(current_line[5])  # 宽 高

                x_c, y_c = x0 + w // 2, y0 + h // 2  # 中心点 x y

                image_w, image_h = image_wh_dict[seq][0], image_wh_dict[seq][1]  # 图像高宽
                # 归一化
                w, h = w / image_w, h / image_h
                x_c, y_c = x_c / image_w, y_c / image_h


                with open(to_file, 'a') as f_to:
                    write_line = '{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        category_id, x_c, y_c, w, h)

                    f_to.write(write_line)

                f_to.close()
        

        f.close()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='VisDrone2019-MOT-train', help='train or test')
    parser.add_argument('--generate_imgs', action='store_true', help='whether generate soft link of imgs')
    parser.add_argument('--car_only', action='store_true', help='only cars')
    parser.add_argument('--if_certain_seqs', action='store_true', help='for debug')

    opt = parser.parse_args()
    print('generating images...')
    generate_imgs(opt.split, opt.generate_imgs, opt.if_certain_seqs, opt.car_only)

    print('generating labels...')
    generate_labels(opt.split, opt.if_certain_seqs, opt.car_only)

    print('Done!')


    # python convert_VisDrone_to_yolo.py --split VisDrone2019-MOT-train
    # python convert_VisDrone_to_yolo.py --split VisDrone2019-MOT-train --car_only --if_certain_seqs