import cv2 
import os 
import numpy as np 
from PIL import Image

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

    assert img is not None

    if len(img.shape) > 3:
        img = img.squeeze(0)

    img_ = np.ascontiguousarray(np.copy(img))

    tlwhs, ids, clses = results[0], results[1], results[2]
    for tlwh, id, cls in zip(tlwhs, ids, clses):

        # convert tlwh to tlbr
        tlbr = tuple([int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])])
        # draw a rect
        cv2.rectangle(img_, tlbr[:2], tlbr[2:], get_color(id), thickness=3, )
        # note the id and cls
        text = f'{int(cls)}_{id}'
        cv2.putText(img_, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                        color=(255, 164, 0), thickness=2)

    cv2.imwrite(filename=os.path.join(save_dir, f'{frame_id:05d}.jpg'), img=img_)

def get_color(idx):
    """
    aux func for plot_seq
    get a unique color for each id
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def save_video(images_path):
    """
    save images (frames) to a video
    """

    images_list = sorted(os.listdir(images_path))
    save_video_path = os.path.join(images_path, images_path.split('/')[-1] + '.mp4')

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    img0 = Image.open(os.path.join(images_path, images_list[0]))
    vw = cv2.VideoWriter(save_video_path, fourcc, 15, img0.size)

    for image_name in images_list:
        image = cv2.imread(filename=os.path.join(images_path, image_name))
        vw.write(image)
