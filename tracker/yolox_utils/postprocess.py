import torch 

from yolox.utils import postprocess

def postprocess_yolox(out, num_classes, conf_thresh, img, ori_img):
    """
    convert out to  -> (tlbr, conf, cls)
    """

    out = postprocess(out, num_classes, conf_thresh, )[0]  # (tlbr, obj_conf, cls_conf, cls)

    if out is None: return out

    # merge conf 
    out[:, 4] *= out[:, 5]
    out[:, 5] = out[:, -1]
    out = out[:, :-1]

    # scale to origin size 

    img_size = [img.shape[-2], img.shape[-1]]  # h, w
    ori_img_size = [ori_img.shape[0], ori_img.shape[1]]  # h0, w0
    img_h, img_w = img_size[0], img_size[1]

    scale = min(float(img_h) / ori_img_size[0], float(img_w) / ori_img_size[1])

    out[:, :4] /= scale 

    return out
