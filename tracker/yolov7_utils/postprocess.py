from utils.general import non_max_suppression, scale_coords

def postprocess(out, conf_thresh, nms_thresh, img_size, ori_img_size):
    """
    Args:
        out: out from v7 model
        det_config: configs
    """
    
    out = out[0]
    out = non_max_suppression(out, conf_thresh, nms_thresh, )[0]
    out[:, :4] = scale_coords(img_size, out[:, :4], ori_img_size, ratio_pad=None).round()

    # out: tlbr, conf, cls

    return out