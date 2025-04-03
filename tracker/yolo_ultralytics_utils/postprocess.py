from ultralytics import YOLO

def postprocess(out):

    out = out[0].boxes
    return out.data