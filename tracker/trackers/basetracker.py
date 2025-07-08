"""
Base class for every tracker, embedded some general codes
like init, get_features and tracklet merge
for code clearity
"""

import numpy as np 
import torch 
from .reid_models.engine import crop_and_resize
from .matching import iou_distance

class BaseTracker(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_tracklets = []  # list[Tracklet]
        self.lost_tracklets = []  # list[Tracklet]
        self.removed_tracklets = []  # list[Tracklet]

        self.frame_id = 0
        self.args = args

        self.init_thresh = args.init_thresh
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size

        self.motion = args.kalman_format

    def update(self, output_results, img, ori_img):
        raise NotImplementedError

    @torch.no_grad()
    def get_feature(self, tlwhs, ori_img, crop_size=[128, 64]):
        """
        get apperance feature of an object
        tlwhs: shape (num_of_objects, 4)
        ori_img: original image, np.ndarray, shape(H, W, C)
        crop_size: List[int, int] | Tuple[int, int]
        """

        if tlwhs.size == 0:
            return np.empty((0, 512))

        crop_bboxes = crop_and_resize(tlwhs, ori_img, input_format='tlwh', sz=(crop_size[1], crop_size[0]))
        features = self.reid_model(crop_bboxes).cpu().numpy()

        return features
    
    
    def merge_tracklets(self, activated_tracklets, refind_tracklets, lost_tracklets, removed_tracklets):
        """
        update tracklets with current association results
        """
        self.tracked_tracklets = BaseTracker.joint_tracklets(self.tracked_tracklets, activated_tracklets)
        self.tracked_tracklets = BaseTracker.joint_tracklets(self.tracked_tracklets, refind_tracklets)
        self.lost_tracklets = BaseTracker.sub_tracklets(self.lost_tracklets, self.tracked_tracklets)
        self.lost_tracklets.extend(lost_tracklets)
        self.lost_tracklets = BaseTracker.sub_tracklets(self.lost_tracklets, self.removed_tracklets)
        self.removed_tracklets.extend(removed_tracklets)
        self.tracked_tracklets, self.lost_tracklets = BaseTracker.remove_duplicate_tracklets(self.tracked_tracklets, self.lost_tracklets)

    @staticmethod
    def joint_tracklets(tlista, tlistb):
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_tracklets(tlista, tlistb):
        tracklets = {}
        for t in tlista:
            tracklets[t.track_id] = t
        for t in tlistb:
            tid = t.track_id
            if tracklets.get(tid, 0):
                del tracklets[tid]
        return list(tracklets.values())

    @staticmethod
    def remove_duplicate_tracklets(trackletsa, trackletsb):
        pdist = iou_distance(trackletsa, trackletsb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = list(), list()
        for p, q in zip(*pairs):
            timep = trackletsa[p].frame_id - trackletsa[p].start_frame
            timeq = trackletsb[q].frame_id - trackletsb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(trackletsa) if not i in dupa]
        resb = [t for i, t in enumerate(trackletsb) if not i in dupb]
        return resa, resb