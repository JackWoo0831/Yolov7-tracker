"""
ImproAssoc
"""

import numpy as np  
import torch 

import cv2 
import torchvision.transforms as T

from .basetrack import BaseTrack, TrackState
from .tracklet import Tracklet, Tracklet_w_reid
from .matching import *

# for reid
from .reid_models.engine import load_reid_model, crop_and_resize, select_device

from .camera_motion_compensation.cmc import GMC

# base class
from .basetracker import BaseTracker

class ImproAssocTracker(BaseTracker):
    def __init__(self, args, frame_rate=30):

        super().__init__(args, frame_rate=frame_rate)

        self.with_reid = args.reid

        self.reid_model = None
        if self.with_reid:
            self.reid_model = load_reid_model(args.reid_model, args.reid_model_path, 
                                              device=args.device, trt=args.trt, crop_size=args.reid_crop_size)
            self.reid_model.eval()            

        # some hyper params
        self._lambda = 0.2  # weight in Eq. 4
        self._o_min = 0.1  # IoU thresh in Eq. 4
        self._o_max = 0.55  # Iou thresh in occlusion aware initialization
        self._d_h_max, self._d_l_max = 0.65, 0.19  # empirically set in paper

        # camera motion compensation module
        self.gmc = GMC(method=args.cmc_method, downscale=2, verbose=None)

        # once init, clear all trackid count to avoid large id
        BaseTrack.clear_count()
    
    def update(self, output_results, img, ori_img):
        """
        output_results: processed detections (scale to original size) tlwh format
        """

        self.frame_id += 1
        activated_tracklets = []
        refind_tracklets = []
        lost_tracklets = []
        removed_tracklets = []

        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
        categories = output_results[:, -1]

        remain_inds = scores > self.args.conf_thresh
        inds_low = scores > self.args.conf_thresh_low
        inds_high = scores < self.args.conf_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]

        cates = categories[remain_inds]
        cates_second = categories[inds_second]
        
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        """Step 1: Extract reid features"""
        if self.with_reid:
            # here we get all detections features
            features_keep = self.get_feature(tlwhs=dets[:, :4], ori_img=ori_img, crop_size=self.args.reid_crop_size)
            features_second = self.get_feature(tlwhs=dets_second[:, :4], ori_img=ori_img, crop_size=self.args.reid_crop_size)

        if len(dets) > 0:
            if self.with_reid:
                detections = [Tracklet_w_reid(tlwh, s, cate, motion=self.motion, feat=feat) for
                            (tlwh, s, cate, feat) in zip(dets, scores_keep, cates, features_keep)]
            else:
                detections = [Tracklet(tlwh, s, cate, motion=self.motion) for
                            (tlwh, s, cate) in zip(dets, scores_keep, cates)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_tracklets'''
        # different from bot sort, improassoc does not handle unconfirmed tracklets
        # unconfirmed = []
        tracked_tracklets = self.tracked_tracklets  # type: list[Tracklet]

        ''' Step 2: Combined Matching'''
        tracklet_pool = BaseTracker.joint_tracklets(tracked_tracklets, self.lost_tracklets)

        # Predict the current location with Kalman
        for tracklet in tracklet_pool:
            tracklet.predict()

        # Camera motion compensation
        warp = self.gmc.apply(ori_img, dets)
        self.gmc.multi_gmc(tracklet_pool, warp)

        d_iou_dists, iou_dists = d_iou_distance(tracklet_pool, detections, return_original_iou=True)

        iou_dists_mask = iou_dists < self._o_min 

        if self.with_reid:
            # mixed cost matrix
            emb_dists = embedding_distance(tracklet_pool, detections)
            dists = self._lambda * (1.0 - d_iou_dists) + (1. - self._lambda) * emb_dists
            dists[iou_dists_mask] = 1.0 
        else:
            dists = 1.0 - d_iou_dists
            dists[iou_dists_mask] = 1.0 

        # init low conf detections
        if len(dets_second) > 0:
            if self.with_reid:
                detections_second = [Tracklet_w_reid(tlwh, s, cate, motion=self.motion, feat=feat) for
                            (tlwh, s, cate, feat) in zip(dets_second, scores_second, cates_second, features_second)]
            else:
                detections_second = [Tracklet(tlwh, s, cate, motion=self.motion) for
                            (tlwh, s, cate) in zip(dets_second, scores_second, cates_second)]
        else:
            detections_second = []

        dists_second = iou_distance(tracklet_pool, detections_second)
        
        # concat D^h and D^l, Eq. 1 in paper
        beta = self._d_h_max / self._d_l_max
        dists_concat = np.hstack([dists, beta * dists_second])
        detections_all = detections + detections_second

        # solve assignment
        matches, u_track, u_detection = linear_assignment(dists_concat, thresh=0.9)

        for itracked, idet in matches:
            track = tracklet_pool[itracked]
            det = detections_all[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        for it in u_track:
            track = tracklet_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracklets.append(track)

        ''' Step 3: occlusion aware initialization'''
        # For each unmatched detection, the IoU with all updated tracks is calculated
        # if the maximum IoU exceeds the overlap threshold omax, the detection is removed

        detections_remain = [detections_all[i] for i in u_detection]
        num_of_detections_remain = len(detections_remain)
        valid_flag = [True for _ in range(num_of_detections_remain)]

        # calculate iou between all tracklets and low conf detections
        # note that the tracklet in tracklet_pool have benn already updated
        iou_matrix = 1. - iou_distance(tracklet_pool, detections_remain)

        # check each row, if any iou > self._o_max, the remain detection is removed
        if iou_matrix.size != 0:
            for col in range(num_of_detections_remain):
                if np.max(iou_matrix[:, col]) > self._o_max:
                    valid_flag[col] = False

        # for valid detections, initialize it
        for it in range(num_of_detections_remain):
            track = detections_remain[it]
            if valid_flag[it] and track.score > self.init_thresh:
                track.activate(self.frame_id)
                activated_tracklets.append(track)
                
        """ Step 4: Update state"""
        for track in self.lost_tracklets:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracklets.append(track)        

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        self.merge_tracklets(activated_tracklets, refind_tracklets, lost_tracklets, removed_tracklets)

        output_tracklets = [track for track in self.tracked_tracklets]

        return output_tracklets

