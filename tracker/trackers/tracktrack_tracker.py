"""
TrackTrack
"""

import numpy as np  
import torch 
from torchvision.ops import nms, batched_nms

import cv2 
import torchvision.transforms as T

from .basetrack import BaseTrack, TrackState
from .tracklet import Tracklet_w_velocity_four_corner
from .matching import *

# for reid
from .reid_models.engine import load_reid_model, crop_and_resize, select_device

from .camera_motion_compensation.cmc import GMC

# base class
from .basetracker import BaseTracker

class TrackTrackTracker(BaseTracker):
    def __init__(self, args, frame_rate=30):
        
        super().__init__(args, frame_rate=frame_rate)

        self.with_reid = args.reid

        self.reid_model = None
        if self.with_reid:
            self.reid_model = load_reid_model(args.reid_model, args.reid_model_path, 
                                              device=args.device, trt=args.trt, crop_size=args.reid_crop_size)
            self.reid_model.eval()

        # camera motion compensation module
        self.gmc = GMC(method=args.cmc_method, downscale=2, verbose=None)

        self.delta_t = 3  # observations backtrack depth

        # once init, clear all trackid count to avoid large id
        BaseTrack.clear_count()

    @staticmethod
    def _find_deleted_detections(output_results):
        """
        find the high conf deleted detections in NMS, module TPA in paper

        here, we first use NMS with high threshold (i.e., output_results)
        and then use NMS in low threshold to obtain the strict result
        """

        det_xyxy = output_results[:, :4].copy()  # tlwh
        # convert tlwh to xyxy
        det_xyxy[:, 2:] += det_xyxy[:, :2]

        # nms with low thresh
        indicies = batched_nms(
            boxes=torch.from_numpy(det_xyxy.astype(np.float32)), 
            scores=torch.from_numpy(output_results[:, 4].astype(np.float32)), 
            idxs=torch.from_numpy(output_results[:, -1].astype(np.float32)), 
            iou_threshold=0.45
        ).numpy()

        output_results_strict_nms = output_results[indicies]

        # get delelted dets
        indicies_not = np.ones((output_results.shape[0], ), dtype=bool)
        indicies_not[indicies] = False
        output_results_delete = output_results[indicies_not]

        return output_results_strict_nms, output_results_delete
    
    @staticmethod
    def k_previous_obs(observations, cur_age, k):
        if len(observations) == 0:
            return [-1, -1, -1, -1, -1]
        for i in range(k):
            dt = k - i
            if cur_age - dt in observations:
                return observations[cur_age - dt]
        max_age = max(observations.keys())
        return observations[max_age]
    
    def track_aware_initialization(self, new_dets, tracklets_not_removed):
        
        iou_matrix = 1. - iou_distance(tracklets_not_removed + new_dets, 
                                       tracklets_not_removed + new_dets)
        scores = np.array([d.score for d in new_dets])

        allow_indices = self._track_aware_nms(iou_matrix, scores, len(tracklets_not_removed), 
                                              nms_thresh=0.55, score_thresh=self.init_thresh)

        return allow_indices

    @staticmethod
    def _track_aware_nms(sim_matrix, scores, num_tracks, nms_thresh, score_thresh):
        # Initialization
        num_dets = len(sim_matrix) - num_tracks
        allow_indices = np.ones(num_dets) * (scores > score_thresh)

        # Run
        for idx in range(num_dets):
            # Check 1
            if allow_indices[idx] == 0:
                continue

            # Check 2
            if num_tracks > 0:
                if np.max(sim_matrix[num_tracks + idx, :num_tracks]) > nms_thresh:
                    allow_indices[idx] = 0
                    continue

            # Check 3
            for jdx in range(num_dets):
                if idx != jdx and allow_indices[jdx] == 1 and scores[idx] > scores[jdx]:
                    if sim_matrix[num_tracks + idx, num_tracks + jdx] > nms_thresh:
                        allow_indices[jdx] = 0

        return allow_indices == 1


    def update(self, output_results, img, ori_img):
        """
        output_results: processed detections (scale to original size) tlwh format
        """

        self.frame_id += 1
        activated_tracklets = []
        refind_tracklets = []
        lost_tracklets = []
        removed_tracklets = []

        # get deleted dets, module TPA in paper
        output_results, output_results_delete = self._find_deleted_detections(output_results)

        # devide detections
        # output_results
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

        # output_results_delete
        scores_delete = output_results_delete[:, 4]
        bboxes_delete = output_results_delete[:, :4]
        categories_delete = output_results_delete[:, -1]

        remain_inds = scores_delete > self.args.conf_thresh

        dets_delete = bboxes_delete[remain_inds]
        scores_delete = scores_delete[remain_inds]
        cates_delete = categories_delete[remain_inds]


        """Step 1: Extract reid features for all detections"""
        if self.with_reid:
            features_keep = self.get_feature(tlwhs=dets[:, :4], ori_img=ori_img, crop_size=self.args.reid_crop_size)
            features_second = self.get_feature(tlwhs=dets_second[:, :4], ori_img=ori_img, crop_size=self.args.reid_crop_size)
            features_delete = self.get_feature(tlwhs=dets_delete[:, :4], ori_img=ori_img, crop_size=self.args.reid_crop_size)

        # initalize all detections
        if len(dets) > 0:
            if self.with_reid:
                detections = [Tracklet_w_velocity_four_corner(tlwh, s, cate, motion=self.motion, feat=feat, enable_state_new=True) for
                            (tlwh, s, cate, feat) in zip(dets, scores_keep, cates, features_keep)]
            else:
                detections = [Tracklet_w_velocity_four_corner(tlwh, s, cate, motion=self.motion, enable_state_new=True) for
                            (tlwh, s, cate) in zip(dets, scores_keep, cates)]
        else:
            detections = []

        if len(dets_second) > 0:
            if self.with_reid:
                detections_second = [Tracklet_w_velocity_four_corner(tlwh, s, cate, motion=self.motion, feat=feat, enable_state_new=True) for
                            (tlwh, s, cate, feat) in zip(dets_second, scores_second, cates_second, features_second)]
            else:
                detections_second = [Tracklet_w_velocity_four_corner(tlwh, s, cate, motion=self.motion, enable_state_new=True) for
                          (tlwh, s, cate) in zip(dets_second, scores_second, cates_second)]
        else:
            detections_second = []

        if len(dets_delete) > 0:
            if self.with_reid:
                detections_delete = [Tracklet_w_velocity_four_corner(tlwh, s, cate, motion=self.motion, feat=feat, enable_state_new=True) for
                            (tlwh, s, cate, feat) in zip(dets_delete, scores_delete, cates_delete, features_delete)]
            else:
                detections_delete = [Tracklet_w_velocity_four_corner(tlwh, s, cate, motion=self.motion, enable_state_new=True) for
                          (tlwh, s, cate) in zip(dets_delete, scores_delete, cates_delete)]
        else:
            detections_delete = []

        ''' Step 2: First association, (tracked and lost tracks) & (high confidence detections)'''

        tracklet_tracked_and_lost = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        tracklet_tracked_and_lost = BaseTracker.joint_tracklets(tracklet_tracked_and_lost, self.lost_tracklets)
        tracklet_new = [t for t in self.tracked_tracklets if t.state == TrackState.New]

        # get velocities and historical observations of tracklet_tracked_and_lost
        velocities = np.array(
            [trk.get_velocity() for trk in tracklet_tracked_and_lost])  # (N, 4, 2)

        # historical observations
        k_observations = np.array(
            [self.k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in tracklet_tracked_and_lost])

        # Predict the current location with Kalman
        for tracklet in tracklet_tracked_and_lost:
            tracklet.predict()
        for tracklet in tracklet_new:
            tracklet.predict()

        # Camera motion compensation
        warp = self.gmc.apply(ori_img, dets)
        self.gmc.multi_gmc(tracklet_tracked_and_lost, warp)
        self.gmc.multi_gmc(tracklet_new, warp)

        # Associate
        detections_all = detections + detections_second + detections_delete
        matches, u_tracks, u_dets = iterative_assignment(
            tracklets=tracklet_tracked_and_lost, 
            dets=detections, 
            dets_second=detections_second, 
            dets_delete=detections_delete, 
            velocities=velocities, 
            previous_obs=k_observations, 
            # default setting in TrackTrack
            match_thresh=0.8, penalty_p=0.2, penalty_q=0.4, reduce_step=0.05, with_reid=self.with_reid
        )

        # update matched tracklets
        for itracked, idet in matches:
            track = tracklet_tracked_and_lost[itracked]
            det = detections_all[idet]
            if track.state in [TrackState.Tracked, TrackState.New]:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        # mark lost to unmatched tracklets
        for it in u_tracks:
            track = tracklet_tracked_and_lost[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracklets.append(track)  

        # get remained high conf dets
        detections_left = [detections_all[i] for i in u_dets if i < len(detections)]

        ''' Step 3: Second association, (new tracks) & (left high confidence detections)'''
        # get velocities and historical observations of tracklet_new
        velocities = np.array(
            [trk.get_velocity() for trk in tracklet_new])  # (N, 4, 2)

        # historical observations
        k_observations = np.array(
            [self.k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in tracklet_new])

        matches, u_tracks, u_dets = iterative_assignment(
            tracklets=tracklet_new, 
            dets=detections_left, 
            dets_second=[], 
            dets_delete=[], 
            velocities=velocities, 
            previous_obs=k_observations, 
            # default setting in TrackTrack
            match_thresh=0.8, penalty_p=0.2, penalty_q=0.4, reduce_step=0.05, with_reid=self.with_reid
        )

        # update matched tracklets
        for itracked, idet in matches:
            track = tracklet_new[itracked]
            det = detections[idet]
            if track.state in [TrackState.Tracked, TrackState.New]:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        # mark remove to unmatched tracklets
        for it in u_tracks:
            track = tracklet_new[it]
            track.mark_removed()
            removed_tracklets.append(track)

        # mark remove of too old tracklets
        for track in self.lost_tracklets:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracklets.append(track)

        '''Step 4: Inital new tracklets, track-aware initalization'''
        # possible new tracklets, unmatched high-conf dets
        new_dets = [detections_left[i] for i in u_dets]
        # tracked, new, lost
        tracklets_not_removed = [t for t in self.tracked_tracklets if t.state != TrackState.Removed]  
        tracklets_not_removed += [t for t in self.lost_tracklets if t.state != TrackState.Removed]
        
        allow_indices = self.track_aware_initialization(new_dets, tracklets_not_removed)
        for idx, flag in enumerate(allow_indices):
            if flag:  # should be initialized as new tracklet
                track = new_dets[idx]
                track.activate(self.frame_id)
                activated_tracklets.append(track)

        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state in [TrackState.Tracked, TrackState.New]]
        self.merge_tracklets(activated_tracklets, refind_tracklets, lost_tracklets, removed_tracklets)

        output_tracklets = [track for track in self.tracked_tracklets if track.state == TrackState.Tracked]

        return output_tracklets

