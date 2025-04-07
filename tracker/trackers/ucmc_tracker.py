"""
UCMC Track
"""

import os 
import numpy as np
from collections import deque
from .basetrack import BaseTrack, TrackState
from .tracklet import Tracklet_w_UCMC
from .matching import *

class UCMCTracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_tracklets = []  # type: list[Tracklet_w_UCMC]
        self.lost_tracklets = []  # type: list[Tracklet_w_UCMC]
        self.removed_tracklets = []  # type: list[Tracklet_w_UCMC]

        self.frame_id = 0
        self.args = args

        self.det_thresh = args.conf_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size

        self.motion = args.kalman_format

        self.cam_param_file = args.cam_param_file  # NOTE the camera parameter file path (folder of txt)
        # storing the intrisic and extrinsic param of camera
        # this can be obtained by the tool provided by https://github.com/corfyi/UCMCTrack

        Ki, Ko = self._read_cam_param()
        KiKo = np.dot(Ki, Ko)
        
        # The A matrix in Eq. 17
        z0 = 0  # the mapped z axis value in world coordinate, by defualt the objects are on the ground
        # so z0 = 0

        A = np.zeros((3, 3))
        A[:, :2] = KiKo[:, :2]
        A[:, 2] = KiKo[:, 2] * z0 + KiKo[:, 3]
        InvA = np.linalg.inv(A)

        # Set in class `Tracklet_w_UCMC`
        Tracklet_w_UCMC.KiKo = KiKo
        Tracklet_w_UCMC.A = A 
        Tracklet_w_UCMC.InvA = InvA

        # once init, clear all trackid count to avoid large id
        BaseTrack.clear_count()

    def _read_cam_param(self, ):
        """
        read the camera param, borrowed from https://github.com/corfyi/UCMCTrack
        """

        assert os.path.isfile(self.cam_param_file), 'check your camera parameter path'

        R = np.zeros((3, 3))
        T = np.zeros((3, 1))
        IntrinsicMatrix = np.zeros((3, 3))

        with open(self.cam_param_file, 'r') as f_in:
            lines = f_in.readlines()
            
        i = 0
        while i < len(lines):
            if lines[i].strip() == "RotationMatrices":
                i += 1
                for j in range(3):
                    R[j] = np.array(list(map(float, lines[i].split())))
                    i += 1
            elif lines[i].strip() == "TranslationVectors":
                i += 1
                T = np.array(list(map(float, lines[i].split()))).reshape(-1,1)
                T = T / 1000
                i += 1
            elif lines[i].strip() == "IntrinsicMatrix":
                i += 1
                for j in range(3):
                    IntrinsicMatrix[j] = np.array(list(map(float, lines[i].split())))
                    i += 1
            else:
                i += 1


        Ki = np.zeros((3, 4))
        Ki[:, :3] = IntrinsicMatrix

        Ko = np.eye(4)
        Ko[:3, :3] = R
        Ko[:3, 3] = T.flatten()

        return Ki, Ko
        

    def update(self, output_results, img, ori_img):
        """
        output_results: processed detections (scale to original size) tlbr format
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

        if len(dets) > 0:
            '''Detections'''
            detections = [Tracklet_w_UCMC(tlwh, s, cate, motion=self.motion) for
                          (tlwh, s, cate) in zip(dets, scores_keep, cates)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_tracklets'''
        unconfirmed = []
        tracked_tracklets = []  # type: list[Tracklet_w_UCMC]
        for track in self.tracked_tracklets:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracklets.append(track)


        ''' Step 2: First association, with high score detection boxes'''
        tracklet_pool = joint_tracklets(tracked_tracklets, self.lost_tracklets)

        # Predict the current location with Kalman
        for tracklet in tracklet_pool:
            tracklet.predict()

        # maha distance
        dists = np.zeros((len(tracklet_pool), len(detections)))

        if not dists.size == 0:
            for i in range(len(tracklet_pool)):
                for j in range(len(detections)):
                    ground_xy, sigma_ground_xy = detections[j].ground_xy, detections[j].sigma_ground_xy
                    dists[i, j] = tracklet_pool[i].cal_maha_distance(ground_xy, sigma_ground_xy)

        
        matches, u_track, u_detection = linear_assignment(dists, thresh=12)

        for itracked, idet in matches:
            track = tracklet_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)


        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [Tracklet_w_UCMC(tlwh, s, cate, motion=self.motion) for
                          (tlwh, s, cate) in zip(dets_second, scores_second, cates_second)]
        else:
            detections_second = []

        r_tracked_tracklets = [tracklet_pool[i] for i in u_track if tracklet_pool[i].state == TrackState.Tracked]

        # maha distance
        dists = np.zeros((len(r_tracked_tracklets), len(detections_second)))

        if not dists.size == 0:
            for i in range(len(r_tracked_tracklets)):
                for j in range(len(detections_second)):
                    ground_xy, sigma_ground_xy = detections_second[j].ground_xy, detections_second[j].sigma_ground_xy
                    dists[i, j] = r_tracked_tracklets[i].cal_maha_distance(ground_xy, sigma_ground_xy)

        matches, u_track, u_detection_second = linear_assignment(dists, thresh=12)
        for itracked, idet in matches:
            track = r_tracked_tracklets[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        for it in u_track:
            track = r_tracked_tracklets[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracklets.append(track)
                

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]

        # maha distance
        dists = np.zeros((len(unconfirmed), len(detections)))

        if not dists.size == 0:
            for i in range(len(unconfirmed)):
                for j in range(len(detections)):
                    ground_xy, sigma_ground_xy = detections[j].ground_xy, detections[j].sigma_ground_xy
                    dists[i, j] = unconfirmed[i].cal_maha_distance(ground_xy, sigma_ground_xy)
       
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=12)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_tracklets.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracklets.append(track)

        """ Step 4: Init new tracklets"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id)
            activated_tracklets.append(track)

        """ Step 5: Update state"""
        for track in self.lost_tracklets:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracklets.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        self.tracked_tracklets = joint_tracklets(self.tracked_tracklets, activated_tracklets)
        self.tracked_tracklets = joint_tracklets(self.tracked_tracklets, refind_tracklets)
        self.lost_tracklets = sub_tracklets(self.lost_tracklets, self.tracked_tracklets)
        self.lost_tracklets.extend(lost_tracklets)
        self.lost_tracklets = sub_tracklets(self.lost_tracklets, self.removed_tracklets)
        self.removed_tracklets.extend(removed_tracklets)
        # self.tracked_tracklets, self.lost_tracklets = remove_duplicate_tracklets(self.tracked_tracklets, self.lost_tracklets)
        # get scores of lost tracks
        output_tracklets = [track for track in self.tracked_tracklets if track.is_activated]

        return output_tracklets


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


def sub_tracklets(tlista, tlistb):
    tracklets = {}
    for t in tlista:
        tracklets[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if tracklets.get(tid, 0):
            del tracklets[tid]
    return list(tracklets.values())


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