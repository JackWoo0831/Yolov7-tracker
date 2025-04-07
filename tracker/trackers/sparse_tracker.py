"""
Sparse Track
"""

import numpy as np  
import torch 
from torchvision.ops import nms

import cv2 
import torchvision.transforms as T

from .basetrack import BaseTrack, TrackState
from .tracklet import Tracklet, Tracklet_w_depth
from .matching import *

from .camera_motion_compensation.cmc import GMC

class SparseTracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_tracklets = []  # type: list[Tracklet]
        self.lost_tracklets = []  # type: list[Tracklet]
        self.removed_tracklets = []  # type: list[Tracklet]

        self.frame_id = 0
        self.args = args

        self.det_thresh = args.conf_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size

        self.motion = args.kalman_format            

        # camera motion compensation module
        self.gmc = GMC(method='orb', downscale=2, verbose=None)

        # once init, clear all trackid count to avoid large id
        BaseTrack.clear_count()

    def get_deep_range(self, obj, step):
        col = []
        for t in obj:
            lend = (t.deep_vec)[2]
            col.append(lend)
        max_len, mix_len = max(col), min(col)
        if max_len != mix_len:
            deep_range =np.arange(mix_len, max_len, (max_len - mix_len + 1) / step)
            if deep_range[-1] < max_len:
                deep_range = np.concatenate([deep_range, np.array([max_len],)])
                deep_range[0] = np.floor(deep_range[0])
                deep_range[-1] = np.ceil(deep_range[-1])
        else:    
            deep_range = [mix_len,] 
        mask = self.get_sub_mask(deep_range, col)      
        return mask
    
    def get_sub_mask(self, deep_range, col):
        mix_len=deep_range[0]
        max_len=deep_range[-1]
        if max_len == mix_len:
            lc = mix_len   
        mask = []
        for d in deep_range:
            if d > deep_range[0] and d < deep_range[-1]:
                mask.append((col >= lc) & (col < d)) 
                lc = d
            elif d == deep_range[-1]:
                mask.append((col >= lc) & (col <= d)) 
                lc = d 
            else:
                lc = d
                continue
        return mask
    
    # core function
    def DCM(self, detections, tracks, activated_tracklets, refind_tracklets, levels, thresh, is_fuse):
        if len(detections) > 0:
            det_mask = self.get_deep_range(detections, levels) 
        else:
            det_mask = []

        if len(tracks)!=0:
            track_mask = self.get_deep_range(tracks, levels)
        else:
            track_mask = []

        u_detection, u_tracks, res_det, res_track = [], [], [], []
        if len(track_mask) != 0:
            if  len(track_mask) < len(det_mask):
                for i in range(len(det_mask) - len(track_mask)):
                    idx = np.argwhere(det_mask[len(track_mask) + i] == True)
                    for idd in idx:
                        res_det.append(detections[idd[0]])
            elif len(track_mask) > len(det_mask):
                for i in range(len(track_mask) - len(det_mask)):
                    idx = np.argwhere(track_mask[len(det_mask) + i] == True)
                    for idd in idx:
                        res_track.append(tracks[idd[0]])
        
            for dm, tm in zip(det_mask, track_mask):
                det_idx = np.argwhere(dm == True)
                trk_idx = np.argwhere(tm == True)
                
                # search det 
                det_ = []
                for idd in det_idx:
                    det_.append(detections[idd[0]])
                det_ = det_ + u_detection
                # search trk
                track_ = []
                for idt in trk_idx:
                    track_.append(tracks[idt[0]])
                # update trk
                track_ = track_ + u_tracks
                
                dists = iou_distance(track_, det_)

                matches, u_track_, u_det_ = linear_assignment(dists, thresh)
                for itracked, idet in matches:
                    track = track_[itracked]
                    det = det_[idet]
                    if track.state == TrackState.Tracked:
                        track.update(det_[idet], self.frame_id)
                        activated_tracklets.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refind_tracklets.append(track)
                u_tracks = [track_[t] for t in u_track_]
                u_detection = [det_[t] for t in u_det_]
                
            u_tracks = u_tracks + res_track
            u_detection = u_detection + res_det

        else:
            u_detection = detections
            
        return activated_tracklets, refind_tracklets, u_tracks, u_detection


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

        if len(dets) > 0:
            detections = [Tracklet_w_depth(tlwh, s, cate, motion=self.motion) for
                            (tlwh, s, cate) in zip(dets, scores_keep, cates)]
        else:
            detections = []

        ''' Step 1: Add newly detected tracklets to tracked_tracklets'''
        unconfirmed = []
        tracked_tracklets = []  # type: list[Tracklet]
        for track in self.tracked_tracklets:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracklets.append(track)

        ''' Step 2: First association, with high score detection boxes, depth cascade mathcing'''
        tracklet_pool = joint_tracklets(tracked_tracklets, self.lost_tracklets)

        # Predict the current location with Kalman
        for tracklet in tracklet_pool:
            tracklet.predict()

        # Camera motion compensation
        warp = self.gmc.apply(ori_img, dets)
        self.gmc.multi_gmc(tracklet_pool, warp)
        self.gmc.multi_gmc(unconfirmed, warp)

        # depth cascade matching
        activated_tracklets, refind_tracklets, u_track, u_detection_high = self.DCM(
                                                                                detections, 
                                                                                tracklet_pool, 
                                                                                activated_tracklets,
                                                                                refind_tracklets, 
                                                                                levels=3, 
                                                                                thresh=0.75, 
                                                                                is_fuse=True)  
        
        ''' Step 3: Second association, with low score detection boxes, depth cascade mathcing'''
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [Tracklet_w_depth(tlwh, s, cate, motion=self.motion) for
                          (tlwh, s, cate) in zip(dets_second, scores_second, cates_second)]
        else:
            detections_second = []

        r_tracked_tracklets = [t for t in u_track if t.state == TrackState.Tracked]  

        activated_tracklets, refind_tracklets, u_track, u_detection_sec = self.DCM(
                                                                                detections_second, 
                                                                                r_tracked_tracklets, 
                                                                                activated_tracklets, 
                                                                                refind_tracklets, 
                                                                                levels=3, 
                                                                                thresh=0.3, 
                                                                                is_fuse=False) 
        
        for track in u_track:
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracklets.append(track)


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = u_detection_high
        dists = iou_distance(unconfirmed, detections)
       
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

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
        self.tracked_tracklets, self.lost_tracklets = remove_duplicate_tracklets(self.tracked_tracklets, self.lost_tracklets)
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