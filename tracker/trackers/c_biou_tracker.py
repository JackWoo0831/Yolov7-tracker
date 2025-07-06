"""
C_BIoU Track
"""

import numpy as np
from collections import deque
from .basetrack import BaseTrack, TrackState
from .tracklet import Tracklet, Tracklet_w_bbox_buffer
from .matching import *

# base class
from .basetracker import BaseTracker

class C_BIoUTracker(BaseTracker):
    def __init__(self, args, frame_rate=30):
        
        super().__init__(args, frame_rate=frame_rate)

        # once init, clear all trackid count to avoid large id
        BaseTrack.clear_count()

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
        dets = bboxes[remain_inds]

        cates = categories[remain_inds]
        
        scores_keep = scores[remain_inds]

        if len(dets) > 0:
            '''Detections'''
            detections = [Tracklet_w_bbox_buffer(tlwh, s, cate, motion=self.motion) for
                          (tlwh, s, cate) in zip(dets, scores_keep, cates)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_tracklets'''
        unconfirmed = []
        tracked_tracklets = []  # type: list[Tracklet]
        for track in self.tracked_tracklets:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracklets.append(track)

        ''' Step 2: First association, with small buffer IoU'''
        tracklet_pool = BaseTracker.joint_tracklets(tracked_tracklets, self.lost_tracklets)

        # Predict the current location with Kalman
        for tracklet in tracklet_pool:
            tracklet.predict()

        dists = buffered_iou_distance(tracklet_pool, detections, level=1)
        
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.9)

        for itracked, idet in matches:
            track = tracklet_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        unmatched_tracklets = [tracklet_pool[i] for i in u_track if tracklet_pool[i].state == TrackState.Tracked]
        unmatched_detections = [detections[i] for i in u_detection]

        '''Step 3: Second association, with large buffer IoU'''

        dists = buffered_iou_distance(unmatched_tracklets, unmatched_detections, level=2)

        matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = unmatched_tracklets[itracked]
            det = unmatched_detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [unmatched_detections[i] for i in u_detection]
        dists = buffered_iou_distance(unconfirmed, detections, level=1)

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_tracklets.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracklets.append(track)

        '''Step 4. Inital new tracks'''
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.init_thresh:
                continue
            track.activate(self.frame_id)
            activated_tracklets.append(track)

        ''' Step 5: Update state'''
        for track in self.lost_tracklets:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracklets.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        self.merge_tracklets(activated_tracklets, refind_tracklets, lost_tracklets, removed_tracklets)
        output_tracklets = [track for track in self.tracked_tracklets if track.is_activated]

        return output_tracklets

