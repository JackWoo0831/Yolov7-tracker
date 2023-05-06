"""
Copyed from ByteTrack
"""
# from numba import jit

import numpy as np
from collections import OrderedDict
import torch 
from torchvision.ops import nms

from kalman_filter import KalmanFilter, NaiveKalmanFilter, BoTSORTKalmanFilter, NSAKalmanFilter
import matching

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


KALMAN_DICT = {
    'default': KalmanFilter,
    'naive': NaiveKalmanFilter, 
    'botsort': BoTSORTKalmanFilter,
    'strongsort': NSAKalmanFilter, 
}  # different kalman filters

"""
General class to describe a trajectory
"""
class STrack(BaseTrack):
    def __init__(self, cls, tlwh, score, kalman_format='default', 
        feature=None, use_avg_of_feature=True, store_features_budget=100) -> None:
        """
        cls: category of this obj 
        tlwh: positoin   score: conf score 
        kalman_format: choose different state vector of kalman 
        feature: re-id feature 
        use_avg_of_feature: whether to use moving average
        """
        super().__init__()
        # info of this track
        self.cls = cls
        self._tlwh = np.asarray(tlwh, dtype=np.float32)  # init tlwh
        self.score = score
        self.is_activated = False
        self.tracklet_len = 0  

        self.track_id = None
        self.start_frame = None
        self.frame_id = None
        self.time_since_update = None

        self.features = []
        self.store_features_budget = store_features_budget
        self.has_feature = True if feature is not None else False
        self.use_avg_of_feature = use_avg_of_feature
        if feature is not None:
            self.features.append(feature)

        # Kalman filter
        self.kalman_format = kalman_format
        self.kalman = KALMAN_DICT[self.kalman_format]()
        self.mean, self.cov = None, None  # for kalman predict

    # some tool funcs 
    @staticmethod
    def tlbr2tlwh(tlbr):
        """
        convert tlbr to tlwh
        """
        result = np.asarray(tlbr).copy()
        result[2] -= result[0]
        result[3] -= result[1]

        return result

    @staticmethod
    def tlwh2xyah(tlwh):
        """
        convert tlwh to xyah
        """
        result = np.asarray(tlwh).copy()
        result[:2] += result[2:] / 2
        result[2] /= result[-1]
        return result

    @staticmethod
    def tlwh2xyar(tlwh):
        """
        convert tlwh to xyar, r is constant, a is area
        """
        result = np.asarray(tlwh).copy()
        result[:2] += result[2:] / 2
        result[2] *= result[3]
        result[3] = tlwh[-1] / tlwh[-2]

        return result
    
    @staticmethod
    def tlwh2xywh(tlwh):
        """
        convert tlwh to xc, yc, w, h
        """
        result = np.asarray(tlwh).copy()
        result[:2] += result[2:] // 2
        return result

    @staticmethod
    def xywh2tlbr(xywh):
        """
        convert xc, yc, wh to tlbr
        """
        if len(xywh.shape) > 1:  # case shape (N, 4)  used for Tracker update
            result = np.asarray(xywh).copy()
            result[:, :2] -= result[:, 2:] // 2
            result[:, 2:] = result[:, :2] + result[:, 2:]
            result = np.maximum(0.0, result)  # in case exists minus
        else:
            result = np.asarray(xywh).copy()
            result[:2] -= result[2:] // 2
            result[2:] = result[:2] + result[2:] 
            result = np.maximum(0.0, result) 
        return result

    @staticmethod
    def xywh2tlwh(xywh):
        """
        convert xc, yc, wh to tlwh
        """
        if len(xywh.shape) > 1:  
            result = np.asarray(xywh).copy()
            result[:, :2] -= result[:, 2:] // 2
        else:
            result = np.asarray(xywh).copy()
            result[:2] -= result[2:] // 2

        return result

    @property
    # @jit 
    def tlwh(self):
        """
        update current bbox with kalman mean
        """
        if self.mean is None:  # No kalman
            return self._tlwh.copy()

        if self.kalman_format in ['default', 'strongsort']:
            # kalman mean: xc, yc, ar, h   where ar = w / h
            ret = self.mean[:4].copy()
            ret[2] *= ret[3]
            ret[:2] -= ret[2:] / 2
            return ret

        elif self.kalman_format == 'naive':
            # kalman mean: xc, yc, area, ar   where ar = h / w
            ret = self.mean[:4].copy()
            ret[-1] = np.sqrt(ret[-1] * ret[-2])
            ret[-2] /= ret[-1]
            return ret
        elif self.kalman_format == 'botsort':
            # kalman mean: xc, yc, w, h
            ret = self.mean[:4].copy()
            ret[:2] -= ret[2:] / 2
            return ret
        else:
            raise NotImplementedError


    @property
    # @jit
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret


    def activate(self, frame_id):
        """
        init a new track
        """
        self.track_id = BaseTrack.next_id()
        # init kalman
        if self.kalman_format in ['default', 'strongsort']:
            measurement = self.tlwh2xyah(self._tlwh)
        elif self.kalman_format == 'naive':
            measurement = self.tlwh2xyar(self._tlwh)
        elif self.kalman_format == 'botsort':
            measurement = self.tlwh2xywh(self._tlwh)
        # measurement = self.tlwh2xyah(self._tlwh) if self.kalman_format == 'default' else self.tlwh2xyar(self.tlwh)
        self.mean, self.cov = self.kalman.initiate(measurement)

        self.state = TrackState.Tracked

        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        self.time_since_update = 0

    def predict(self):
        """
        kalman predict step
        """
        self.mean, self.cov = self.kalman.predict(self.mean, self.cov)

    @staticmethod
    def multi_predict(stracks, kalman):
        """
        predict many tracks 
        stracks: List[class(STrack)]
        """
        # TODO
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.cov for st in stracks])
            for i, st in enumerate(stracks):  # why??
                if st.state != TrackState.Tracked:
                    multi_mean[i][-1] = 0
            multi_mean, multi_covariance = kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].cov = cov

        for strack in stracks: strack.time_since_update += 1

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        reactivate a lost track
        """
        if self.kalman_format in ['default', 'strongsort']:
            measurement = self.tlwh2xyah(new_track.tlwh)
        elif self.kalman_format == 'naive':
            measurement = self.tlwh2xyar(new_track.tlwh)
        elif self.kalman_format == 'botsort':
            measurement = self.tlwh2xywh(new_track.tlwh)
        self.mean, self.cov = self.kalman.update(
            self.mean, self.cov, measurement
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.time_since_update = 0

    def update(self, new_track, frame_id):
        """
        update a track
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        # update position and score
        new_tlwh = new_track.tlwh
        self.score = new_track.score

        # update kalman
        if self.kalman_format in ['default', 'strongsort']:
            measurement = self.tlwh2xyah(new_tlwh)
        elif self.kalman_format == 'naive':
            measurement = self.tlwh2xyar(new_tlwh)
        elif self.kalman_format == 'botsort':
            measurement = self.tlwh2xywh(new_tlwh)

        if self.kalman_format == 'strongsort':  
            # for strongsort, give larger conf object a smaller std.
            self.mean, self.cov = self.kalman.update(
                self.mean, self.cov, measurement, self.score)
        else:
            self.mean, self.cov = self.kalman.update(
                self.mean, self.cov, measurement)

        # update feature
        if new_track.has_feature:        
            feature = new_track.features[0] / np.linalg.norm(new_track.features[0])  # (512, )
            if self.use_avg_of_feature:
                smooth_feat = 0.9 * self.features[-1] + (1 - 0.9) * feature
                smooth_feat /= np.linalg.norm(smooth_feat)
                self.features = [smooth_feat]  # as new feature
            else:
                self.features.append(feature)
                self.features = self.features[-self.store_features_budget: ]
        

        # update status
        self.state = TrackState.Tracked
        self.is_activated = True

        self.time_since_update = 0
        

"""
a very simple SORT Tracker
"""
class BaseTracker(object):
    def __init__(self, opts, frame_rate=30, *args, **kwargs) -> None:
        self.opts = opts 

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opts.conf_thresh
        self.buffer_size = int(frame_rate / 30.0 * opts.track_buffer)
        self.max_time_lost = self.buffer_size

        self.NMS = True  # whether use NMS

        self.kalman = KALMAN_DICT[self.opts.kalman_format]()

        if isinstance(opts.img_size, int):
            self.model_img_size = [opts.img_size, opts.img_size]
        elif isinstance(opts.img_size, (list, tuple)):
            self.model_img_size = opts.img_size

        self.debug_mode = False
    def update(self, det_results, ori_img):
        """
        this func is called by every time step

        det_results: numpy.ndarray or torch.Tensor, shape(N, 6), 6 includes bbox, conf_score, cls
        ori_img: original image, np.ndarray, shape(H, W, C)
        """
        if isinstance(det_results, torch.Tensor):
            det_results = det_results.detach().cpu().numpy()
        if isinstance(ori_img, torch.Tensor):
            ori_img = ori_img.numpy()

        self.frame_id += 1
        activated_starcks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        """step 1. filter results and init tracks"""
        det_results = det_results[det_results[:, 4] > self.det_thresh]


        if det_results.shape[0] > 0:

            # detections: List[Strack]
            detections = [STrack(cls, STrack.tlbr2tlwh(tlbr), score, kalman_format=self.opts.kalman_format)
                            for (cls, tlbr, score) in zip(det_results[:, -1], det_results[:, :4], det_results[:, 4])]

        else:
            detections = []

        # Do some updates
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        """step 2. association with IoU"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(stracks=strack_pool, kalman=self.kalman)
        # cal IoU dist
        IoU_mat = matching.iou_distance(strack_pool, detections)
        
        matched_pair, u_track, u_detection = matching.linear_assignment(IoU_mat, thresh=self.opts.iou_thresh)

        for itracked, idet in matched_pair:  # for those who matched successfully
            track = strack_pool[itracked]
            det = detections[idet]

            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """Step 3. mark unmatched track lost"""
        for itracked in u_track:  # for unmatched track
            track = strack_pool[itracked]
            if track.state == TrackState.Tracked:
                track.mark_lost()
                lost_stracks.append(track)

        """Step 3'. match unconfirmed tracks"""
        u_det = [detections[i] for i in u_detection]
        IoU_mat = matching.iou_distance(unconfirmed, u_det)
        matched_pair1, u_track1, u_detection1 = matching.linear_assignment(IoU_mat, thresh=self.opts.iou_thresh + 0.1)
        for itracked, idet in matched_pair1:
            track = unconfirmed[itracked]
            det = u_det[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        for u_itracked in u_track1:
            track = unconfirmed[u_itracked]
            track.mark_removed()
            removed_stracks.append(track)
        

        """Step 4. init new track"""
        for idet in u_detection1:  # for unmatched detection
            newtrack = u_det[idet]
            if newtrack.score > self.det_thresh + 0.1:  # conf is enough high
                newtrack.activate(self.frame_id)
                activated_starcks.append(newtrack)
        """Step 5. remove long lost tracks"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # update all
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)


        # print
        if self.debug_mode:
            print('===========Frame {}=========='.format(self.frame_id))
            print('Activated: {}'.format([track.track_id for track in activated_starcks]))
            print('Refind: {}'.format([track.track_id for track in refind_stracks]))
            print('Lost: {}'.format([track.track_id for track in lost_stracks]))
            print('Removed: {}'.format([track.track_id for track in removed_stracks]))
        return [track for track in self.tracked_stracks if track.is_activated]

    def update_without_detection(self, det_results, ori_img):
        """
        update tracks when no detection
        only predict current tracks
        """
        if isinstance(ori_img, torch.Tensor):
            ori_img = ori_img.numpy()

        self.frame_id += 1
        activated_starcks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        """step 1. init tracks"""

        # Do some updates
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        """step 2. predict Kalman without updating"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(stracks=strack_pool, kalman=self.kalman)

        # update all
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)


        # print
        if self.debug_mode:
            print('===========Frame {}=========='.format(self.frame_id))
            print('Activated: {}'.format([track.track_id for track in activated_starcks]))
            print('Refind: {}'.format([track.track_id for track in refind_stracks]))
            print('Lost: {}'.format([track.track_id for track in lost_stracks]))
            print('Removed: {}'.format([track.track_id for track in removed_stracks]))
        return [track for track in self.tracked_stracks if track.is_activated]
          

def joint_stracks(tlista, tlistb):
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

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist<0.15)
    dupa, dupb = list(), list()
    for p,q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i,t in enumerate(stracksa) if not i in dupa]
    resb = [t for i,t in enumerate(stracksb) if not i in dupb]
    return resa, resb
            

