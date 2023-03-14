"""
Unofficial Implementation of paper
Hard to Track Objects with Irregular Motions and Similar Appearances? Make It Easier by Buffering the Matching Space(arxiv 2212)
"""

import numpy as np  
from collections import deque
from basetrack import TrackState, BaseTrack, BaseTracker
import matching
import torch 
from torchvision.ops import nms


"""
Because the paper drops Kalman so we rewrite the STrack class and rename it as C_BIoUSTrack
"""
class C_BIoUSTrack(BaseTrack):
    def __init__(self, cls, tlwh, score) -> None:
        """
        cls: category of this obj 
        tlwh: positoin   score: conf score
        """
        super().__init__()
        self.cls = cls
        self._tlwh = np.asarray(tlwh, dtype=np.float32)  # init tlwh
        self.score = score

        self.is_activated = False
        self.tracklet_len = 0  

        self.track_id = None
        self.start_frame = None
        self.frame_id = None
        self.time_since_update = 0  # \delta in paper, use to calculate motion state
        
        # params in motion state
        self.b1, self.b2, self.n = 0.3, 0.5, 5
        self.origin_bbox_buffer = deque()  # a deque store the original bbox(tlwh) from t - self.n to t, where t is the last time detected
        self.origin_bbox_buffer.append(self._tlwh)
        # buffered bbox, two buffer sizes
        self.buffer_bbox1 = self.get_buffer_bbox(level=1)
        self.buffer_bbox2 = self.get_buffer_bbox(level=2)
        # motion state, s^{t + \delta} = o^t + (\delta / n) * \sum_{i=t-n+1}^t(o^i - o^{i-1}) = o^t + (\delta / n) * (o^t - o^{t - n})
        self.motion_state1 = self.buffer_bbox1.copy()
        self.motion_state2 = self.buffer_bbox2.copy()
        

    def get_buffer_bbox(self, level=1, bbox=None):
        """
        get buffered bbox as: (top, left, w, h) -> (top - bw, y - bh, w + 2bw, h + 2bh)
        level = 1: b = self.b1  level = 2: b = self.b2
        bbox: if not None, use bbox to calculate buffer_bbox, else use self._tlwh
        """
        assert level in [1, 2], 'level must be 1 or 2'

        b = self.b1 if level == 1 else self.b2

        if bbox is None:
            buffer_bbox = self._tlwh + np.array([-b*self._tlwh[2], -b*self._tlwh[3], 2*b*self._tlwh[2], 2*b*self._tlwh[3]])
        else:
            buffer_bbox = bbox + np.array([-b*bbox[2], -b*bbox[3], 2*b*bbox[2], 2*b*bbox[3]])
        return np.maximum(0.0, buffer_bbox)


    @property
    def tlbr(self):
        ret = self.origin_bbox_buffer[-1].copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def tlwh(self):
        ret = self.origin_bbox_buffer[-1].copy()
        
        return ret

    def activate(self, frame_id):
        """
        init a new track
        """
        self.track_id = BaseTrack.next_id()
        self.state = TrackState.Tracked

        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
    
    def re_activate(self, new_track, frame_id, new_id=False):
        """
        reactivate a lost track
        """
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self._tlwh = new_track._tlwh
        # update stored bbox
        if (len(self.origin_bbox_buffer) > self.n):
            self.origin_bbox_buffer.popleft()
            self.origin_bbox_buffer.append(self._tlwh)
        else:
            self.origin_bbox_buffer.append(self._tlwh)

        self.buffer_bbox1 = self.get_buffer_bbox(level=1)
        self.buffer_bbox2 = self.get_buffer_bbox(level=2)
        self.motion_state1 = self.buffer_bbox1.copy()
        self.motion_state2 = self.buffer_bbox2.copy()

    def update(self, new_track, frame_id):
        """
        update a track
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        # update position and score
        new_tlwh = new_track.tlwh
        self._tlwh = new_tlwh
        self.score = new_track.score
        
        # update stored bbox
        if (len(self.origin_bbox_buffer) > self.n):
            self.origin_bbox_buffer.popleft()
            self.origin_bbox_buffer.append(new_tlwh)
        else:
            self.origin_bbox_buffer.append(new_tlwh)

        # update motion state
        if self.time_since_update:  # have some unmatched frames
            if len(self.origin_bbox_buffer) < self.n:
                self.motion_state1 = self.get_buffer_bbox(level=1, bbox=new_tlwh)
                self.motion_state2 = self.get_buffer_bbox(level=2, bbox=new_tlwh)
            else:  # s^{t + \delta} = o^t + (\delta / n) * (o^t - o^{t - n})
                motion_state = self.origin_bbox_buffer[-1] + \
                    (self.time_since_update / self.n) * (self.origin_bbox_buffer[-1] - self.origin_bbox_buffer[0])
                self.motion_state1 = self.get_buffer_bbox(level=1, bbox=motion_state)
                self.motion_state2 = self.get_buffer_bbox(level=2, bbox=motion_state)

        else:  # no unmatched frames, use current detection as motion state
            self.motion_state1 = self.get_buffer_bbox(level=1, bbox=new_tlwh)
            self.motion_state2 = self.get_buffer_bbox(level=2, bbox=new_tlwh)

        # update status
        self.state = TrackState.Tracked
        self.is_activated = True

        self.time_since_update = 0

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

    @staticmethod
    def tlwh2tlbr(tlwh):
        """
        convert top, left, wh to tlbr
        """
        if len(tlwh.shape) > 1:
            result = np.asarray(tlwh).copy()
            result[:, 2:] += result[:, :2]
        else:
            result = np.asarray(tlwh).copy()
            result[2:] += result[:2]

        return result


class C_BIoUTracker(BaseTracker):
    def __init__(self, opts, frame_rate=30, *args, **kwargs) -> None:
        super().__init__(opts, frame_rate, *args, **kwargs)

        self.kalman = None  # The paper drops Kalman Filter so we donot use it

    def update(self, det_results, ori_img):
        """
        this func is called by every time step

        det_results: numpy.ndarray or torch.Tensor, shape(N, 6), 6 includes bbox, conf_score, cls
        ori_img: original image, np.ndarray, shape(H, W, C)
        """

        if isinstance(det_results, torch.Tensor):
            det_results = det_results.cpu().numpy()
        if isinstance(ori_img, torch.Tensor):
            ori_img = ori_img.numpy()

        self.frame_id += 1
        activated_starcks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        """step 1. filter results and init tracks"""
        det_results = det_results[det_results[:, 4] > self.det_thresh]

        # convert the scale to origin size
        # NOTE: yolo v7 origin out format: [xc, yc, w, h, conf, cls0_conf, cls1_conf, ..., clsn_conf]
        # TODO: check here, if nesscessary use two ratio
        img_h, img_w = ori_img.shape[0], ori_img.shape[1]
        ratio = [img_h / self.model_img_size[0], img_w / self.model_img_size[1]]  # usually > 1
        det_results[:, 0], det_results[:, 2] =  det_results[:, 0]*ratio[1], det_results[:, 2]*ratio[1]
        det_results[:, 1], det_results[:, 3] =  det_results[:, 1]*ratio[0], det_results[:, 3]*ratio[0]

        if det_results.shape[0] > 0:
            if self.NMS:
                # TODO: Note nms need tlbr format
                bbox_temp = C_BIoUSTrack.xywh2tlbr(det_results[:, :4])
                nms_indices = nms(torch.from_numpy(bbox_temp), torch.from_numpy(det_results[:, 4]), 
                                self.opts.nms_thresh)
                det_results = det_results[nms_indices.numpy()]

            # detections: List[Strack]
            detections = [C_BIoUSTrack(cls, C_BIoUSTrack.xywh2tlwh(xywh), score)
                            for (cls, xywh, score) in zip(det_results[:, -1], det_results[:, :4], det_results[:, 4])]

        else:
            detections = []

         # Do some updates
        unconfirmed = []
        tracked_stracks = []  # type: list[C_BIoUSTrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """step 2. association with IoU in level 1"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # cal IoU dist
        IoU_mat = matching.buffered_iou_distance(strack_pool, detections, level=1)
        
        matched_pair0, u_tracks0_idx, u_dets0_idx = matching.linear_assignment(IoU_mat, thresh=0.9)

        for itracked, idet in matched_pair0:  # for those who matched successfully
            track = strack_pool[itracked]
            det = detections[idet]

            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # tracks and detections that not matched
        u_tracks0 = [strack_pool[i] for i in u_tracks0_idx if strack_pool[i].state == TrackState.Tracked]
        u_dets0 = [detections[i] for i in u_dets0_idx]

        """step 3. association with IoU in level 2"""
        IoU_mat = matching.buffered_iou_distance(u_tracks0, u_dets0, level=2)

        matched_pair1, u_tracks1_idx, u_dets1_idx = matching.linear_assignment(IoU_mat, thresh=0.5)

        for itracked, idet in matched_pair1:
            track = u_tracks0[itracked]
            det = u_dets0[idet]

            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        u_tracks1 = [u_tracks0[i] for i in u_tracks1_idx]
        u_dets1 = [u_dets0[i] for i in u_dets1_idx]

        """step 3'. match unconfirmed tracks"""
        IoU_mat = matching.buffered_iou_distance(unconfirmed, u_dets1, level=1)
        matched_pair_unconfirmed, u_tracks_unconfirmed_idx, u_dets_unconfirmed_idx = \
            matching.linear_assignment(IoU_mat, thresh=0.7)

        for itracked, idet in matched_pair_unconfirmed:
            track = unconfirmed[itracked]
            det = u_dets1[idet]
            track.update(det, self.frame_id)
            activated_starcks.append(track)

        for idx in u_tracks_unconfirmed_idx:
            track = unconfirmed[idx]
            track.mark_removed()
            removed_stracks.append(track)

        # new tracks
        for idx in u_dets_unconfirmed_idx:
            det = u_dets1[idx]
            if det.score > self.det_thresh + 0.1:
                det.activate(self.frame_id)
                activated_starcks.append(det)

        """ Step 4. deal with rest tracks"""
        for u_track in u_tracks1:
            if self.frame_id - u_track.end_frame > self.max_time_lost:
                u_track.mark_removed()
                removed_stracks.append(u_track)
            else:
                u_track.mark_lost()
                u_track.time_since_update = self.frame_id - u_track.end_frame  # u_track.time_since_update += 1
                lost_stracks.append(u_track)

        
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