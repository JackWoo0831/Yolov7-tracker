
import numpy as np  
from basetrack import TrackState, STrack, BaseTracker
from kalman_filter import KalmanFilter, NaiveKalmanFilter, chi2inv95
from reid_models.deepsort_reid import Extractor
import matching
import torch 
from torchvision.ops import nms


class DeepSORT(BaseTracker):
    def __init__(self, opts, frame_rate=30, gamma=0.02, *args, **kwargs) -> None:
        super().__init__(opts, frame_rate)

        self.reid_model = Extractor(opts.reid_model_path, use_cuda=True)
        self.gamma = gamma  # coef that balance the apperance and iou
        self.filter_small_area = False  # filter area < 50 bboxs
        
    def get_feature(self, tlbrs, ori_img):
        """
        get apperance feature of an object
        tlbrs: shape (num_of_objects, 4)
        ori_img: original image, np.ndarray, shape(H, W, C)
        """
        obj_bbox = []

        for tlbr in tlbrs:
            tlbr = list(map(int, tlbr))
            # if any(tlbr_ == -1 for tlbr_ in tlbr):
            #     print(tlbr)
            obj_bbox.append(
                ori_img[tlbr[1]: tlbr[3], tlbr[0]: tlbr[2]]
            )
        
        if obj_bbox:  # obj_bbox is not []
            features = self.reid_model(obj_bbox)  # shape: (num_of_objects, feature_dim)

        else:
            features = np.array([])
        return features
    
    def gate_cost_matrix(self, cost_matrix, tracks, dets, max_apperance_thresh=0.15, gated_cost=1e5, only_position=False):
        """
        gate cost matrix by calculating the Kalman state distance and constrainted by
        0.95 confidence interval of x2 distribution

        cost_matrix: np.ndarray, shape (len(tracks), len(dets))
        tracks: List[STrack]
        dets: List[STrack]
        gated_cost: a very largt const to infeasible associations
        only_position: use [xc, yc, a, h] as state vector or only use [xc, yc]

        return:
        updated cost_matirx, np.ndarray
        """
        gating_dim = 2 if only_position else 4
        gating_threshold = chi2inv95[gating_dim]
        measurements = np.asarray([STrack.tlwh2xyah(det.tlwh) for det in dets])  # (len(dets), 4)

        cost_matrix[cost_matrix > max_apperance_thresh] = gated_cost
        for row, track in enumerate(tracks):
            gating_distance = self.kalman.gating_distance(
                track.mean, track.cov, measurements, only_position
            )
            cost_matrix[row, gating_distance > gating_threshold] = gated_cost
        return cost_matrix

    def gated_metric(self, tracks, dets):
        """
        get cost matrix, firstly calculate apperence cost, then filter by Kalman state.

        tracks: List[STrack]
        dets: List[STrack]
        """
        Apperance_dist = matching.nearest_embedding_distance(tracks=tracks, detections=dets, metric='cosine')
        cost_matrix = self.gate_cost_matrix(Apperance_dist, tracks, dets, )
        return cost_matrix
    
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


        if det_results.shape[0] > 0:

            bbox_temp = det_results[:, :4]
            if self.filter_small_area:  # filter small area bboxs
                small_indicies = det_results[:, 2]*det_results[:, 3] > 50
                det_results = det_results[small_indicies]
                bbox_temp = bbox_temp[small_indicies]
            
            features = self.get_feature(bbox_temp, ori_img)

            # detections: List[Strack]
            detections = [STrack(cls, STrack.tlbr2tlwh(tlbr), score, kalman_format=self.opts.kalman_format, feature=feature, use_avg_of_feature=False)
                            for (cls, tlbr, score, feature) in zip(det_results[:, -1], det_results[:, :4], det_results[:, 4], features)]

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

        """Step 2. association with motion and apperance"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Kalman predict, update every mean and cov of tracks
        STrack.multi_predict(stracks=strack_pool, kalman=self.kalman)

        # match  thresh=0.9 is same in ByteTrack code
        matched_pair0, u_tracks0_idx, u_dets0_idx = matching.matching_cascade(self.gated_metric, 0.9, self.max_time_lost, 
                                                            strack_pool, detections)

        for itrack_match, idet_match in matched_pair0:
            track = strack_pool[itrack_match]
            det = detections[idet_match]

            if track.state == TrackState.Tracked:  # normal track
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, )
                refind_stracks.append(track)

        """ Step 3. association with motion"""
        
        u_tracks0 = [strack_pool[i] for i in u_tracks0_idx if strack_pool[i].state == TrackState.Tracked]
        u_dets0 = [detections[i] for i in u_dets0_idx]

        # calculate IoU
        IoU_dist = matching.iou_distance(atracks=u_tracks0, btracks=u_dets0)
        # match
        matched_pair1, u_tracks1_idx, u_dets1_idx = matching.linear_assignment(IoU_dist, thresh=0.5)
        u_det1 = [u_dets0[i] for i in u_dets1_idx]

        for itrack_match, idet_match in matched_pair1:
            track = u_tracks0[itrack_match]
            det = u_dets0[idet_match]

            if track.state == TrackState.Tracked:  # normal track
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, )
                refind_stracks.append(track)

        """ Step 4. deal with rest tracks and dets"""
        # deal with final unmatched tracks
        for idx in u_tracks1_idx:
            track = strack_pool[idx]
            track.mark_lost()
            lost_stracks.append(track)

        # deal with unconfirmed tracks, match new track of last frame and new high conf det
        IoU_dist = matching.iou_distance(atracks=unconfirmed, btracks=u_det1)
        matched_pair2, u_tracks2_idx, u_det2_idx = matching.linear_assignment(IoU_dist, thresh=0.9)

        for itrack_match, idet_match in matched_pair2:
            track = unconfirmed[itrack_match]
            det = u_det1[idet_match]
            track.update(det, self.frame_id)
            activated_starcks.append(track)

        for u_itrack2_idx in u_tracks2_idx:
            track = unconfirmed[u_itrack2_idx]
            track.mark_removed()
            removed_stracks.append(track)

        # deal with new tracks
        for idx in u_det2_idx:
            det = u_det1[idx]
            if det.score > self.det_thresh:
                det.activate(self.frame_id)
                activated_starcks.append(det)


        """ Step 5. remove long lost tracks"""
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
            
