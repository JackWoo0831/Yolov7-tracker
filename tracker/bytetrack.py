import numpy as np  
from basetrack import TrackState, STrack, BaseTracker
from reid_models.deepsort_reid import Extractor
import matching
import torch 
from torchvision.ops import nms

class ByteTrack(BaseTracker):
    def __init__(self, opts, frame_rate=30, gamma=0.1, *args, **kwargs) -> None:
        super().__init__(opts, frame_rate, *args, **kwargs)
        self.use_apperance_model = False
        self.reid_model = Extractor(opts.reid_model_path, use_cuda=True)
        self.gamma = gamma  # coef that balance the apperance and ious

        self.low_conf_thresh = max(0.15, self.opts.conf_thresh - 0.3)  # low threshold for second matching

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

            obj_bbox.append(
                ori_img[tlbr[1]: tlbr[3], tlbr[0]: tlbr[2]]
            )
        
        if obj_bbox:  # obj_bbox is not []
            features = self.reid_model(obj_bbox)  # shape: (num_of_objects, feature_dim)

        else:
            features = np.array([])
        return features    

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
               
        # filter small area bboxs
        if self.filter_small_area:  
            small_indicies = det_results[:, 2]*det_results[:, 3] > 50
            det_results = det_results[small_indicies]


        # cal high and low indicies
        det_high_indicies = det_results[:, 4] >= self.det_thresh
        det_low_indicies = np.logical_and(np.logical_not(det_high_indicies), det_results[:, 4] > self.low_conf_thresh)

        
        # init saperatly
        det_high, det_low = det_results[det_high_indicies], det_results[det_low_indicies]
        if det_high.shape[0] > 0:
            if self.use_apperance_model:
                features = self.get_feature(det_high[:, :4], ori_img)
                # detections: List[Strack]
                D_high = [STrack(cls, STrack.tlbr2tlwh(tlbr), score, kalman_format=self.opts.kalman_format, feature=feature)
                                for (cls, tlbr, score, feature) in zip(det_high[:, -1], det_high[:, :4], det_high[:, 4], features)]
            else:
                D_high = [STrack(cls, STrack.tlbr2tlwh(tlbr), score, kalman_format=self.opts.kalman_format)
                            for (cls, tlbr, score) in zip(det_high[:, -1], det_high[:, :4], det_high[:, 4])]
        else:
            D_high = []

        if det_low.shape[0] > 0:
            D_low = [STrack(cls, STrack.tlbr2tlwh(tlbr), score, kalman_format=self.opts.kalman_format)
                            for (cls, tlbr, score) in zip(det_high[:, -1], det_high[:, :4], det_high[:, 4])]
        else:
            D_low = []

        # Do some updates
        unconfirmed = []  # unconfirmed means when frame id > 2, new track of last frame
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
       
        # update track state
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Kalman predict, update every mean and cov of tracks
        STrack.multi_predict(stracks=strack_pool, kalman=self.kalman)

        """Step 2. first match, match high conf det with tracks"""
        if self.use_apperance_model:
            # use apperance model, DeepSORT way
            Apperance_dist = matching.embedding_distance(strack_pool, D_high, metric='cosine')
            IoU_dist = matching.iou_distance(atracks=strack_pool, btracks=D_high)
            Dist_mat = self.gamma * IoU_dist + (1. - self.gamma) * Apperance_dist
        else:
            Dist_mat = matching.iou_distance(atracks=strack_pool, btracks=D_high)
        
        # match
        matched_pair0, u_tracks0_idx, u_dets0_idx = matching.linear_assignment(Dist_mat, thresh=0.9)
        for itrack_match, idet_match in matched_pair0:
            track = strack_pool[itrack_match]
            det = D_high[idet_match]

            if track.state == TrackState.Tracked:  # normal track
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, )
                refind_stracks.append(track)

        u_tracks0 = [strack_pool[i] for i in u_tracks0_idx if strack_pool[i].state == TrackState.Tracked]
        u_dets0 = [D_high[i] for i in u_dets0_idx]

        """Step 3. second match, match remain tracks and low conf dets"""
        # only IoU
        Dist_mat = matching.iou_distance(atracks=u_tracks0, btracks=D_low)
        matched_pair1, u_tracks1_idx, u_dets1_idx = matching.linear_assignment(Dist_mat, thresh=0.5)

        for itrack_match, idet_match in matched_pair1:
            track = u_tracks0[itrack_match]
            det = D_low[idet_match]

            if track.state == TrackState.Tracked:  # normal track
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, )
                refind_stracks.append(track)
        
        """ Step 4. deal with rest tracks and dets"""
        # deal with final unmatched tracks
        for idx in u_tracks1_idx:
            track = u_tracks0[idx]
            track.mark_lost()
            lost_stracks.append(track)
        
        # deal with unconfirmed tracks, match new track of last frame and new high conf det
        Dist_mat = matching.iou_distance(unconfirmed, u_dets0)
        matched_pair2, u_tracks2_idx, u_dets2_idx = matching.linear_assignment(Dist_mat, thresh=0.7)
        for itrack_match, idet_match in matched_pair2:
            track = unconfirmed[itrack_match]
            det = u_dets0[idet_match]
            track.update(det, self.frame_id)
            activated_starcks.append(track)

        for idx in u_tracks2_idx:
            track = unconfirmed[idx]
            track.mark_removed()
            removed_stracks.append(track)

        # deal with new tracks
        for idx in u_dets2_idx:
            det = u_dets0[idx]
            if det.score > self.det_thresh + 0.1:
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