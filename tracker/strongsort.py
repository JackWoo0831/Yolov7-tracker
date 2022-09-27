"""
StrongSORT: Make DeepSORT Great Again(arxiv)

Some Codes are partly copied from official repo
"""

import numpy as np  
from basetrack import TrackState, STrack, BaseTracker
from reid_models.OSNet import osnet_x0_25  # OSNet model
from reid_models.load_model_tools import load_pretrained_weights
import matching
import torch 
from torchvision.ops import nms
import torchvision.transforms as T
import cv2

from botsort import GMC, multi_gmc


class StrongSORT(BaseTracker):
    def __init__(self, opts, frame_rate=30, gamma=0.1, use_ECC=True, use_AFLink=True, use_GSI=True, num_of_budget=20, 
                    *args, **kwargs) -> None:
        super().__init__(opts, frame_rate, *args, **kwargs)

        self.gamma = gamma
        self.reid_model = osnet_x0_25(num_classes=1, pretrained=False, )  # Re-ID: OSNet
        # load model
        self.reid_model.cuda().eval()
        load_pretrained_weights(self.reid_model, opts.reid_model_path)

        self.use_ECC = use_ECC  # whether use ECC(Camera compensation)
        self.ECC = GMC(method='ecc', downscale=2, verbose=None)

        # TODO: AFLink
        self.use_AFLink = use_AFLink  # whether use AFLink (module that compute the
        # similarity of two tracks)  defination of this net can be found in reid_model/PostLinker

        # TODO: GSI
        self.use_GSI = use_GSI  # whether use Gaussian smoothed interpolation of tracks

        self.matching_thresh = min(0.3, self.opts.iou_thresh - 0.2)  # matching thresh
        self.num_of_budget = num_of_budget

        # self.metric = matching.NearestNeighborDistanceMetric(metric='euclidean', 
        #     matching_threshold=self.matching_thresh, budget=self.num_of_budget)

    def reid_preprocess(self, obj_bbox):
        """
        preprocess cropped object bboxes 
        
        obj_bbox: np.ndarray, shape=(h_obj, w_obj, c)

        return: 
        torch.Tensor of shape (c, 128, 256)
        """
        obj_bbox = cv2.resize(obj_bbox.astype(np.float32) / 255.0, dsize=(256, 128))  # shape: (128, 256, c)

        transforms = T.Compose([
            # T.ToPILImage(),
            # T.Resize(size=(256, 128)),
            T.ToTensor(),  # (c, 128, 256)
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transforms(obj_bbox)

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
            
            tlbr_tensor = self.reid_preprocess(ori_img[tlbr[1]: tlbr[3], tlbr[0]: tlbr[2]])
            obj_bbox.append(tlbr_tensor)
        
        if not obj_bbox:
            return np.array([])
        
        obj_bbox = torch.stack(obj_bbox, dim=0)
        obj_bbox = obj_bbox.cuda()  
        
        features = self.reid_model(obj_bbox)  # shape: (num_of_objects, feature_dim)
        return features.cpu().detach().numpy()

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

            bbox_temp = STrack.xywh2tlbr(det_results[:, :4])            
            if self.NMS:
                # NOTE: Note nms need tlbr format
                nms_indices = nms(torch.from_numpy(bbox_temp), torch.from_numpy(det_results[:, 4]), 
                                self.opts.nms_thresh)
                det_results = det_results[nms_indices.numpy()]
                features = self.get_feature(bbox_temp[nms_indices.numpy()], ori_img)
            else:
                features = self.get_feature(bbox_temp, ori_img)

            # detections: List[Strack]
            detections = [STrack(cls, STrack.xywh2tlwh(xywh), score, kalman_format=self.opts.kalman_format, feature=feature)
                            for (cls, xywh, score, feature) in zip(det_results[:, -1], det_results[:, :4], det_results[:, 4], features)]

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

        if self.use_ECC:
            # update first, of ECC module (Camera Copensation)
            # NOTE: this part is similiar to BoTSORT, so we use that in BoTSORT
            wrap = self.ECC.apply(raw_frame=ori_img, detections=detections)
            multi_gmc(strack_pool, wrap)

        # Kalman predict, update every mean and cov of tracks
        STrack.multi_predict(stracks=strack_pool, kalman=self.kalman)

        # calculate apperance distance, shape:(len(strack_pool), len(detections))
        Apperance_dist = matching.embedding_distance(tracks=strack_pool, detections=detections, metric='euclidean')
        # calculate iou distance, shape:(len(strack_pool), len(detections))
        IoU_dist = matching.iou_distance(atracks=strack_pool, btracks=detections)
        # fuse
        Dist_mat = self.gamma * IoU_dist + (1. - self.gamma) * Apperance_dist

        # match  thresh=0.9 is same in ByteTrack code
        matched_pair0, u_tracks0_idx, u_dets0_idx = matching.linear_assignment(Dist_mat, thresh=0.7)

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
                exit(0)
                track.re_activate(det, self.frame_id, )
                refind_stracks.append(track)

        """ Step 4. deal with rest tracks and dets"""
        # deal with final unmatched tracks
        for idx in u_tracks1_idx:
            track = strack_pool[idx]
            track.mark_lost()
            lost_stracks.append(track)

        # deal with unconfirmed tracks, match new track of last frame and new high conf det
        Apperance_dist = matching.embedding_distance(tracks=unconfirmed, detections=u_det1, metric='euclidean')
        IoU_dist = matching.iou_distance(atracks=unconfirmed, btracks=u_det1)
        Dist_mat = self.gamma * IoU_dist + (1. - self.gamma) * Apperance_dist
        matched_pair2, u_tracks2_idx, u_det2_idx = matching.linear_assignment(Dist_mat, thresh=0.7)

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

