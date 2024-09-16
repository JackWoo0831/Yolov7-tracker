"""
Strong Sort
"""

import numpy as np  
import torch 
from torchvision.ops import nms

import cv2 
import torchvision.transforms as T

from .basetrack import BaseTrack, TrackState
from .tracklet import Tracklet, Tracklet_w_reid
from .matching import *

from .reid_models.OSNet import *
from .reid_models.load_model_tools import load_pretrained_weights
from .reid_models.deepsort_reid import Extractor

REID_MODEL_DICT = {
    'osnet_x1_0': osnet_x1_0, 
    'osnet_x0_75': osnet_x0_75, 
    'osnet_x0_5': osnet_x0_5, 
    'osnet_x0_25': osnet_x0_25, 
    'deepsort': Extractor
}


def load_reid_model(reid_model, reid_model_path):
    
    if 'osnet' in reid_model:
        func = REID_MODEL_DICT[reid_model]
        model = func(num_classes=1, pretrained=False, )
        load_pretrained_weights(model, reid_model_path)
        model.cuda().eval()
        
    elif 'deepsort' in reid_model:
        model = REID_MODEL_DICT[reid_model](reid_model_path, use_cuda=True)

    else:
        raise NotImplementedError
    
    return model


class StrongSortTracker(object):

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

        self.with_reid = not args.discard_reid

        self.reid_model, self.crop_transforms = None, None 
        if self.with_reid:
            self.reid_model = load_reid_model(args.reid_model, args.reid_model_path)
            self.crop_transforms = T.Compose([
            # T.ToPILImage(),
            # T.Resize(size=(256, 128)),
            T.ToTensor(),  # (c, 128, 256)
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
            
        self.bbox_crop_size = (64, 128) if 'deepsort' in args.reid_model else (128, 128)

        self.lambda_ = 0.98  # the coef of cost mix in eq. 10 in paper
        

    def reid_preprocess(self, obj_bbox):
        """
        preprocess cropped object bboxes 
        
        obj_bbox: np.ndarray, shape=(h_obj, w_obj, c)

        return: 
        torch.Tensor of shape (c, 128, 256)
        """

        obj_bbox = cv2.resize(obj_bbox.astype(np.float32) / 255.0, dsize=self.bbox_crop_size)  # shape: (h, w, c)

        return self.crop_transforms(obj_bbox)

    def get_feature(self, tlwhs, ori_img):
        """
        get apperance feature of an object
        tlwhs: shape (num_of_objects, 4)
        ori_img: original image, np.ndarray, shape(H, W, C)
        """
        obj_bbox = []

        for tlwh in tlwhs:
            tlwh = list(map(int, tlwh))

            # limit to the legal range
            tlwh[0], tlwh[1] = max(tlwh[0], 0), max(tlwh[1], 0)
            
            tlbr_tensor = self.reid_preprocess(ori_img[tlwh[1]: tlwh[1] + tlwh[3], tlwh[0]: tlwh[0] + tlwh[2]])

            obj_bbox.append(tlbr_tensor)
        
        if not obj_bbox:
            return np.array([])
        
        obj_bbox = torch.stack(obj_bbox, dim=0)
        obj_bbox = obj_bbox.cuda()  
        
        features = self.reid_model(obj_bbox)  # shape: (num_of_objects, feature_dim)
        return features.cpu().detach().numpy()
    
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

        features_keep = self.get_feature(tlwhs=dets[:, :4], ori_img=ori_img)

        if len(dets) > 0:
            '''Detections'''
            detections = [Tracklet_w_reid(tlwh, s, cate, motion=self.motion, feat=feat) for
                          (tlwh, s, cate, feat) in zip(dets, scores_keep, cates, features_keep)]
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

        ''' Step 2: First association, with appearance'''
        tracklet_pool = joint_tracklets(tracked_tracklets, self.lost_tracklets)

        # Predict the current location with Kalman
        for tracklet in tracklet_pool:
            tracklet.predict()

        # vallina matching
        cost_matrix = self.gated_metric(tracklet_pool, detections)
        matches, u_track, u_detection = linear_assignment(cost_matrix, thresh=0.9)

        for itracked, idet in matches:
            track = tracklet_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        '''Step 3: Second association, with iou'''
        tracklet_for_iou = [tracklet_pool[i] for i in u_track if tracklet_pool[i].state == TrackState.Tracked]
        detection_for_iou = [detections[i] for i in u_detection]

        dists = iou_distance(tracklet_for_iou, detection_for_iou)

        matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = tracklet_for_iou[itracked]
            det = detection_for_iou[idet]
            if track.state == TrackState.Tracked:
                track.update(detection_for_iou[idet], self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        for it in u_track:
            track = tracklet_for_iou[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracklets.append(track)



        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detection_for_iou[i] for i in u_detection]
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
    
    def gated_metric(self, tracks, dets):
        """
        get cost matrix, firstly calculate apperence cost, then filter by Kalman state.

        tracks: List[STrack]
        dets: List[STrack]
        """
        apperance_dist = embedding_distance(tracks=tracks, detections=dets, metric='cosine')
        cost_matrix = self.gate_cost_matrix(apperance_dist, tracks, dets, )
        return cost_matrix
    
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
        measurements = np.asarray([Tracklet.tlwh_to_xyah(det.tlwh) for det in dets])  # (len(dets), 4)

        cost_matrix[cost_matrix > max_apperance_thresh] = gated_cost
        for row, track in enumerate(tracks):
            gating_distance = track.kalman_filter.gating_distance(measurements, )
            cost_matrix[row, gating_distance > gating_threshold] = gated_cost

            cost_matrix[row] = self.lambda_ * cost_matrix[row] + (1 - self.lambda_) *  gating_distance
        return cost_matrix
    

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