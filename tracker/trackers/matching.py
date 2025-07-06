"""
Measurement functions, Assignment and matching functions, 
Distance fusion functions
"""

import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
import math
# from cython_bbox import bbox_overlaps as bbox_ious
import torch 
from torchvision.ops import box_iou
import time

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


"""
Part I. Measurement functions
"""

def ious(atlbrs, btlbrs):
    """
    Compute IoU

    Args:
        atlbrs: List[np.ndarray], length of m
        btlbrs: List[np.ndarray], length of n
    
    Returns:
        np.ndarray, shape (m, n)
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float32),
        np.ascontiguousarray(btlbrs, dtype=np.float32)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU

    Args:
        atracks: List[Tracklet], length of m
        btracks: List[Tracklet], length of n

    Returns:
        cost: 1.0 - IoU, np.ndarray, shape (m, n)
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    Calculate the feature embedding distance

    Args: 
        tracks: List[Tracklet], length of m
        detections: List[Tracklet], length of m
        metric: str, cosine or eculid
    
    Returns:
        cost: 1.0 - cosine, ..., np.ndarray, shape (m, n)
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def buffered_iou_distance(atracks, btracks, level=1):
    """
    Calculate buffered IoU, used in C_BIoU Tracker

    Args: 
        atracks: List[Tracklet], length of m
        btracks: List[Tracklet], length of m
        level: int, cascade level, 1 or 2
    
    Returns:
        cost: 1 - IoU, np.ndarray, shape (m, n)
    """
    assert level in [1, 2], 'level must be 1 or 2'
    if level == 1:  # use motion_state1(tracks) and buffer_bbox1(detections) to calculate
        atlbrs = [track.tlwh_to_tlbr(track.motion_state1) for track in atracks]
        btlbrs = [det.tlwh_to_tlbr(det.buffer_bbox1) for det in btracks]
    else:
        atlbrs = [track.tlwh_to_tlbr(track.motion_state2) for track in atracks]
        btlbrs = [det.tlwh_to_tlbr(det.buffer_bbox2) for det in btracks]
    _ious = ious(atlbrs, btlbrs)

    cost_matrix = 1 - _ious
    return cost_matrix


def hm_iou_distance(atracks, btracks, return_original_iou=False):
    """
    calculate HM IoU, used in Hybrid Sort

    Args: 
        atracks: List[Tracklet], length of m
        btracks: List[Tracklet], length of m
        return_original_iou: bool, if True, also return the original iou
    
    Returns:
        distance: original hm_iou (and iou), np.ndarray, shape (m, n)
    """
    # hm iou = iou * height iou
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = ious(atlbrs, btlbrs)  # original iou

    if _ious.size == 0: 
        # case if len of tracks == 0, no need to further calculating
        return (_ious, _ious) if return_original_iou else _ious

    if isinstance(atlbrs, list): atlbrs = np.ascontiguousarray(atlbrs)
    if isinstance(btlbrs, list): btlbrs = np.ascontiguousarray(btlbrs)

    # height iou = (y2_min - y1_max) / (y2_max - y1_min)
    atlbrs_ = np.expand_dims(atlbrs, axis=1)  # (M, 4) -> (M, 1, 4) to apply boardcast mechanism
    btlbrs_ = np.expand_dims(btlbrs, axis=0)  # (N, 4) -> (1, N, 4)

    y2_min = np.minimum(atlbrs_[..., 3], btlbrs_[..., 3])  # (M, N)
    y1_max = np.maximum(atlbrs_[..., 1], btlbrs_[..., 1])

    y2_max = np.maximum(atlbrs_[..., 3], btlbrs_[..., 3])
    y1_min = np.minimum(atlbrs_[..., 1], btlbrs_[..., 1])

    _h_ious = (y2_min - y1_max) / (y2_max - y1_min)

    return (_ious * _h_ious, _ious) if return_original_iou else _ious * _h_ious
    
def d_iou_distance(atracks, btracks, return_original_iou=False):
    """
    Compute cost based on D-IoU, used in ImproAssoc

    Args:
        atracks: List[Tracklet], length of m
        btracks: List[Tracklet], length of n
        return_original_iou: bool, if True, also return the original iou

    Returns:
        distance: original d_iou (and iou), np.ndarray, shape (m, n)
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = ious(atlbrs, btlbrs)  # original iou

    if _ious.size == 0: 
        # case if len of tracks == 0, no need to further calculating
        return (_ious, _ious) if return_original_iou else _ious

    if isinstance(atlbrs, list): atlbrs = np.ascontiguousarray(atlbrs)
    if isinstance(btlbrs, list): btlbrs = np.ascontiguousarray(btlbrs)

    atlbrs_ = np.expand_dims(atlbrs, axis=1)  # (M, 4) -> (M, 1, 4) to apply boardcast mechanism
    btlbrs_ = np.expand_dims(btlbrs, axis=0)  # (N, 4) -> (1, N, 4)

    centerx1 = (atlbrs_[..., 0] + atlbrs_[..., 2]) / 2.0
    centery1 = (atlbrs_[..., 1] + atlbrs_[..., 3]) / 2.0
    centerx2 = (btlbrs_[..., 0] + btlbrs_[..., 2]) / 2.0
    centery2 = (btlbrs_[..., 1] + btlbrs_[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(atlbrs_[..., 0], btlbrs_[..., 0])
    yyc1 = np.minimum(atlbrs_[..., 1], btlbrs_[..., 1])
    xxc2 = np.maximum(atlbrs_[..., 2], btlbrs_[..., 2])
    yyc2 = np.maximum(atlbrs_[..., 3], btlbrs_[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    _d_ious = _ious - inner_diag / outer_diag
    _d_ious = (_d_ious + 1) * 0.5  # resize from (-1,1) to (0,1)

    return (_d_ious, _ious) if return_original_iou else _d_ious


def score_distance(atracks, btracks):
    """
    calculate the confidence score difference, c_{i, j} = abs(atracks[i].score - btracks[j].score)

    Args:
        atlbrs: List[np.ndarray], length of m
        btlbrs: List[np.ndarray], length of n
    
    Returns:
        distance: np.ndarray, shape (m, n)
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        ascores = atracks
        bscores = btracks
    else:
        ascores = [track.score for track in atracks]
        bscores = [track.score for track in btracks]

    return score_diff_batch(det_scores=np.ascontiguousarray(bscores), 
                            track_scores=np.ascontiguousarray(ascores))


def nearest_embedding_distance(tracks, detections, metric='cosine'):
    """
    different from embedding distance, this func calculate the 
    nearest distance among all track history features and detections

    used in Deep SORT

    Args:
        tracks: List[Tracklet], length of m
        detections: List[Tracklet], length of n
        metric: str, cosine
    
    Returns:
        cost: 1.0 - cosine, np.ndarray, shape (m, n)
    """
    cost_matrix = np.zeros((len(tracks), len(detections)))
    det_features = np.asarray([det.features[-1] for det in detections])

    for row, track in enumerate(tracks):
        track_history_features = np.asarray(track.features)
        dist = 1. - cal_cosine_distance(track_history_features, det_features)
        dist = dist.min(axis=0)
        cost_matrix[row, :] = dist
    
    return cost_matrix

def angle_distance(det_tlbrs, previous_obs, velocities, mode='center'):
    """
    calculate the angle between tracklet historical motion and the 
    direction similarity of tracklet/detections
    i.e., this func measures the motion trend

    used in (Deep) OC SORT, Hybrid SORT and TrackTrack

    Args:
        det_tlbrs: np.ndarray, shape (N, 4)
        previous_obs: np.ndarray, shape (M, 4)
        velocities: np.ndarray, shape (M, 2)
        mode: str, the point position of bbox, center, should be tl, tr, bl or br

    Returns:
        distance: np.ndarray, shape (M, N)
    """

    Y, X = speed_direction_batch(det_tlbrs, previous_obs, mode=mode)
    inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)

    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    return diff_angle


"""
PART II. Assignment and matching functions
"""

def linear_assignment(cost_matrix, thresh):
    """
    solve the optimal assignment problem

    Args:
        cost_matrix: np.ndarray, shape (M, N)
        thresh: float, the maximum cost for matching

    Returns:
        matches: the successful match, np.ndarray, shape (num_of_match, 2)
        unmatched_a: the unsuccessful match in row, np.ndarray, shape (num_of_unmatch, )
        unmatched_a: the unsuccessful match in col, np.ndarray, shape (num_of_unmatch, )
    """


    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def greedy_assignment_cascade(dists, thresh):
    """
    iteratively record the best match
    used in TrackTrack

    Args:
        dists: np.ndarray, shape (m, n)
        thresh: float

    Returns:
        List[List[int, int]], matched row and col idx
    """
    matches = []

    # Run
    if dists.shape[0] > 0 and dists.shape[1] > 0:
        # Get index for minimum similarity
        min_ddx = np.argmin(dists, axis=1)
        min_tdx = np.argmin(dists, axis=0)

        # Match tracks with detections
        for tdx, ddx in enumerate(min_ddx):
            if min_tdx[ddx] == tdx and dists[tdx, ddx] < thresh:
                matches.append([tdx, ddx])

    return matches


def matching_cascade(
        distance_metric, matching_thresh, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    """
    Run matching cascade in DeepSORT

    Args:
        distance_metirc: function that calculate the cost matrix
        matching_thresh: float, Associations with cost larger than this value are disregarded.
        cascade_path: int, equal to max_age of a tracklet
        tracks: List[STrack], current tracks
        detections: List[STrack], current detections
        track_indices: List[int], tracks that will be calculated, Default None
        detection_indices: List[int], detections that will be calculated, Default None

    Returns:
        matched pair, unmatched tracks, unmatced detections: List[int], List[int], List[int]
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    detections_to_match = detection_indices
    matches = []

    for level in range(cascade_depth):
        """
        match new track with detection firstly
        """
        if not len(detections_to_match):  # No detections left
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]  # filter tracks whose age is equal to level + 1 (The age of Newest track = 1)

        if not len(track_indices_l):  # Nothing to match at this level
            continue
        
        # tracks and detections which will be mathcted in current level
        track_l = [tracks[idx] for idx in track_indices_l]  # List[STrack]
        det_l = [detections[idx] for idx in detections_to_match]  # List[STrack]

        # calculate the cost matrix
        cost_matrix = distance_metric(track_l, det_l)

        # solve the linear assignment problem
        matched_row_col, umatched_row, umatched_col = \
            linear_assignment(cost_matrix, matching_thresh)
        
        for row, col in matched_row_col:  # for those who matched
            matches.append((track_indices_l[row], detections_to_match[col]))

        umatched_detecion_l = []  # current detections not matched
        for col in umatched_col:  # for detections not matched
            umatched_detecion_l.append(detections_to_match[col])
        
        detections_to_match = umatched_detecion_l  # update detections to match for next level
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))

    return matches, unmatched_tracks, detections_to_match


def observation_centric_association(tracklets, detections, velocities, previous_obs, vdc_weight=0.05, iou_threshold=0.3):    
    """
    observation centric association, with velocity, for OC Sort

    Args:
        tracklets: List[Tracklet], length of m
        detections: List[Tracklet], length of n
        previous_obs: the nearest location of tracklets, np.ndarray, shape (M, 4)
        velocities: the motion direction of tracklets, np.ndarray, shape (M, 2)
        vdc_weight, iou_thtrshold: float
    
    Returns:
        matches: the successful match, np.ndarray, shape (num_of_match, 2)
        unmatched_a: the unsuccessful match in row, np.ndarray, shape (num_of_unmatch, )
        unmatched_a: the unsuccessful match in col, np.ndarray, shape (num_of_unmatch, ) 
    """

    if len(tracklets) == 0 or len(detections) == 0:
        return np.empty((0, 2), dtype=int), tuple(range(len(tracklets))), tuple(range(len(detections)))
    
    # get numpy format bboxes
    trk_tlbrs = np.array([track.tlbr for track in tracklets])
    det_tlbrs = np.array([det.tlbr for det in detections])
    det_scores = np.array([det.score for det in detections])

    # number of tracklets and detections
    num_of_trks = trk_tlbrs.shape[0]
    num_of_dets = det_tlbrs.shape[0]

    iou_matrix = bbox_ious(trk_tlbrs, det_tlbrs)

    # NOTE for iou < iou_threshold, directly set to -inf, otherwise after solving the linear assignment, 
    # some matched pairs will have no overlaps
    iou_matrix[iou_matrix < iou_threshold] = - 1e5

    diff_angle = angle_distance(det_tlbrs, previous_obs, velocities)

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    scores = np.repeat(det_scores[:, np.newaxis], num_of_trks, axis=1)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], num_of_dets, axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost * scores.T

    matches, unmatched_a, unmatched_b = linear_assignment(- (iou_matrix + angle_diff_cost), thresh=0.0)


    return matches, unmatched_a, unmatched_b

def observation_centric_association_w_reid(tracklets, detections, velocities, previous_obs, vdc_weight=0.05, iou_threshold=0.3, 
                                           aw_off=False, w_assoc_emb=0.5, aw_param=0.5):    
    """
    used in Deep OC SORT

    the Args and Returns is similar to observation_centric_association
    """

    if len(tracklets) == 0 or len(detections) == 0:
        return np.empty((0, 2), dtype=int), tuple(range(len(tracklets))), tuple(range(len(detections)))
    
    # get numpy format bboxes
    trk_tlbrs = np.array([track.tlbr for track in tracklets])
    det_tlbrs = np.array([det.tlbr for det in detections])
    det_scores = np.array([det.score for det in detections])

    # number of tracklets and detections
    num_of_trks = trk_tlbrs.shape[0]
    num_of_dets = det_tlbrs.shape[0]

    iou_matrix = bbox_ious(trk_tlbrs, det_tlbrs)

    # cal embedding distance
    embed_cost = embedding_distance(tracklets, detections, metric='cosine')

    # NOTE for iou < iou_threshold, directly set to -inf, otherwise after solving the linear assignment, 
    # some matched pairs will have no overlaps
    iou_matrix[iou_matrix < iou_threshold] = - 1e5

    diff_angle = angle_distance(det_tlbrs, previous_obs, velocities)

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    scores = np.repeat(det_scores[:, np.newaxis], num_of_trks, axis=1)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], num_of_dets, axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost * scores.T


    # cal embedding cost, eq. 4~6 in paper
    embed_cost[iou_matrix <= 0] = 0
    if not aw_off:
        embed_cost = compute_aw_max_metric(embed_cost, w_assoc_emb, bottom=aw_param)
    else:
        embed_cost *= w_assoc_emb

    matches, unmatched_a, unmatched_b = linear_assignment(- (iou_matrix + angle_diff_cost + embed_cost), thresh=0.0)


    return matches, unmatched_a, unmatched_b


def association_weak_cues(tracklets, detections, velocities, previous_obs, 
                          score_diff_weight=1.0, vdc_weight=0.05, iou_threshold=0.25):    
    """
    observation centric association with four corner point velocity, confidence score and HM IoU, for Hybrid Sort

    the Args and Returns is similar to observation_centric_association
    """

    if len(tracklets) == 0 or len(detections) == 0:
        return np.empty((0, 2), dtype=int), tuple(range(len(tracklets))), tuple(range(len(detections)))
    
    # get numpy format bboxes
    trk_tlbrs = np.array([track.tlbr for track in tracklets])
    det_tlbrs = np.array([det.tlbr for det in detections])
    det_scores = np.array([det.score for det in detections])
    # Note that the kalman-predicted score is used in first round assocication
    trk_scores = np.array([trk.kalman_score for trk in tracklets])   

    # number of tracklets and detections
    num_of_trks = trk_tlbrs.shape[0]
    num_of_dets = det_tlbrs.shape[0]

    # hm iou
    iou_matrix = hm_iou_distance(trk_tlbrs, det_tlbrs)

    # NOTE for iou < iou_threshold, directly set to -inf, otherwise after solving the linear assignment, 
    # some matched pairs will have no overlaps
    iou_matrix[iou_matrix < iou_threshold] = - 1e5

    # cal four corner distance
    velocity_cost = np.zeros((len(tracklets), len(detections)))
    for idx, corner in enumerate(['tl', 'tr', 'bl', 'br']):  # tl, tr, bl, br
        # get the velocity directoin between detections and historical observations

        diff_angle = angle_distance(det_tlbrs, previous_obs, velocities[:, idx], mode=corner)

        valid_mask = np.ones(previous_obs.shape[0])
        valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

        scores = np.repeat(det_scores[:, np.newaxis], num_of_trks, axis=1)
        valid_mask = np.repeat(valid_mask[:, np.newaxis], num_of_dets, axis=1)

        angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
        angle_diff_cost = angle_diff_cost * scores.T

        # add all angle diff cost from four corners
        velocity_cost += angle_diff_cost

    # minus the score difference
    velocity_cost -= score_diff_batch(det_scores, trk_scores) * score_diff_weight

    matches, unmatched_a, unmatched_b = linear_assignment(- (iou_matrix + velocity_cost), thresh=0.0)

    return matches, unmatched_a, unmatched_b

def iterative_assignment(tracklets, dets, dets_second, dets_delete, velocities, previous_obs, 
        match_thresh=0.8, penalty_p=0.2, penalty_q=0.4, reduce_step=0.05, with_reid=True):
    """
    Used in TrackTrack
    assocaition strategy in TPA module

    Args:
        tracklets: List[Tracklet]
        dets, dets_second, dets_delete: List[Tracklet], refers to high-conf dets, low-conf dets and 
            deleted dets in NMS respectively
        velocities, previous_obs: same as observation_centric_association
        match_threshold, penalty_p, penalty_q, reduce_step: float, params in TPA
        with_reid: bool, whether consider the reid features

    Returns:
        same as observation_centric_association 
    """

    dets_all = dets + dets_second + dets_delete  # get all detections

    if len(tracklets) == 0 or len(dets_all) == 0:
        return np.empty((0, 2), dtype=int), tuple(range(len(tracklets))), tuple(range(len(dets_all)))

    dets_all_tlbrs = np.array([det.tlbr for det in dets_all])

    hm_iou_matrix, iou_matrix = hm_iou_distance(tracklets, dets_all, return_original_iou=True)
    hm_iou_dist = 1. - hm_iou_matrix

    cost = 0.5 * hm_iou_dist + 0.5 * embedding_distance(tracklets, dets_all) if with_reid else hm_iou_dist
    # add score dist and angle dist, is same as Hybrid SORT
    # cal angle distance
    diff_angle = 0.
    for idx, corner in enumerate(['tl', 'tr', 'bl', 'br']):  # tl, tr, bl, br
        # get the velocity directoin between detections and historical observations

        diff_angle += angle_distance(dets_all_tlbrs, previous_obs, velocities[:, idx], mode=corner) / 4

    cost += 0.1 * score_distance(tracklets, dets_all) + 0.05 * diff_angle

    # give penalty (Eq. 1 in paper)
    cost[:, len(dets): len(dets + dets_second)] += penalty_p
    cost[:, len(dets + dets_second): ] += penalty_q

    # Constraint & Clip
    cost[iou_matrix <= 0.10] = 1.
    cost = np.clip(cost, 0, 1)

    # Match
    matches = []
    while True:
        # Match tracks with detections
        matches_ = greedy_assignment_cascade(cost, match_thresh)
        match_thresh -= reduce_step

        # Check (if there are no more matchable pairs)
        if len(matches_) == 0:
            break

        # Append
        matches += matches_

        # Update cost matrix
        for t, d in matches:
            cost[t, :] = 1.
            cost[:, d] = 1.

    # Find indices of unmatched tracks and detections
    m_tracks = [t for t, _ in matches]
    u_tracks = [t for t in range(len(tracklets)) if t not in m_tracks]
    m_dets = [d for _, d in matches]
    u_dets = [d for d in range(len(dets_all)) if d not in m_dets]

    return matches, u_tracks, u_dets


"""
PART III. Distance fusion functions
"""

def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    """
    Fuse motion information into cost matrix using Kalman filter gating

    Args:
        kf: Kalman filter object
        cost_matrix: Original cost matrix of shape (M, N)
        tracks: List of track objects
        detections: List of detection objects
        only_position: If True, use only position for gating. Defaults to False
        lambda_: Weight for cost fusion. Defaults to 0.98

    Returns:
        np.ndarray: Fused cost matrix of shape (M, N)
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    """
    Fuse IoU similarity into ReID cost matrix.

    Args:
        cost_matrix: ReID cost matrix of shape (M, N)
        tracks: List of track objects
        detections: List of detection objects

    Returns:
        np.ndarray: Fused cost matrix of shape (M, N)
    """
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_det_score(cost_matrix, detections):
    """
    Weight detection scores into cost matrix.

    Args:
        cost_matrix: Original cost matrix of shape (M, N)
        detections: List of detection objects

    Returns:
        np.ndarray: Weighted cost matrix of shape (M, N)
    """
    # weight detection score into cost matrix
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_det_trk_score(cost_matrix, detections, tracks):
    """
    Weight both detection and track scores into cost matrix

    Args:
        cost_matrix: Original cost matrix of shape (M, N)
        detections: List of detection objects
        tracks: List of track objects

    Returns:
        np.ndarray: Weighted cost matrix of shape (M, N)
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    trk_scores = np.array([trk.score for trk in tracks])
    trk_scores = np.expand_dims(trk_scores, axis=1).repeat(cost_matrix.shape[1], axis=1)
    mid_scores = (det_scores + trk_scores) / 2
    fuse_sim = iou_sim * mid_scores
    fuse_cost = 1 - fuse_sim
    
    return fuse_cost



"""
PART IV. Helper and auxiliary functions
"""

def bbox_ious(atlbrs, btlbrs, use_torchvision=False):
    """
    the function replace of the cython_bbox.bbox_overlaps.bbox_ious
    due to the numpy version issue

    Args:
        atlbrs: np.ndarray, shape (M, 4)
        btlbrs: np.ndarray, shape (N, 4)
        use_torchvision: bool, whether use torchvision.box_iou
    
    Returns:
        iou_matrix: np.ndarray, shape (M, N)
    """
    if use_torchvision:
        atlbrs_tensor = torch.from_numpy(atlbrs.astype(np.float32))
        btlbrs_tensor = torch.from_numpy(btlbrs.astype(np.float32))
        iou_matrix = box_iou(atlbrs_tensor, btlbrs_tensor).numpy()
    else:
        btlbrs_ = np.expand_dims(btlbrs, 0)
        atlbrs_ = np.expand_dims(atlbrs, 1)
        
        xx1 = np.maximum(atlbrs_[..., 0], btlbrs_[..., 0])
        yy1 = np.maximum(atlbrs_[..., 1], btlbrs_[..., 1])
        xx2 = np.minimum(atlbrs_[..., 2], btlbrs_[..., 2])
        yy2 = np.minimum(atlbrs_[..., 3], btlbrs_[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        iou_matrix = wh / ((atlbrs_[..., 2] - atlbrs_[..., 0]) * (atlbrs_[..., 3] - atlbrs_[..., 1])                                      
            + (btlbrs_[..., 2] - btlbrs_[..., 0]) * (btlbrs_[..., 3] - btlbrs_[..., 1]) - wh)      

    return iou_matrix

def speed_direction_batch(dets, tracks, mode='center'):
    """
    helper func of observation_centric_association (OC Sort) and association_weak_cues (Hybrid Sort)

    Args:
        dets: np.ndaray (N, 4), tlbr format bbox
        tracks: np.ndaray (M, 4), tlbr format bbox
        mode: str, the position of point in bbox

    Returns:
        dy, dx: np.ndarray, shape (M, N), the direction diff between dets and trks
    """
    # check the dim of dets
    if len(dets.shape) == 1:
        dets = dets[np.newaxis, ...]

    tracks = tracks[..., np.newaxis]
    
    if mode == 'center':
        CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:,1] + dets[:,3]) / 2.0
        CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    elif mode == 'tl':
        CX1, CY1 = dets[:,0], dets[:,1]
        CX2, CY2 = tracks[:,0], tracks[:,1]
    elif mode == 'tr':
        CX1, CY1 = dets[:,2], dets[:,1]
        CX2, CY2 = tracks[:,2], tracks[:,1]
    elif mode == 'bl':
        CX1, CY1 = dets[:,0], dets[:,3]
        CX2, CY2 = tracks[:,0], tracks[:,3]
    else:
        CX1, CY1 = dets[:,2], dets[:,3]
        CX2, CY2 = tracks[:,2], tracks[:,3]

    dx = CX2 - CX1 
    dy = CY2 - CY1 
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm 
    dy = dy / norm
    return dy, dx  # size: num_track x num_det


def score_diff_batch(det_scores, track_scores):
    """
    helper func of association_weak_cues (Hybrid Sort)

    Args:
        det_scores, np.ndarray, shape (N, )
        track_scores, np.ndarray, shape (M, )
    
    Returns:
        np.ndarray, shape (M, N)
    """
    track_scores = track_scores[:, None]
    det_scores = det_scores[None, :]
    return np.abs(track_scores - det_scores)


def cal_cosine_distance(mat1, mat2):
    """
    simple func to calculate cosine distance between 2 matrixs
    used in nearest_embedding_distance

    Args:
        mat1: np.ndarray, shape(M, dim)
        mat2: np.ndarray, shape(N, dim)

    Returns:
        np.ndarray, shape(M, N)
    """
    # result = mat1·mat2^T / |mat1|·|mat2|
    # norm mat1 and mat2
    mat1 = mat1 / np.linalg.norm(mat1, axis=1, keepdims=True)
    mat2 = mat2 / np.linalg.norm(mat2, axis=1, keepdims=True)

    return np.dot(mat1, mat2.T)  

def compute_aw_max_metric(embed_cost, w_association_emb, bottom=0.5):
    """
    observation centric association, with velocity and reid feature, for Deep OC Sort
    helper func of observation_centric_association_w_reid

    Args:
        embed_cost: np.ndarray
        w_association_emb, bottom: float, corresponding weights

    Returns:
        np.ndarray, shape: embed_cost.shape
    """
    w_emb = np.full_like(embed_cost, w_association_emb)

    for idx in range(embed_cost.shape[0]):
        inds = np.argsort(-embed_cost[idx])
        # If there's less than two matches, just keep original weight
        if len(inds) < 2:
            continue
        if embed_cost[idx, inds[0]] == 0:
            row_weight = 0
        else:
            row_weight = 1 - max(
                (embed_cost[idx, inds[1]] / embed_cost[idx, inds[0]]) - bottom, 0
            ) / (1 - bottom)
        w_emb[idx] *= row_weight

    for idj in range(embed_cost.shape[1]):
        inds = np.argsort(-embed_cost[:, idj])
        # If there's less than two matches, just keep original weight
        if len(inds) < 2:
            continue
        if embed_cost[inds[0], idj] == 0:
            col_weight = 0
        else:
            col_weight = 1 - max(
                (embed_cost[inds[1], idj] / embed_cost[inds[0], idj]) - bottom, 0
            ) / (1 - bottom)
        w_emb[:, idj] *= col_weight

    return w_emb * embed_cost