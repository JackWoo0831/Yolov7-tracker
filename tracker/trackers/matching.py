import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
import math
from cython_bbox import bbox_overlaps as bbox_ious
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


def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
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


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
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


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def greedy_assignment_iou(dist, thresh):
        matched_indices = []
        if dist.shape[1] == 0:
            return np.array(matched_indices, np.int32).reshape(-1, 2)
        for i in range(dist.shape[0]):
            j = dist[i].argmin()
            if dist[i][j] < thresh:
                dist[:, j] = 1.
                matched_indices.append([j, i])
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    
def greedy_assignment(dists, threshs):
    matches = greedy_assignment_iou(dists.T, threshs)
    u_det = [d for d in range(dists.shape[1]) if not (d in matches[:, 1])]
    u_track = [d for d in range(dists.shape[0]) if not (d in matches[:, 0])]
    return matches, u_track,  u_det

def fuse_score_matrix(cost_matrix, detections, tracks):
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
calculate buffered IoU, used in C_BIoU_Tracker
"""
def buffered_iou_distance(atracks, btracks, level=1):
    """
    atracks: list[C_BIoUSTrack], tracks
    btracks: list[C_BIoUSTrack], detections
    level: cascade level, 1 or 2
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

"""
observation centric association, with velocity, for OC Sort
"""
def observation_centric_association(tracklets, detections, iou_threshold, velocities, previous_obs, vdc_weight):    

    if(len(tracklets) == 0):
        return np.empty((0, 2), dtype=int), tuple(range(len(tracklets))), tuple(range(len(detections)))
    
    # get numpy format bboxes
    trk_tlbrs = np.array([track.tlbr for track in tracklets])
    det_tlbrs = np.array([det.tlbr for det in detections])
    det_scores = np.array([det.score for det in detections])

    iou_matrix = bbox_ious(trk_tlbrs, det_tlbrs)

    Y, X = speed_direction_batch(det_tlbrs, previous_obs)
    inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    scores = np.repeat(det_scores[:, np.newaxis], trk_tlbrs.shape[0], axis=1)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost * scores.T

    matches, unmatched_a, unmatched_b = linear_assignment(- (iou_matrix + angle_diff_cost), thresh=0.9)


    return matches, unmatched_a, unmatched_b

"""
helper func of observation_centric_association
"""
def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:,1] + dets[:,3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX2 - CX1 
    dy = CY2 - CY1 
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm 
    dy = dy / norm
    return dy, dx  # size: num_track x num_det


def matching_cascade(
        distance_metric, matching_thresh, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    """
    Run matching cascade in DeepSORT

    distance_metirc: function that calculate the cost matrix
    matching_thresh: float, Associations with cost larger than this value are disregarded.
    cascade_path: int, equal to max_age of a tracklet
    tracks: List[STrack], current tracks
    detections: List[STrack], current detections
    track_indices: List[int], tracks that will be calculated, Default None
    detection_indices: List[int], detections that will be calculated, Default None

    return:
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

def nearest_embedding_distance(tracks, detections, metric='cosine'):
    """
    different from embedding distance, this func calculate the 
    nearest distance among all track history features and detections

    tracks: list[STrack]
    detections: list[STrack]
    metric: str, cosine or euclidean
    TODO: support euclidean distance

    return:
    cost_matrix, np.ndarray, shape(len(tracks), len(detections))
    """
    cost_matrix = np.zeros((len(tracks), len(detections)))
    det_features = np.asarray([det.features[-1] for det in detections])

    for row, track in enumerate(tracks):
        track_history_features = np.asarray(track.features)
        dist = 1. - cal_cosine_distance(track_history_features, det_features)
        dist = dist.min(axis=0)
        cost_matrix[row, :] = dist
    
    return cost_matrix

def cal_cosine_distance(mat1, mat2):
    """
    simple func to calculate cosine distance between 2 matrixs
    
    :param mat1: np.ndarray, shape(M, dim)
    :param mat2: np.ndarray, shape(N, dim)
    :return: np.ndarray, shape(M, N)
    """
    # result = mat1·mat2^T / |mat1|·|mat2|
    # norm mat1 and mat2
    mat1 = mat1 / np.linalg.norm(mat1, axis=1, keepdims=True)
    mat2 = mat2 / np.linalg.norm(mat2, axis=1, keepdims=True)

    return np.dot(mat1, mat2.T)  