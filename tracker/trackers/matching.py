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

"""
Some basic functions
"""

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


def fuse_det_score(cost_matrix, detections):
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
    # weight detection and tracklet score into cost matrix
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
def observation_centric_association(tracklets, detections, velocities, previous_obs, vdc_weight=0.05, iou_threshold=0.3):    

    if len(tracklets) == 0 or len(detections) == 0:
        return np.empty((0, 2), dtype=int), tuple(range(len(tracklets))), tuple(range(len(detections)))
    
    # get numpy format bboxes
    trk_tlbrs = np.array([track.tlbr for track in tracklets])
    det_tlbrs = np.array([det.tlbr for det in detections])
    det_scores = np.array([det.score for det in detections])

    iou_matrix = bbox_ious(trk_tlbrs, det_tlbrs)

    # NOTE for iou < iou_threshold, directly set to -inf, otherwise after solving the linear assignment, 
    # some matched pairs will have no overlaps
    iou_matrix[iou_matrix < iou_threshold] = - 1e5

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

    matches, unmatched_a, unmatched_b = linear_assignment(- (iou_matrix + angle_diff_cost), thresh=0.0)


    return matches, unmatched_a, unmatched_b

"""
observation centric association, with velocity and reid feature, for Deep OC Sort
"""
def compute_aw_max_metric(embed_cost, w_association_emb, bottom=0.5):
    '''
    helper func of observation_centric_association_w_reid
    '''
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

def observation_centric_association_w_reid(tracklets, detections, velocities, previous_obs, vdc_weight=0.05, iou_threshold=0.3, 
                                           aw_off=False, w_assoc_emb=0.5, aw_param=0.5):    

    if len(tracklets) == 0 or len(detections) == 0:
        return np.empty((0, 2), dtype=int), tuple(range(len(tracklets))), tuple(range(len(detections)))
    
    # get numpy format bboxes
    trk_tlbrs = np.array([track.tlbr for track in tracklets])
    det_tlbrs = np.array([det.tlbr for det in detections])
    det_scores = np.array([det.score for det in detections])

    iou_matrix = bbox_ious(trk_tlbrs, det_tlbrs)

    # cal embedding distance
    embed_cost = 1. - embedding_distance(tracklets, detections, metric='cosine')

    # NOTE for iou < iou_threshold, directly set to -inf, otherwise after solving the linear assignment, 
    # some matched pairs will have no overlaps
    iou_matrix[iou_matrix < iou_threshold] = - 1e5

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


    # cal embedding cost, eq. 4~6 in paper
    embed_cost[iou_matrix <= 0] = 0
    if not aw_off:
        embed_cost = compute_aw_max_metric(embed_cost, w_assoc_emb, bottom=aw_param)
    else:
        embed_cost *= w_assoc_emb

    matches, unmatched_a, unmatched_b = linear_assignment(- (iou_matrix + angle_diff_cost + embed_cost), thresh=0.0)


    return matches, unmatched_a, unmatched_b



"""
helper func of observation_centric_association (OC Sort) and association_weak_cues (Hybrid Sort)
"""
def speed_direction_batch(dets, tracks, mode='center'):
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

"""
helper func of association_weak_cues (Hybrid Sort)
"""
def score_diff_batch(det_scores, track_scores):
    """
    Args:
    det_scores, np.ndarray, shape (N, )
    track_scores, np.ndarray, shape (M, )
    """
    track_scores = track_scores[:, None]
    det_scores = det_scores[None, :]
    return np.abs(track_scores - det_scores)

def score_distance(atracks, btracks):
    """
    calculate the confidence score difference, c_{i, j} = abs(atracks[i].score - btracks[j].score)
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        ascores = atracks
        bscores = btracks
    else:
        ascores = [track.score for track in atracks]
        bscores = [track.score for track in btracks]

    return score_diff_batch(det_scores=np.ascontiguousarray(bscores), 
                            track_scores=np.ascontiguousarray(ascores))

"""
calculate HM IoU, used in Hybrid Sort
"""
def hm_iou_distance(atracks, btracks):
    # hm iou = iou * hright iou
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = ious(atlbrs, btlbrs)  # original iou

    if _ious.size == 0: 
        return _ious  # case if len of tracks == 0, no need to further calculating

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

    return _ious * _h_ious


"""
observation centric association with four corner point velocity, confidence score and HM IoU, for Hybrid Sort
"""
def association_weak_cues(tracklets, detections, velocities, previous_obs, 
                          score_diff_weight=1.0, vdc_weight=0.05, iou_threshold=0.25):    

    if len(tracklets) == 0 or len(detections) == 0:
        return np.empty((0, 2), dtype=int), tuple(range(len(tracklets))), tuple(range(len(detections)))
    
    # get numpy format bboxes
    trk_tlbrs = np.array([track.tlbr for track in tracklets])
    det_tlbrs = np.array([det.tlbr for det in detections])
    det_scores = np.array([det.score for det in detections])
    # Note that the kalman-predicted score is used in first round assocication
    trk_scores = np.array([trk.kalman_score for trk in tracklets])   

    # hm iou
    iou_matrix = hm_iou_distance(trk_tlbrs, det_tlbrs)

    # NOTE for iou < iou_threshold, directly set to -inf, otherwise after solving the linear assignment, 
    # some matched pairs will have no overlaps
    iou_matrix[iou_matrix < iou_threshold] = - 1e5

    # cal four corner distance
    velocity_cost = np.zeros((len(tracklets), len(detections)))
    for idx, corner in enumerate(['tl', 'tr', 'bl', 'br']):  # tl, tr, bl, br
        # get the velocity directoin between detections and historical observations
        Y, X = speed_direction_batch(det_tlbrs, previous_obs, mode=corner)   # shape (num track, num det)
        inertia_Y, inertia_X = velocities[:, idx, 0], velocities[:, idx, 1]  # velocities: shape (N, 4, 2)
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

        # add all angle diff cost from four corners
        velocity_cost += angle_diff_cost

    # minus the score difference
    velocity_cost -= score_diff_batch(det_scores, trk_scores) * score_diff_weight

    matches, unmatched_a, unmatched_b = linear_assignment(- (iou_matrix + velocity_cost), thresh=0.0)

    return matches, unmatched_a, unmatched_b



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