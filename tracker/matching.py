"""Partly Copyed from JDE code"""


import numpy as np
import scipy
from scipy.spatial.distance import cdist
import lap

from cython_bbox import bbox_overlaps as bbox_ious
import kalman_filter

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

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[STrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.features[-1] for track in detections], dtype=np.float)
    track_features = np.asarray([track.features[-1] for track in tracks], dtype=np.float)
    if metric == 'euclidean':
        cost_matrix = np.maximum(0.0, cdist(track_features, det_features)) # Nomalized features
    elif metric == 'cosine':
        cost_matrix = 1. - cal_cosine_distance(track_features, det_features)
    else:
        raise NotImplementedError
    return cost_matrix

def ecu_iou_distance(tracks, detections, img0_shape):
    """
    combine eculidian center-point distance and iou distance
    :param tracks: list[STrack]
    :param detections: list[STrack]
    :param img0_shape: list or tuple, origial (h, w) of frame image

    :rtype cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    
    # calculate eculid dist
    ecu_dist = []
    
    det_bbox = np.asarray([det.tlwh for det in detections])  # shape: (len(detections), 4)
    trk_bbox = np.asarray([trk.tlwh for trk in tracks])  # shape: (len(tracks), 4)
    
    det_cx, det_cy = det_bbox[:, 0] + 0.5*det_bbox[:, 2], det_bbox[:, 1] + 0.5*det_bbox[:, 3]
    trk_cx, trk_cy = trk_bbox[:, 0] + 0.5*trk_bbox[:, 2], trk_bbox[:, 1] + 0.5*trk_bbox[:, 3]
    for trkIdx in range(len(tracks)):
        # solve center xy
        ecu_dist.append(
            np.sqrt((det_cx - trk_cx[trkIdx])**2 + (det_cy - trk_cy[trkIdx])**2)
        )
    ecu_dist = np.asarray(ecu_dist)
    norm_factor = float((img0_shape[0]**2 + img0_shape[1]**2)**0.5)
    ecu_dist = 1. - np.exp(-5*ecu_dist / norm_factor)

    # calculate iou dist
    iou_dist = iou_distance(tracks, detections)
    cost_matrix = 0.5*(ecu_dist + iou_dist)
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


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1-lambda_)* gating_distance
    return cost_matrix
