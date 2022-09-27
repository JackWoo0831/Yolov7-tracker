"""Partly Copyed from JDE code"""


import numpy as np
import scipy
from scipy.spatial.distance import cdist
import lap

from cython_bbox import bbox_overlaps as bbox_ious
import kalman_filter
import math 

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

def cal_eculidian_distance(mat1, mat2):
    """
    NOTE: another version to cal ecu dist

    simple func to calculate ecu distance between 2 matrixs
    
    :param mat1: np.ndarray, shape(M, dim)
    :param mat2: np.ndarray, shape(N, dim)
    :return: np.ndarray, shape(M, N)
    """
    if len(mat1) == 0 or len(mat2) == 0:
        return np.zeros((len(mat1), len(mat2)))

    mat1_sq, mat2_sq = np.square(mat1).sum(axis=1), np.square(mat2).sum(axis=1)

    # -2ab + a^2 + b^2 = (a-b)^2
    dist = -2 * np.dot(mat1, mat2.T) + mat1_sq[:, None] + mat2_sq[None, :]
    dist = np.clip(dist, 0, np.inf)

    return np.minimum(0.0, dist.min(axis=0))
    

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


"""
distance metric that combines multi-frame info
used in StrongSORT
TODO: use in DeepSORT
"""

class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = cal_eculidian_distance
        elif metric == "cosine":
            self._metric = cal_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix


"""
funcs to cal similarity, copied from UAVMOT
"""
def local_relation_fuse_motion(cost_matrix,
                tracks,
                detections,
                only_position=False,
                lambda_=0.98):
    """
    :param kf:
    :param cost_matrix:
    :param tracks:
    :param detections:
    :param only_position:
    :param lambda_:
    :return:
    """

    # print(cost_matrix.shape)
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    # gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.tlwh2xyah() for det in detections])
    structure_distance = structure_similarity_distance(tracks,
                                                       detections)
    cost_matrix = lambda_ * cost_matrix + (1 - lambda_) * structure_distance

    return cost_matrix
def structure_similarity_distance(tracks, detections):
    track_structure = structure_representation(tracks)
    detection_structure = structure_representation(detections,mode='detection')

    # for debug
    # print(track_structure.shape, detection_structure.shape)
    # exit(0)
    cost_matrix = np.maximum(0.0, cdist(track_structure, detection_structure, metric="cosine"))

    return cost_matrix
def angle(v1, v2):
    # dx1 = v1[2] - v1[0]
    # dy1 = v1[3] - v1[1]
    # dx2 = v2[2] - v2[0]
    # dy2 = v2[3] - v2[1]
    dx1 = v1[0]
    dy1 = v1[1]
    dx2 = v2[0]
    dy2 = v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

def structure_representation(tracks,mode='trcak'):
    local_R =400
    structure_matrix =[]
    for i, track_A in enumerate(tracks):
        length = []
        index =[]
        for j, track_B in enumerate(tracks):
            # print(track_A.mean[0:2])
            # pp: 中心点距离  shape: (1, )
            if mode =="detection":
                pp = list(
                    map(lambda x: np.linalg.norm(np.array(x[0] - x[1])), zip(track_A.get_xy(), track_B.get_xy())))
            else:
                pp=list(map(lambda x: np.linalg.norm(np.array(x[0] - x[1])), zip(track_A.mean[0:2],track_B.mean[0:2])))
            lgt = np.linalg.norm(pp)
            if lgt < local_R and lgt >0:
                length.append(lgt)
                index.append(j)

        if length==[]:
            v =[0.0001,0.0001,0.0001]

        else:
            max_length = max(length)
            min_length = min(length)
            if max_length == min_length:
                v = [max_length, min_length, 0.0001]
            else:
                max_index = index[length.index(max_length)]
                min_index = index[length.index(min_length)]
                if mode == "detection":
                    v1 = tracks[max_index].get_xy() - track_A.get_xy()
                    v2 = tracks[min_index].get_xy() - track_A.get_xy()
                else:
                    v1 = tracks[max_index].mean[0:2] - track_A.mean[0:2]
                    v2 = tracks[min_index].mean[0:2] - track_A.mean[0:2]

                include_angle = angle(v1, v2)
                v = [max_length, min_length, include_angle]

        structure_matrix.append(v)

    return np.asarray(structure_matrix)