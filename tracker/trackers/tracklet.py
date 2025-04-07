"""
implements base elements of trajectory
"""

import numpy as np 
from collections import deque

from .basetrack import BaseTrack, TrackState 
from .kalman_filters.bytetrack_kalman import ByteKalman
from .kalman_filters.botsort_kalman import BotKalman
from .kalman_filters.ocsort_kalman import OCSORTKalman
from .kalman_filters.sort_kalman import SORTKalman
from .kalman_filters.strongsort_kalman import NSAKalman
from .kalman_filters.ucmctrack_kalman import UCMCKalman
from .kalman_filters.hybridsort_kalman import HybridSORTKalman

MOTION_MODEL_DICT = {
    'sort': SORTKalman, 
    'byte': ByteKalman, 
    'bot': BotKalman, 
    'ocsort': OCSORTKalman, 
    'strongsort': NSAKalman,
    'ucmc': UCMCKalman,  
    'hybridsort': HybridSORTKalman
}

STATE_CONVERT_DICT = {
    'sort': 'xysa', 
    'byte': 'xyah', 
    'bot': 'xywh', 
    'ocsort': 'xysa', 
    'strongsort': 'xyah',
    'ucmc': 'ground', 
    'hybridsort': 'xysca'
}

class Tracklet(BaseTrack):
    def __init__(self, tlwh, score, category, motion='byte'):

        # initial position
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.is_activated = False

        self.score = score
        self.category = category

        # kalman
        self.motion = motion
        self.kalman_filter = MOTION_MODEL_DICT[motion]()
        
        self.convert_func = self.__getattribute__('tlwh_to_' + STATE_CONVERT_DICT[motion])

        # init kalman
        self.kalman_filter.initialize(self.convert_func(self._tlwh))

    def predict(self):
        self.kalman_filter.predict(is_activated=self.state == TrackState.Tracked)
        self.time_since_update += 1

    def activate(self, frame_id):
        self.track_id = self.next_id()

        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id


    def re_activate(self, new_track, frame_id, new_id=False):
        self.frame_id = frame_id
        
        # TODO different convert
        self.kalman_filter.update(self.convert_func(new_track.tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True
        
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id

        new_tlwh = new_track.tlwh
        self.score = new_track.score

        self.kalman_filter.update(self.convert_func(new_tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True

        self.time_since_update = 0
    
    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        return self.__getattribute__(STATE_CONVERT_DICT[self.motion] + '_to_tlwh')()
    
    def xyah_to_tlwh(self, ):
        x = self.kalman_filter.kf.x 
        ret = x[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def xywh_to_tlwh(self, ):
        x = self.kalman_filter.kf.x 
        ret = x[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret
    
    def xysa_to_tlwh(self, ):
        x = self.kalman_filter.kf.x 
        ret = x[:4].copy()
        ret[2] = np.sqrt(x[2] * x[3])
        ret[3] = x[2] / ret[2]

        ret[:2] -= ret[2:] / 2
        return ret
    

class Tracklet_w_reid(Tracklet):
    """
    Tracklet class with reid features, for botsort, deepsort, etc.
    """
    
    def __init__(self, tlwh, score, category, motion='byte', 
                 feat=None, feat_history=50):
        super().__init__(tlwh, score, category, motion)

        self.smooth_feat = None  # EMA feature
        self.curr_feat = None  # current feature
        self.features = deque([], maxlen=feat_history)  # all features
        if feat is not None:
            self.update_features(feat)

        self.alpha = 0.9

    def update_features(self, feat, alpha=None):
        '''
        alpha: if specified, use alpha instead of self.alpha
        '''
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            alpha_ = self.alpha if alpha is None else alpha
            self.smooth_feat = alpha_ * self.smooth_feat + (1 - alpha_) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def re_activate(self, new_track, frame_id, new_id=False):
        
        # TODO different convert
        if isinstance(self.kalman_filter, NSAKalman):
            self.kalman_filter.update(self.convert_func(new_track.tlwh), new_track.score)
        else:
            self.kalman_filter.update(self.convert_func(new_track.tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id

        new_tlwh = new_track.tlwh
        self.score = new_track.score

        if isinstance(self.kalman_filter, NSAKalman):
            self.kalman_filter.update(self.convert_func(new_tlwh), self.score)
        else:
            self.kalman_filter.update(self.convert_func(new_tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True


        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.time_since_update = 0


class Tracklet_w_velocity(Tracklet):
    """
    Tracklet class with center point velocity, for ocsort or deep ocsort
    """
    
    def __init__(self, tlwh, score, category, motion='byte', delta_t=3, 
                 feat=None, feat_history=50, det_conf_thresh=0.1):
        super().__init__(tlwh, score, category, motion)

        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

        self.age = 0  # mark the age

        # reid featurs, for deep ocsort
        self.features = deque([], maxlen=feat_history)  # all features
        self.smooth_feat = None  # EMA feature
        self.curr_feat = None  # current feature
        if feat is not None:
            self.update_features(feat)

        # the dynamic alpha in eq.2~3 in deep ocsort paper
        self.alpha_fixed_emb = 0.95  # defult param
        trust = (score - det_conf_thresh) / (1. - det_conf_thresh)
        self.dynamic_alpha = self.alpha_fixed_emb + (1. - self.alpha_fixed_emb) * (1. - trust)

    def update_features(self, feat, alpha=1.0):
        '''
        alpha: if specified, use alpha instead of self.alpha
        '''
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    @property
    def tlwh(self):
        """
        NOTE: note that for OC Sort, when querying tlwh, instead of returning the kalman state, 
        directly return the last observation (so is called observation-centric)
        """
        if self.last_observation.sum() < 0:  # no last observation
            return self.__getattribute__(STATE_CONVERT_DICT[self.motion] + '_to_tlwh')()
        
        return self.tlbr_to_tlwh(self.last_observation[: 4])

    @staticmethod
    def speed_direction(bbox1, bbox2):
        cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
        cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
        speed = np.array([cy2 - cy1, cx2 - cx1])
        norm = np.sqrt((cy2 - cy1)**2 + (cx2 - cx1)**2) + 1e-6
        return speed / norm
    
    def predict(self):
        self.kalman_filter.predict(is_activated=self.state == TrackState.Tracked)

        self.age += 1
        self.time_since_update += 1

    def update(self, new_track, frame_id):
        self.frame_id = frame_id

        new_tlwh = new_track.tlwh
        self.score = new_track.score

        self.kalman_filter.update(self.convert_func(new_tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0

        # update velocity and history buffer
        new_tlbr = self.tlwh_to_tlbr(new_tlwh)

        if self.last_observation.sum() >= 0:  # exists previous observation
            previous_box = None
            for dt in range(self.delta_t, 0, -1):  # from old to new
                if self.age - dt in self.observations:
                    previous_box = self.observations[self.age - dt]
                    break
            if previous_box is None:
                previous_box = self.last_observation
            """
                Estimate the track speed direction with observations \Delta t steps away
            """
            self.velocity = self.speed_direction(previous_box, new_tlbr)

        new_observation = np.r_[new_tlbr, new_track.score]
        self.last_observation = new_observation
        self.observations[self.age] = new_observation
        self.history_observations.append(new_observation)

        # update reid features
        if self.curr_feat is not None:
            self.update_features(self.curr_feat, alpha=new_track.dynamic_alpha)


class Tracklet_w_velocity_four_corner(Tracklet):
    """
    Tracklet class with four corner points velocity and previous confidence, for hybrid sort.
    """
    def __init__(self, tlwh, score, category, motion='byte', delta_t=3, score_thresh=0.4):
        # initial position
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.is_activated = False

        self.score = score
        self.category = category

        # kalman
        self.motion = motion
        self.kalman_filter = MOTION_MODEL_DICT[motion]()
        
        self.convert_func = self.__getattribute__('tlwh_to_' + STATE_CONVERT_DICT[motion])

        # init kalman
        self.kalman_filter.initialize(self.convert_func(np.r_[self._tlwh, self.score]))  # confidence score is addtional

        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []

        # velocity of top-left, top-right, bottom-left, bottom-right
        self.velocity_tl, self.velocity_tr, self.velocity_bl, self.velocity_br = None, None, None, None
        # prev score
        self.prev_score = None

        self.score_thresh = score_thresh  # score threshold to limit the range of kalman-predicted score and observation score

        self.delta_t = delta_t

        self.age = 0  # mark the age

    @property
    def tlwh(self):
        """
        NOTE: note that for Hybrid Sort, same as OC Sort, when querying tlwh, instead of returning the kalman state, 
        directly return the last observation 
        """
        if self.last_observation.sum() < 0:  # no last observation
            return self.__getattribute__(STATE_CONVERT_DICT[self.motion] + '_to_tlwh')()
        
        return self.tlbr_to_tlwh(self.last_observation[: 4])

    @staticmethod
    def speed_direction(point1, point2):
        """
        In order to jointly calculating the four corner velocity, parse point coordinate as input. 
        
        Args:
            point1, point2: list or np.ndarray, shape: (2, )
        """
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        speed = np.array([y2 - y1, x2 - x1])
        norm = np.sqrt((y2 - y1)**2 + (x2 - x1)**2) + 1e-6
        return speed / norm

    def predict(self):
        self.kalman_filter.predict(is_activated=self.state == TrackState.Tracked)

        self.age += 1
        self.time_since_update += 1

        # update score with linear model
        if not self.prev_score:
            self.score = np.clip(self.score, 0.1, self.score_thresh)
        else:
            self.score = np.clip(self.score + (self.score - self.prev_score), 0.1, self.score_thresh)

    def update(self, new_track, frame_id):
        self.frame_id = frame_id

        new_tlwh = new_track.tlwh
        self.prev_score = self.score  # save previous score
        self.score = new_track.score

        self.kalman_filter.update(self.convert_func(np.r_[new_tlwh, new_track.score]))

        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0
        
        # get four corner velocity
        new_tlbr = self.tlwh_to_tlbr(new_tlwh)

        self.velocity_tl, self.velocity_tr = np.array([0, 0], dtype=float), np.array([0, 0], dtype=float)
        self.velocity_bl, self.velocity_br = np.array([0, 0], dtype=float), np.array([0, 0], dtype=float)

        if self.last_observation.sum() >= 0:  # exists previous observation
            previous_box = None
            for dt in range(1, self.delta_t + 1):  # from new to old
                if self.age - dt in self.observations:
                    previous_box = self.observations[self.age - dt]  # t-l-b-r

                    self.velocity_tl += self.speed_direction([previous_box[0], previous_box[1]], [new_tlbr[0], new_tlbr[1]])
                    self.velocity_tr += self.speed_direction([previous_box[2], previous_box[1]], [new_tlbr[2], new_tlbr[1]])
                    self.velocity_bl += self.speed_direction([previous_box[0], previous_box[3]], [new_tlbr[0], new_tlbr[3]])
                    self.velocity_br += self.speed_direction([previous_box[2], previous_box[3]], [new_tlbr[2], new_tlbr[3]])

            if previous_box is None:       
                previous_box = self.last_observation

                self.velocity_tl += self.speed_direction([previous_box[0], previous_box[1]], [new_tlbr[0], new_tlbr[1]])
                self.velocity_tr += self.speed_direction([previous_box[2], previous_box[1]], [new_tlbr[2], new_tlbr[1]])
                self.velocity_bl += self.speed_direction([previous_box[0], previous_box[3]], [new_tlbr[0], new_tlbr[3]])
                self.velocity_br += self.speed_direction([previous_box[2], previous_box[3]], [new_tlbr[2], new_tlbr[3]])

        new_observation = np.r_[new_tlbr, new_track.score]
        self.last_observation = new_observation
        self.observations[self.age] = new_observation
        self.history_observations.append(new_observation)

    def re_activate(self, new_track, frame_id, new_id=False):
        
        self.kalman_filter.update(self.convert_func(np.r_[new_track.tlwh, new_track.score]))

        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def get_velocity(self, ):
        """
        Get four corner velocity
        Return: 
            np.ndarray, shape (4, 2)
        """
        if self.velocity_bl is None:
            return np.zeros((4, 2))
        
        return np.vstack([self.velocity_tl, 
                          self.velocity_tr,
                          self.velocity_bl, 
                          self.velocity_br, ])

    @property
    def kalman_score(self, ):
        # return kalman-predicted score
        return np.clip(self.kalman_filter.kf.x[3], self.score_thresh, 1.0)
    
    def xysca_to_tlwh(self, ):
        # used in @property tlwh()
        x = self.kalman_filter.kf.x 
        ret = x[:5].copy()
        ret[3], ret[4] = ret[4], ret[3]
        ret = ret[:4]  # xysa

        ret[2] = np.sqrt(x[2] * x[4])  # s * a = w
        ret[3] = x[2] / ret[2]  # s / w = h

        ret[:2] -= ret[2:] / 2

        return ret
        
    @staticmethod
    def tlwh_to_xysca(tlwh):
        # note that tlwh is actually tlwhc
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2: 4] / 2
        ret[2] = tlwh[2] * tlwh[3]
        ret[3] = tlwh[2] / tlwh[3]
        ret[3], ret[4] = ret[4], ret[3]  # xysac -> xysca
        return ret

class Tracklet_w_bbox_buffer(Tracklet):
    """
    Tracklet class with buffer of bbox, for C_BIoU track.
    """
    def __init__(self, tlwh, score, category, motion='byte'):
        # initial position
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.is_activated = False

        self.score = score
        self.category = category

        # Note in C-BIoU tracker the kalman filter is abandoned

        # params in motion state
        self.b1, self.b2, self.n = 0.3, 0.5, 5
        self.origin_bbox_buffer = deque()  # a deque store the original bbox(tlwh) from t - self.n to t, where t is the last time detected
        self.origin_bbox_buffer.append(self._tlwh)
        # buffered bbox, two buffer sizes
        self.buffer_bbox1 = self.get_buffer_bbox(level=1)
        self.buffer_bbox2 = self.get_buffer_bbox(level=2)
        # motion state, s^{t + \delta} = o^t + (\delta / n) * \sum_{i=t-n+1}^t(o^i - o^{i-1}) = o^t + (\delta / n) * (o^t - o^{t - n})
        self.motion_state0 = self._tlwh  # original tlwh
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
    
    def re_activate(self, new_track, frame_id, new_id=False):
        
        # TODO different convert
        self.kalman_filter.update(self.convert_func(new_track.tlwh))

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

        self.buffer_bbox1 = self.get_buffer_bbox(level=1)
        self.buffer_bbox2 = self.get_buffer_bbox(level=2)

        self.motion_state0 = self._tlwh
        self.motion_state1 = self.buffer_bbox1.copy()
        self.motion_state2 = self.buffer_bbox2.copy()

    def predict(self):
        # Note that in C-BIoU Tracker, no need to use Kalman Filter
        self.time_since_update += 1

        # Average motion model: s^{t + \delta} = o^t + (\delta / n) * (o^t - o^{t - n})
        assert len(self.origin_bbox_buffer), 'The bbox buffer is empty'

        motion_state = self.origin_bbox_buffer[-1] + \
                    (self.time_since_update / len(self.origin_bbox_buffer)) * (self.origin_bbox_buffer[-1] - self.origin_bbox_buffer[0])
        
        self.motion_state0 = motion_state
        self.motion_state1 = self.get_buffer_bbox(level=1, bbox=motion_state)
        self.motion_state2 = self.get_buffer_bbox(level=2, bbox=motion_state)


    def update(self, new_track, frame_id):
        self.frame_id = frame_id

        new_tlwh = new_track.tlwh
        self.score = new_track.score

        # self.kalman_filter.update(self.convert_func(new_tlwh))  # no need to use Kalman Filter

        self.state = TrackState.Tracked
        self.is_activated = True

        self.time_since_update = 0

        # update stored bbox
        if (len(self.origin_bbox_buffer) > self.n):
            self.origin_bbox_buffer.popleft()
        self.origin_bbox_buffer.append(new_tlwh)

    # Drop kalman filter, rewrite the tlwh function
    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        return self.motion_state0

class Tracklet_w_depth(Tracklet):
    """
    tracklet with depth info (i.e., 2000 - y2), for SparseTrack
    """

    def __init__(self, tlwh, score, category, motion='byte'):
        super().__init__(tlwh, score, category, motion)


    @property
    # @jit(nopython=True)
    def deep_vec(self):
        """Convert bounding box to format `((top left, bottom right)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        cx = ret[0] + 0.5 * ret[2]
        y2 = ret[1] +  ret[3]
        lendth = 2000 - y2
        return np.asarray([cx, y2, lendth], dtype=np.float)
    

class Tracklet_w_UCMC(Tracklet):
    """
    tracklet with a grounding map and uniform camera motion compensation, for UCMC Track
    """

    configs = dict(
        sigma_x=1.0,  # noise factor in x axis (Eq. 11)
        sigma_y=1.0,  # noise factor in y axis (Eq. 11)
        vmax=1.0,  # TODO
        dt=1/30,  # interval between frames
    )

    KiKo = None  # the multiplication of intrinsic matrix and extrinsic matrix
    A = None  # The A matrix in Eq. 17
    InvA = None 

    def __init__(self, tlwh, score, category, motion='ucmc'):

        # initial position
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.is_activated = False

        self.score = score
        self.category = category

        # kalman
        self.motion = motion
        self.kalman_filter = MOTION_MODEL_DICT[motion](**self.configs)
        
        self.convert_func = self.__getattribute__('tlwh_to_' + STATE_CONVERT_DICT[motion])

        # init kalman
        self.ground_xy, self.sigma_ground_xy = self.convert_func(self._tlwh)  # save as property variable
        self.kalman_filter.initialize(observation=self.ground_xy, R=self.sigma_ground_xy)
        
    def ground_to_tlwh(self, ):
        x_vector = self.kalman_filter.kf.x 
        x, y = x_vector[0, 0], x_vector[2, 0]  # get ground coordinate

        ground_xy = np.array([x, y, 1])

        xc_yc = np.dot(self.A, ground_xy)
        xc_yc[:2] /= xc_yc[2]  # normalization

        w, h = self._tlwh[2], self._tlwh[3]
        xc, yc = xc_yc[0], xc_yc[1]

        ret = np.array([xc - 0.5 * w, yc - h, w, h])  # note xc, yc is the center of bottom line of bbox

        return ret


    def tlwh_to_ground(self, tlwh=None):
        """
        Key function, map tlwh in camera plane to world coordinate ground

        """

        if tlwh is None: tlwh = self._tlwh

        xc, yc = tlwh[0] + tlwh[2] * 0.5, tlwh[1] + tlwh[3]  # the center of bottom line of bbox
        # the uncertainty (variance) of xc, yc
        sigma_xc = max(2, min(13, 0.05 * tlwh[2]))
        sigma_yc = max(2, min(10, 0.05 * tlwh[3]))
        sigma = np.array([[sigma_xc * sigma_xc, 0], 
                          [0, sigma_yc * sigma_yc]])        
        
        # map to ground
        xc_yc = np.array([xc, yc, 1])
        
        b = np.dot(self.InvA, xc_yc)  # Eq. 19
        gamma = 1. / b[2]
        C = gamma * self.InvA[:2, :2] - (gamma**2) * b[:2] * self.InvA[2, :2]  # Eq. 27
 
        ground_xy = b[:2] * gamma  # Eq. 20
        sigma_ground_xy = np.dot(np.dot(C, sigma), C.T)  # Eq. 26

        return ground_xy, sigma_ground_xy

    def cal_maha_distance(self, det_ground_xy, det_sigma_ground_xy):
        """
        cal a mahalanobis dist between a track and det (Eq. 8)
        """
        
        diff = det_ground_xy[:, None] - np.dot(self.kalman_filter.kf.H, self.kalman_filter.kf.x)  # match the dimension
        S = np.dot(self.kalman_filter.kf.H, np.dot(self.kalman_filter.kf.P, self.kalman_filter.kf.H.T)) + det_sigma_ground_xy

        SI = np.linalg.inv(S)
        mahalanobis = np.dot(diff.T, np.dot(SI, diff))
        logdet = np.log(np.linalg.det(S))
        return mahalanobis[0, 0] + logdet
    

    def update(self, new_track, frame_id):
        self.frame_id = frame_id

        self.score = new_track.score

        self.kalman_filter.update(z=new_track.ground_xy, R=new_track.sigma_ground_xy)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.time_since_update = 0

        self._tlwh = new_track._tlwh  # update the tlwh directly for maintaining w and h

    def re_activate(self, new_track, frame_id, new_id=False):
        
        # TODO different convert
        self.kalman_filter.update(z=new_track.ground_xy, R=new_track.sigma_ground_xy)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score