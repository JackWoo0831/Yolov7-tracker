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

MOTION_MODEL_DICT = {
    'sort': SORTKalman, 
    'byte': ByteKalman, 
    'bot': BotKalman, 
    'ocsort': OCSORTKalman, 
    'strongsort': NSAKalman, 
}

STATE_CONVERT_DICT = {
    'sort': 'xysa', 
    'byte': 'xyah', 
    'bot': 'xywh', 
    'ocsort': 'xysa', 
    'strongsort': 'xyah'
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
        self.kalman_filter.predict()
        self.time_since_update += 1

    def activate(self, frame_id):
        self.track_id = self.next_id()

        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id


    def re_activate(self, new_track, frame_id, new_id=False):
        
        # TODO different convert
        self.kalman_filter.update(self.convert_func(new_track.tlwh))

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

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
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
    Tracklet class with reid features, for ocsort.
    """
    
    def __init__(self, tlwh, score, category, motion='byte', delta_t=3):
        super().__init__(tlwh, score, category, motion)

        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

        self.age = 0  # mark the age

    @staticmethod
    def speed_direction(bbox1, bbox2):
        cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
        cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
        speed = np.array([cy2 - cy1, cx2 - cx1])
        norm = np.sqrt((cy2 - cy1)**2 + (cx2 - cx1)**2) + 1e-6
        return speed / norm
    
    def predict(self):
        self.kalman_filter.predict()

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
        new_tlbr = Tracklet_w_bbox_buffer.tlwh_to_tlbr(new_tlwh)

        if self.last_observation.sum() >= 0:  # no previous observation
            previous_box = None
            for i in range(self.delta_t):
                dt = self.delta_t - i
                if self.age - dt in self.observations:
                    previous_box = self.observations[self.age-dt]
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


    

class Tracklet_w_bbox_buffer(Tracklet):
    """
    Tracklet class with buffer of bbox, for C_BIoU track.
    """
    def __init__(self, tlwh, score, category, motion='byte'):
        super().__init__(tlwh, score, category, motion)

        # params in motion state
        self.b1, self.b2, self.n = 0.3, 0.5, 5
        self.origin_bbox_buffer = deque()  # a deque store the original bbox(tlwh) from t - self.n to t, where t is the last time detected
        self.origin_bbox_buffer.append(self._tlwh)
        # buffered bbox, two buffer sizes
        self.buffer_bbox1 = self.get_buffer_bbox(level=1)
        self.buffer_bbox2 = self.get_buffer_bbox(level=2)
        # motion state, s^{t + \delta} = o^t + (\delta / n) * \sum_{i=t-n+1}^t(o^i - o^{i-1}) = o^t + (\delta / n) * (o^t - o^{t - n})
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
        else:
            self.origin_bbox_buffer.append(self._tlwh)

        self.buffer_bbox1 = self.get_buffer_bbox(level=1)
        self.buffer_bbox2 = self.get_buffer_bbox(level=2)
        self.motion_state1 = self.buffer_bbox1.copy()
        self.motion_state2 = self.buffer_bbox2.copy()

    def update(self, new_track, frame_id):
        self.frame_id = frame_id

        new_tlwh = new_track.tlwh
        self.score = new_track.score

        self.kalman_filter.update(self.convert_func(new_tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True

        self.time_since_update = 0

        # update stored bbox
        if (len(self.origin_bbox_buffer) > self.n):
            self.origin_bbox_buffer.popleft()
            self.origin_bbox_buffer.append(new_tlwh)
        else:
            self.origin_bbox_buffer.append(new_tlwh)

        # update motion state
        if self.time_since_update:  # have some unmatched frames
            if len(self.origin_bbox_buffer) < self.n:
                self.motion_state1 = self.get_buffer_bbox(level=1, bbox=new_tlwh)
                self.motion_state2 = self.get_buffer_bbox(level=2, bbox=new_tlwh)
            else:  # s^{t + \delta} = o^t + (\delta / n) * (o^t - o^{t - n})
                motion_state = self.origin_bbox_buffer[-1] + \
                    (self.time_since_update / self.n) * (self.origin_bbox_buffer[-1] - self.origin_bbox_buffer[0])
                self.motion_state1 = self.get_buffer_bbox(level=1, bbox=motion_state)
                self.motion_state2 = self.get_buffer_bbox(level=2, bbox=motion_state)

        else:  # no unmatched frames, use current detection as motion state
            self.motion_state1 = self.get_buffer_bbox(level=1, bbox=new_tlwh)
            self.motion_state2 = self.get_buffer_bbox(level=2, bbox=new_tlwh)


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