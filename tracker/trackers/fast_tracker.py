"""
FastTracker
"""

import numpy as np
from collections import deque
from .basetrack import BaseTrack, TrackState
from .tracklet import Tracklet, Tracklet_w_reid, Tracklet_w_occluded
from .matching import *

# for reid
import torch
import torchvision.transforms as T
from .reid_models.engine import load_reid_model, crop_and_resize

# base class
from .basetracker import BaseTracker

# config w.r.t. occlusion modeling and road ROI
FAST_TRACKER_CONFIG = {
    "reset_velocity_offset_occ": 5,
    "reset_pos_offset_occ": 3,
    "enlarge_bbox_occ": 1.1,
    "dampen_motion_occ": 0.89,
    "active_occ_to_lost_thresh": 10,
    "init_iou_suppress": 0.8,

    # "ROIs": {
    #     "roi_1_points": [[312, 196], [422, 188], [1399, 694], [152, 697]],
    #     "roi_2_points": [[774, 407], [1541, 382], [614, 163], [498, 176]]
    # },
    "roi_repair_max_gap": 15,   
    "dir_window_N": 10,         
    "dir_margin_deg": 2.0       
}


class FastTracker(BaseTracker):
    def __init__(self, args, frame_rate=30):
        
        super().__init__(args, frame_rate=frame_rate)

        self.reset_velocity_offset_occ = FAST_TRACKER_CONFIG["reset_velocity_offset_occ"]
        self.reset_pos_offset_occ = FAST_TRACKER_CONFIG["reset_pos_offset_occ"]
        self.enlarge_bbox_occ = FAST_TRACKER_CONFIG["enlarge_bbox_occ"]
        self.dampen_motion_occ = FAST_TRACKER_CONFIG["dampen_motion_occ"]
        self.active_occ_to_lost_thresh = FAST_TRACKER_CONFIG["active_occ_to_lost_thresh"]
        self.init_iou_suppress = FAST_TRACKER_CONFIG["init_iou_suppress"]

        # Dynamic ROI loading
        self.roi_points = []
        self.theta_values = []
        rois = FAST_TRACKER_CONFIG.get("ROIs", {})

        for name, pts in rois.items():
            try:
                roi_np = np.array(pts)
                self.roi_points.append(roi_np)
                theta = self.compute_theta(roi_np)
                self.theta_values.append(theta)
                print(f"[ROI] {name} loaded with theta = {theta:.2f} degrees.")
            except Exception as e:
                print(f"[Warning] Failed to load {name}: {e}")

        self.roi_repair_max_gap = FAST_TRACKER_CONFIG.get("roi_repair_max_gap", 15)
        self.dir_window_N = FAST_TRACKER_CONFIG.get("dir_window_N", 10)
        self.dir_margin_deg = FAST_TRACKER_CONFIG.get("dir_margin_deg", 2.0)

        # once init, clear all trackid count to avoid large id
        BaseTrack.clear_count()


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
        inds_low = scores > self.args.conf_thresh_low
        inds_high = scores < self.args.conf_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]

        cates = categories[remain_inds]
        cates_second = categories[inds_second]
        
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        """Step 1: Init detections"""

        if len(dets) > 0:
            detections = [Tracklet_w_occluded(tlwh, s, cate, motion=self.motion) for
                            (tlwh, s, cate) in zip(dets, scores_keep, cates)]
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

        ''' Step 2: First association, with high score detection boxes'''
        tracklet_pool = BaseTracker.joint_tracklets(tracked_tracklets, self.lost_tracklets)

        # Predict the current location with Kalman
        for tracklet in tracklet_pool:
            tracklet.predict()

        dists = iou_distance(tracklet_pool, detections)

        # fuse detection conf into iou dist
        if self.args.fuse_detection_score:
            dists = fuse_det_score(dists, detections)
        
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.9)

        for itracked, idet in matches:
            track = tracklet_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

            # tracklets that matched successfully is marked no occlusion
            track.is_occluded = False
            track.not_matched = 0
            track.occluded_len = 0

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [Tracklet(tlwh, s, cate, motion=self.motion) for
                          (tlwh, s, cate) in zip(dets_second, scores_second, cates_second)]
        else:
            detections_second = []

        r_tracked_tracklets = [tracklet_pool[i] for i in u_track if tracklet_pool[i].state == TrackState.Tracked]
        
        dists = iou_distance(r_tracked_tracklets, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_tracklets[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

            # tracklets that matched successfully is marked no occlusion
            track.is_occluded = False
            track.not_matched = 0
            track.occluded_len = 0

        # occlusion handling
        for it in u_track:
            track = r_tracked_tracklets[it]
            track.not_matched += 1

            # Try detecting occlusion
            if not track.is_occluded and track.state == TrackState.Tracked:
                for other in activated_tracklets:
                    if track.track_id == other.track_id:
                        continue
                    if not other.is_activated or other.is_occluded:
                        continue

                    if FastTracker._is_occluded_by(track.tlbr, other.tlbr):
                        track.is_occluded = True
                        track.occluded_len += 1
                        track.last_occluded_frame = self.frame_id
                        track.was_recently_occluded = True

                        # Reset velocity
                        if len(track.mean_history) >= self.reset_velocity_offset_occ:
                            old_mean = track.mean_history[-self.reset_velocity_offset_occ]
                            track.kalman_filter.kf.x[4: 8] = old_mean[4: 8]

                        # Reset position
                        if len(track.mean_history) >= self.reset_pos_offset_occ:
                            old_mean = track.mean_history[-self.reset_pos_offset_occ]
                            track.kalman_filter.kf.x[0: 4] = old_mean[0: 4]

                        # Enlarge once
                        if track.occluded_len == 1:
                            track.kalman_filter.kf.x[3] *= self.enlarge_bbox_occ  # increase height
                            # track.mean[2] = track.mean[2] / track.mean[3]  # adjust aspect ratio

                        # Dampen motion
                        track.kalman_filter.kf.x[4: 8] *= self.dampen_motion_occ
                        break

            # Handle occlusion flags
            if not track.is_occluded:
                track.occluded_len = 0
            else:
                track.occluded_len += 1

            if track.was_recently_occluded and (self.frame_id - track.last_occluded_frame > 40):
                track.was_recently_occluded = False

            # Finally decide whether to mark as lost
            if track.state != TrackState.Lost:
                if track.not_matched > 2 and (
                    not track.is_occluded or track.occluded_len > self.active_occ_to_lost_thresh
                ):
                    track.mark_lost()
                    lost_tracklets.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        
        # fuse detection conf into iou dist
        if self.args.fuse_detection_score:
            dists = fuse_det_score(dists, detections)
       
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_tracklets.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracklets.append(track)

        # After handling Stage-2 matches and updating tracks enforece environment constraints:
        for t in activated_tracklets:
            self.enforce_environment_constraints(t)
        for t in refind_tracklets:
            self.enforce_environment_constraints(t)
        for t in self.tracked_tracklets:
            if t.state == TrackState.Tracked and t not in activated_tracklets and t not in refind_tracklets:
                self.enforce_environment_constraints(t)
        

        """ Step 4: Init new tracklets, with IoU suppression"""
        # Gather active tracks *now* (already-updated ones + still-tracked ones)
        active_now = {
            t.track_id: t 
            for t in self.tracked_tracklets if t.state == TrackState.Tracked
        }

        for t in activated_tracklets:
            active_now[t.track_id] = t
        active_now = list(active_now.values())

        init_iou_thr = getattr(self, "init_iou_suppress", None)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.init_thresh:
                continue

            # compute max IoU with any active track this frame
            det_box = BaseTrack.tlwh_to_tlbr(track.tlwh)
            max_iou = 0.0
            for at in active_now:
                at_box = at.tlbr  # already tlbr
                max_iou = max(max_iou, FastTracker._iou(det_box, at_box))
                if max_iou >= init_iou_thr:
                    break

            # Only initialize if it does NOT heavily overlap an active track
            if max_iou < init_iou_thr:
                track.activate(self.frame_id)
                activated_tracklets.append(track)

        """ Step 5: Update state"""
        for track in self.lost_tracklets:
            recently_occluded = (
                track.was_recently_occluded and
                (self.frame_id - track.last_occluded_frame <= 40)  # configurable if needed
            )

            if not recently_occluded and (self.frame_id - track.end_frame > self.max_time_lost):
                track.mark_removed()
                removed_tracklets.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        self.merge_tracklets(activated_tracklets, refind_tracklets, lost_tracklets, removed_tracklets)

        output_tracklets = [track for track in self.tracked_tracklets if track.is_activated]

        return output_tracklets

    @staticmethod
    def _is_occluded_by(box_a, box_b, iou_thresh=0.7):
        """
        Returns True if box_a is significantly overlapped by box_b
        """
        inter = (
            max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) *
            max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
        )
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        if area_a == 0:
            return False
        iou = inter / area_a
        return iou > iou_thresh

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-9)

    def enforce_environment_constraints(self, t):
        """
        Enforce ROI containment and cone direction for a single track t.
        Assumes t.tlwh exists and t.history (list of tlwh or centers) exists/updated per frame.
        """
        if not self.roi_points:
            return

        # Ensure we have a minimal trajectory history (store centers per frame)
        if not hasattr(t, "center_history"):
            t.center_history = []
        curr_center = self._get_center_from_tlwh(t.tlwh)
        t.center_history.append(curr_center.copy())

        # 1) Determine which ROI contains the current center (if any)
        roi_idx = -1
        for i, roi in enumerate(self.roi_points):
            if len(roi) >= 3 and self._point_in_polygon(curr_center, roi):
                roi_idx = i
                break
        if roi_idx < 0:
            # Not inside any ROI -> no constraint
            return

        roi = self.roi_points[roi_idx]

        # ==========================================================
        # ROI History Repair: if the track is currently inside ROI,
        # but had a short out-of-ROI excursion in the past few frames,
        # project those out-of-bound points back onto the ROI boundary.
        # ==========================================================

        if self._point_in_polygon(curr_center, roi):
            # Track is inside ROI -> check its recent history
            if len(t.center_history) > 2:
                last_inside_idx = None
                last_outside_idx = None

                # Iterate backward through history to find last in/out transitions
                for i in range(len(t.center_history) - 2, -1, -1):
                    pt = t.center_history[i]
                    inside = self._point_in_polygon(pt, roi)

                    if inside and last_outside_idx is not None:
                        # Found transition from outside -> inside
                        last_inside_idx = i
                        break
                    if not inside and last_outside_idx is None:
                        # First time we see an outside segment
                        last_outside_idx = i

                # If we found an out-of-ROI segment that ended recently
                if last_outside_idx is not None and last_inside_idx is not None:
                    gap = last_outside_idx - last_inside_idx
                    if 0 < gap <= self.roi_repair_max_gap:
                        # Repair the short outside segment
                        for j in range(last_inside_idx + 1, last_outside_idx + 1):
                            pt_out = t.center_history[j]
                            # Clamp each point back to the ROI boundary
                            clamped_point = self._clamp_point_to_polygon(pt_out, roi)

                            # Update both geometric and KF mean history
                            t.center_history[j] = clamped_point
                            if hasattr(t, "mean_history") and j < len(t.mean_history):
                                t.mean_history[j][:2] = clamped_point

                        # Update the track’s current center (last frame)
                        curr_center = t.center_history[-1]
                        x, y, w, h = t.tlwh
                        new_x = curr_center[0] - 0.5 * w
                        new_y = curr_center[1] - 0.5 * h
                        t.kalman_filter.kf.x[0: 2] = np.array([new_x, new_y], dtype=float)

                        print(f"[ROI-Repair] Track {t.track_id}: repaired short excursion ({gap} frames).")

        # 2) Direction cone enforcement
        if len(roi) == 4:
            axis_u, theta_deg = self._cone_axis_and_theta(roi)
        else:
            # If not a quad, skip direction constraint
            return

        N = self.dir_window_N
        if len(t.center_history) >= (N + 1):
            pk   = t.center_history[-1]
            pk_N = t.center_history[-1 - N]
            delta = pk - pk_N
            if np.linalg.norm(delta) > 1e-6:
                # Compare phi vs theta/2 -> if violated, rotate last step to boundary
                prev = t.center_history[-2]
                adjusted = self._clamp_to_cone(pk_N, pk, axis_u, theta_deg)
                if not np.allclose(adjusted, pk, atol=1e-3):
                    # Apply adjusted position
                    t.center_history[-1] = adjusted
                    if hasattr(t, "mean_history") and len(t.mean_history) > 0:
                        t.mean_history[-1][:2] = adjusted
                    # Reflect to tlwh (keep size; shift position)
                    x, y, w, h = t.tlwh
                    new_x = adjusted[0] - 0.5*w
                    new_y = adjusted[1] - 0.5*h
                    # t.tlwh = np.array([new_x, new_y, w, h], dtype=float)
                    t.kalman_filter.kf.x[0: 2] = np.array([new_x, new_y], dtype=float)

    @staticmethod
    def compute_theta(roi):
        """
        Computes the opening angle theta of the direction cone from four ROI points.
        ROI assumed to be ordered as [(E1), (E2), (O2), (O1)].
        """
        E1, E2, O2, O1 = roi
        v1 = np.array(O2) - np.array(E1)
        v2 = np.array(O1) - np.array(E2)
        dot = np.dot(v1, v2)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        theta = np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))
        return theta
    
    @staticmethod
    def _get_center_from_tlwh(tlwh):
        x, y, w, h = tlwh
        return np.array([x + 0.5*w, y + 0.5*h], dtype=float)

    @staticmethod
    def _point_in_polygon(pt, poly):
        """Ray casting; poly shape (M,2). Returns True if inside or on boundary."""
        x, y = pt
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # Check intersection with horizontal ray
            cond = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ( (y2 - y1) + 1e-9 ) + x1)
            if cond:
                inside = not inside
        return inside

    @staticmethod
    def _closest_point_on_segment(p, a, b):
        """Project point p to segment ab, return closest point."""
        ap = p - a
        ab = b - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9)
        t = max(0.0, min(1.0, t))
        return a + t * ab

    @classmethod
    def _clamp_point_to_polygon(cls, pt, poly):
        """Clamp point to nearest point on polygon boundary."""
        best = None
        best_d2 = 1e18
        n = len(poly)
        for i in range(n):
            a = poly[i].astype(float)
            b = poly[(i + 1) % n].astype(float)
            q = cls._closest_point_on_segment(pt, a, b)
            d2 = np.sum((q - pt)**2)
            if d2 < best_d2:
                best_d2 = d2
                best = q
        return best if best is not None else pt

    @staticmethod
    def _normalize(v):
        n = np.linalg.norm(v)
        return v / (n + 1e-9)

    @staticmethod
    def _angle_of(vec):
        """Angle of vector in radians (−pi, pi]."""
        return math.atan2(vec[1], vec[0])

    @staticmethod
    def _angle_diff(a, b):
        """Smallest signed angle a−b in radians (−pi, pi]."""
        d = (a - b + math.pi) % (2*math.pi) - math.pi
        return d

    @staticmethod
    def _cone_axis_and_theta(roi):
        """From four ROI points [(E1),(E2),(O2),(O1)] get cone axis unit vector and theta (degrees)."""
        E1, E2, O2, O1 = roi
        v1 = FastTracker._normalize(np.array(O2) - np.array(E1))
        v2 = FastTracker._normalize(np.array(O1) - np.array(E2))
        axis = FastTracker._normalize(v1 + v2)  # average direction
        dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        theta = math.degrees(math.acos(dot))    # opening angle
        return axis, theta

    def _clamp_to_cone(self, anchor_pt, curr_pt, axis_u, theta_deg):
        """
        Enforce a direction cone centered on 'axis_u' with opening angle 'theta_deg'.
        If the displacement delta = curr_pt - anchor_pt deviates beyond theta/2 from axis_u,
        clamp delta to the nearest cone boundary while preserving its magnitude.
        Returns the adjusted current point.
        """
        delta = np.asarray(curr_pt, dtype=float) - np.asarray(anchor_pt, dtype=float)
        mag = np.linalg.norm(delta)
        if mag < 3.0: # configurable
            return curr_pt  # no displacement, nothing to adjust

        # Normalize vectors
        delta_u = delta / mag
        axis_u  = self._normalize(np.asarray(axis_u, dtype=float))

        # Angle between delta and axis_u (0..pi)
        cosang = float(np.clip(np.dot(delta_u, axis_u), -1.0, 1.0))
        ang = math.acos(cosang)
        half = math.radians(theta_deg) * 0.5

        if ang <= half:
            # Already within cone
            return curr_pt

        # Determine which side to clamp to (sign via 2D cross product z-component)
        cross_z = axis_u[0] * delta_u[1] - axis_u[1] * delta_u[0]
        sign = 1.0 if cross_z > 0 else -1.0  # +half on left side, -half on right side

        # Build the boundary direction by rotating axis_u by -+half
        c, s = math.cos(sign * half), math.sin(sign * half)

        # Rotation matrix R(theta/2) = [ [c -s], [s  c] ]
        # rotate axis_u by this matrix to get boundary direction
        boundary_dir = np.array([axis_u[0]*c - axis_u[1]*s, axis_u[0]*s + axis_u[1]*c], dtype=float)

        # Preserve the magnitude of delta
        # New point = pk_N + boundary_dir * |mag|
        clamped = np.asarray(anchor_pt, dtype=float) + boundary_dir * mag
        return clamped