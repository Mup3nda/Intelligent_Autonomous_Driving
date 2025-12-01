import numpy as np
import math
from collections import deque
import cv2
class CVKalman2D:
    """
    Constant-velocity Kalman filter on image plane using cv2.KalmanFilter.
    State: [u, v, vu, vv]^T (px, px, px/frame, px/frame)
    Measurement: [u, v]^T
    """
    def __init__(self, u0, v0, dt=1.0, process_var=1.0, meas_var=200.0):
        self.kf = cv2.KalmanFilter(4, 2)

        # F: state transition matrix
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], np.float32)

        # H: measurement matrix (we only observe position)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        # Q: process noise
        self.kf.processNoiseCov = (process_var *
                                   np.eye(4, dtype=np.float32))

        # R: measurement noise
        self.kf.measurementNoiseCov = (meas_var *
                                       np.eye(2, dtype=np.float32))

        # P: initial error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0

        # Initial state
        self.kf.statePost = np.array([[u0], [v0], [0.0], [0.0]],
                                     dtype=np.float32)

    def predict(self):
        pred = self.kf.predict()
        return pred  # 4x1

    def update(self, u, v):
        meas = np.array([[u], [v]], dtype=np.float32)
        est = self.kf.correct(meas)
        return est  # 4x1

    @property
    def pos(self):
        # current best estimate of (u, v)
        state = self.kf.statePost
        return float(state[0, 0]), float(state[1, 0])

    @property
    def vel(self):
        state = self.kf.statePost
        return float(state[2, 0]), float(state[3, 0])


class CVKalman3D:
    """
    Constant-velocity Kalman filter in 3D camera space.
    State: [X, Y, Z, Vx, Vy, Vz]^T
    Measurement: [X, Y, Z]^T (from stereo/depth)
    """
    def __init__(self, X0, Y0, Z0, dt=1.0,
                 process_var=1.0, meas_var=300):
        self.kf = cv2.KalmanFilter(6, 3)

        # F: state transition
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ], np.float32)

        # H: we observe positions only
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], np.float32)

        self.kf.processNoiseCov = (process_var *
                                   np.eye(6, dtype=np.float32))
        self.kf.measurementNoiseCov = (meas_var *
                                       np.eye(3, dtype=np.float32))
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 100.0

        # Initial state
        self.kf.statePost = np.array(
            [[X0], [Y0], [Z0], [0.0], [0.0], [0.0]],
            dtype=np.float32
        )

    def predict(self):
        pred = self.kf.predict()
        return pred  # 6x1

    def update(self, X, Y, Z):
        meas = np.array([[X], [Y], [Z]], dtype=np.float32)
        est = self.kf.correct(meas)
        return est

    @property
    def pos(self):
        state = self.kf.statePost
        return (float(state[0, 0]),
                float(state[1, 0]),
                float(state[2, 0]))

    @property
    def vel(self):
        state = self.kf.statePost
        return (float(state[3, 0]),
                float(state[4, 0]),
                float(state[5, 0]))
    
class SimpleTracker3D:
    def __init__(self, K,
                 max_disappeared=30,
                 max_distance_3d=5.0,    # meters
                 max_distance_2d=30.0,   # pixels (fallback)
                 history_len=10):
        """
        K: 3x3 camera intrinsics for projection.
        """
        self.tracks = {}        # id -> track dict
        self.next_id = 0
        self.trajectories = {}  # track_id -> list of 3D points

        self.max_disappeared = max_disappeared
        self.max_distance_3d = max_distance_3d
        self.max_distance_2d = max_distance_2d
        self.history_len = history_len

        self.fx = K[0, 0]; self.fy = K[1, 1]
        self.cx = K[0, 2]; self.cy = K[1, 2]
        
        self.min_speed_for_valid = 1  # m/frame *and* px/frame
        self.hallucinated_frac = 0.3     # 0.3 * max_disappeared
        self.history = {}
    # ---------- helpers ----------
    def _clamp_vec(self, v, max_norm):
        n = float(np.linalg.norm(v))
        if n > max_norm and n > 1e-6:
            v = v * (max_norm / n)
        return v
    def _bbox_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    
    def _is_near_edge(self, point, frame_shape, margin=40):
        if frame_shape is None:
            return False
        H, W = frame_shape
        x, y = point
        return (x < margin or x > W - margin or
                y < margin or y > H - margin)
    def _cleanup_duplicate_tracks(self, iou_threshold=0.7, motion_threshold=0.5):
        """
        Remove duplicate tracks that overlap heavily and aren't moving.

        iou_threshold:     how similar bboxes must be (0.0–1.0)
        motion_threshold:  max allowed speed (px/frame AND m/frame)
        """
        tids = list(self.tracks.keys())
        to_delete = set()

        def iou(b1, b2):
            x1,y1,x2,y2 = b1
            x1b,y1b,x2b,y2b = b2
            inter_x1 = max(x1, x1b)
            inter_y1 = max(y1, y1b)
            inter_x2 = min(x2, x2b)
            inter_y2 = min(y2, y2b)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return 0.0

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x2b - x1b) * (y2b - y1b)
            return inter_area / float(area1 + area2 - inter_area)

        for i in range(len(tids)):
            if tids[i] in to_delete: continue
            trA = self.tracks[tids[i]]

            for j in range(i+1, len(tids)):
                if tids[j] in to_delete: continue
                trB = self.tracks[tids[j]]

                # same class?
                if trA['class_id'] != trB['class_id']:
                    continue

                # bounding box overlap?
                ov = iou(trA['bbox'], trB['bbox'])
                if ov < iou_threshold:
                    continue

                # both barely moving?
                v2A = np.linalg.norm(trA.get('vel2d', np.zeros(2)))
                v2B = np.linalg.norm(trB.get('vel2d', np.zeros(2)))
                v3A = np.linalg.norm(trA.get('vel3d', np.zeros(3)))
                v3B = np.linalg.norm(trB.get('vel3d', np.zeros(3)))

                if (v2A < motion_threshold and v2B < motion_threshold and
                    v3A < motion_threshold and v3B < motion_threshold):

                    # delete whichever one is the *newer* track
                    older = tids[i]
                    newer = tids[j]
                    # keep oldest, delete youngest
                    to_delete.add(newer)

        for tid in to_delete:
            if tid in self.tracks:
                del self.tracks[tid]
    def _is_likely_hallucination(self, tr):
        """
        Decide if a track is probably a spurious detection that we
        can safely delete early.

        Heuristic:
        - young track (age < max_disappeared)
        - has been occluded for most of its life
        - has very few visible frames
        - basically not moving in 2D and 3D
        """
        age = tr.get('age', 0)
        if age <= 0:
            return False

        disappeared = tr.get('disappeared', 0)
        visible = tr.get('visible_count', 0)
        v3 = tr.get('vel3d', np.zeros(3, dtype=np.float32))
        v2 = tr.get('vel2d', np.zeros(2, dtype=np.float32))

        # fraction of life spent occluded
        frac_occluded = disappeared / float(age)

        # only kill *young* tracks
        if age >= self.max_disappeared:
            return False

        # must be occluded most of their life
        if frac_occluded < 0.4:       # e.g. >60% of life occluded
            return False

        # must have been barely visible
        if visible > 5:               # seen clearly more than a couple frames -> keep
            return False

        # must be almost not moving
        #if (np.linalg.norm(v3) >= self.min_speed_for_valid or
        #    np.linalg.norm(v2) >= self.min_speed_for_valid*2):
        #    return False

        return True
    
    def _has_exited_frame(self, tr, frame_shape, margin=20):
        """
        Heuristic: if the last bbox center is near the border and the track
        is disappearing, we assume they left the frame.
        """
        if frame_shape is None:
            return False

        if tr.get('disappeared', 0) == 0:
            return False

        H, W = frame_shape
        x1, y1, x2, y2 = tr['bbox']
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        near_edge = (
            cx < margin or cx > W - margin or
            cy < margin or cy > H - margin
        )
        return near_edge

    # ---------- main update ----------

    def update(self, detections, frame_shape=None,frame_idx=None):
        """
        detections: list of (bbox, class_id, confidence, pos3d, feat)
        pos3d: np.array([X,Y,Z]) in meters, or None if depth invalid.
        """
        self.frame_shape = frame_shape
        # increment age for all existing tracks
        for tr in self.tracks.values():
            tr['age'] = tr.get('age', 0) + 1
        # Kalman prediction for all tracks
        for tid, tr in self.tracks.items():
            
            kf2d = tr.get('kf2d', None)
            if kf2d is not None:
                kf2d.predict()
                tr['pred_center'] = kf2d.pos  # (u_pred, v_pred)
            else:
                tr['pred_center'] = self._bbox_centroid(tr['bbox'])

            kf3d = tr.get('kf3d', None)
            if kf3d is not None:
                kf3d.predict()
                tr['pos3d_pred'] = np.array(kf3d.pos, dtype=np.float32)
            else:
                tr['pos3d_pred'] = tr['pos3d']

        # 1) No existing tracks -> start one per detection
        if len(self.tracks) == 0:
            for bbox, cls, conf, pos3d, feat in detections:
                if pos3d is None:
                    pos3d = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

                tid = self.next_id
                cx, cy = self._bbox_centroid(bbox)

                kf2d = CVKalman2D(cx, cy)
                kf3d = None
                if pos3d is not None and np.all(np.isfinite(pos3d)):
                    X0, Y0, Z0 = float(pos3d[0]), float(pos3d[1]), float(pos3d[2])
                    kf3d = CVKalman3D(X0, Y0, Z0)

                self.tracks[tid] = {
                    'bbox': bbox,
                    'class_id': cls,
                    'conf': conf,
                    'disappeared': 0,
                    'pos3d': pos3d,
                    'vel3d': np.zeros(3, dtype=np.float32),
                    'history3d': deque([pos3d], maxlen=self.history_len),
                    'history2d': deque([(cx, cy)], maxlen=self.history_len),
                    'vel2d': np.zeros(2, dtype=np.float32),
                    'kf2d': kf2d,        # 2D Kalman filter
                    'kf3d': kf3d,        # 3D Kalman filter 
                    'feat': feat,  # current appearance
                    'age': 1,             # new track
                    'visible_count': 1,   # visible this frame
                    #error tracking
                    'reacquired_this_frame': False,
                    'last_reacq_jump2d': None,
                    'last_reacq_jump3d': None,

                }
                self.trajectories[tid] = [pos3d.copy()]
                self.next_id += 1
            return

        # 2) No detections -> predict forward and mark disappeared
        if len(detections) == 0:
            ids_to_delete = []
            for tid, tr in self.tracks.items():
                tr['disappeared'] += 1
                
                if self._is_likely_hallucination(tr):
                    ids_to_delete.append(tid)
                    continue  # skip prediction
                 # Early kill if they clearly exited at the edge
                if self._has_exited_frame(tr, frame_shape):
                    ids_to_delete.append(tid)
                    continue
                kf2d = tr.get('kf2d', None)
                if kf2d is not None:
                    u_pred, v_pred = tr.get('pred_center', kf2d.pos)
                    x1, y1, x2, y2 = tr['bbox']
                    w = x2 - x1
                    h = y2 - y1
                    tr['bbox'] = (
                        int(u_pred - w / 2), int(v_pred - h / 2),
                        int(u_pred + w / 2), int(v_pred + h / 2)
                    )

                kf3d = tr.get('kf3d', None)
                if kf3d is not None:
                    # we've already called predict() at the top, so just copy
                    tr['pos3d'] = np.array(tr['pos3d_pred'], dtype=np.float32)
                    tr['vel3d'] = np.array(kf3d.vel, dtype=np.float32)
                if tr['disappeared'] > self.max_disappeared:
                    ids_to_delete.append(tid)

            for tid in ids_to_delete:
                del self.tracks[tid]
            return

        # 3) We have tracks and detections -> match

        track_ids = list(self.tracks.keys())
        track_centroids_2d = [
                                self.tracks[tid].get('pred_center',
                                                    self._bbox_centroid(self.tracks[tid]['bbox']))
                                for tid in track_ids
                             ]
        det_centroids_2d = [self._bbox_centroid(d[0]) for d in detections]

        used_dets = set()

        for i, tid in enumerate(track_ids):
            tr = self.tracks[tid]
            cx_t, cy_t = track_centroids_2d[i]
            cls_t = tr['class_id']
            pos3d_t = tr.get('pos3d_pred', tr['pos3d'])
            feat_t = tr.get('feat', None)

            is_occluded = tr['disappeared'] > 0

            best_j = None
            best_cost = float('inf')

            for j, det in enumerate(detections):
                if j in used_dets:
                    continue

                bbox_d, cls_d, conf_d, pos3d_d, feat_d = det

                # class gating
                if cls_d != cls_t:
                    continue

                # figure out what data we have
                track_has_depth = pos3d_t is not None and np.all(np.isfinite(pos3d_t))
                det_has_depth   = pos3d_d is not None and np.all(np.isfinite(pos3d_d))

                use_3d = track_has_depth and det_has_depth

                # ---------- GEOMETRIC COST ----------
                if use_3d:
                    # 3D distance
                    if is_occluded:
                        local_max_3d = self.max_distance_3d * 0.5  # stricter when occluded
                    else:
                        local_max_3d = self.max_distance_3d

                    d3 = np.linalg.norm(pos3d_t - pos3d_d)
                    if d3 > local_max_3d:
                        continue
                    geom_cost = d3 / local_max_3d

                else:
                    # ---------- 2D fallback matching ----------
                    cx_d, cy_d = det_centroids_2d[j]

                    # Velocity-aware predicted center
                    v2d = tr.get('vel2d', np.zeros(2, dtype=np.float32))
                    cx_pred, cy_pred = self.tracks[tid].get('pred_center', track_centroids_2d[i])
                    d2 = math.hypot(cx_pred - cx_d, cy_pred - cy_d)

                    # Dynamic distance gate based on motion speed
                    speed = float(np.linalg.norm(v2d))

                    if is_occluded:
                        # near the frame edge we relax the gate
                        if frame_shape is not None:
                            H, W = frame_shape
                            near_edge = (
                                cx_t < 40 or cx_t > W - 40 or
                                cy_t < 40 or cy_t > H - 40
                            )
                        else:
                            near_edge = False

                        if near_edge:
                            base_2d = self.max_distance_2d * 0.5   # stricter at edges when occluded
                        else:
                            base_2d = self.max_distance_2d * 0.7   # slightly stricter overall

                    else:
                        base_2d = self.max_distance_2d

                    # dynamic threshold grows with speed
                    local_max_2d = base_2d + 0.4 * speed

                    if d2 > local_max_2d:
                        continue

                    geom_cost = d2 / local_max_2d

                # ---------- APPEARANCE COST ----------
                app_cost = 0.5  # neutral default
                cos_sim = None

                if feat_t is not None and feat_d is not None:
                    color_t = feat_t.get('color', None)
                    color_d = feat_d.get('color', None)
                    orb_t   = feat_t.get('orb', None)
                    orb_d   = feat_d.get('orb', None)

                    sims = []

                    # color similarity
                    if color_t is not None and color_d is not None:
                        ft_c = color_t / (np.linalg.norm(color_t) + 1e-6)
                        fd_c = color_d / (np.linalg.norm(color_d) + 1e-6)
                        cos_c = float(np.dot(ft_c, fd_c))
                        sims.append(cos_c)

                    # ORB similarity (treat mean descriptor as float vector)
                    if orb_t is not None and orb_d is not None:
                        ft_o = orb_t / (np.linalg.norm(orb_t) + 1e-6)
                        fd_o = orb_d / (np.linalg.norm(orb_d) + 1e-6)
                        cos_o = float(np.dot(ft_o, fd_o))
                        sims.append(cos_o)

                    if sims:
                        cos_sim = sum(sims) / len(sims)
                        cos_sim = max(-1.0, min(1.0, cos_sim))
                        app_cost = 0.5 * (1.0 - cos_sim)
                    else:
                        cos_sim = None

                # for occluded tracks, require stronger appearance agreement if we have cos_sim
                if is_occluded:
                    if cos_sim is None:
                        # no appearance → don't trust
                        continue
                    # relax requirement near the border (ROIs are cropped & noisy)
                    cx_t, cy_t = track_centroids_2d[i]
                    near_edge = self._is_near_edge((cx_t, cy_t), frame_shape)

                    if near_edge:
                        min_cos = 0.6   # need pretty good match at edge
                    else:
                        min_cos = 0.5 # moderate match needed

                    if cos_sim < min_cos:
                        continue

                # ---------- COMBINE ----------
                alpha = 0.7  # geometry weight
                total_cost = alpha * geom_cost + (1 - alpha) * app_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_j = j

            if best_j is not None:
                # matched
                bbox_d, cls_d, conf_d, pos3d_d, feat_d = detections[best_j]
                prev_disappeared = tr['disappeared']
                # predicted center before correction (from Kalman)
                u_pred, v_pred = tr.get('pred_center',
                                        self._bbox_centroid(tr['bbox']))
                cx_d, cy_d = self._bbox_centroid(bbox_d)

                # predicted 3D position before correction (if available)
                pos3d_pred_before = tr.get('pos3d_pred', tr.get('pos3d', None))

                # --- reacquisition metrics (must be BEFORE KF update) ---
                if prev_disappeared > 0:
                    # track was occluded, now reacquired
                    jump2d = math.hypot(cx_d - u_pred, cy_d - v_pred)
                    tr['last_reacq_jump2d'] = jump2d

                    if pos3d_d is not None and np.all(np.isfinite(pos3d_d)) \
                    and pos3d_pred_before is not None and np.all(np.isfinite(pos3d_pred_before)):
                        jump3d = float(np.linalg.norm(pos3d_d - pos3d_pred_before))
                        tr['last_reacq_jump3d'] = jump3d
                    else:
                        tr['last_reacq_jump3d'] = None

                    tr['reacquired_this_frame'] = True
                else:
                    tr['reacquired_this_frame'] = False
                    tr['last_reacq_jump2d'] = tr.get('last_reacq_jump2d', None)
                    tr['last_reacq_jump3d'] = tr.get('last_reacq_jump3d', None)
                # --- 2D update ---
                tr['bbox'] = bbox_d
                tr['class_id'] = cls_d
                tr['conf'] = conf_d
                tr['disappeared'] = 0
                tr['visible_count'] = tr.get('visible_count', 0) + 1
                # 2D KF update
                kf2d = tr.get('kf2d', None)
                if kf2d is None:
                    kf2d = CVKalman2D(cx_d, cy_d)
                    tr['kf2d'] = kf2d
                
                kf2d.update(cx_d, cy_d)
                tr['pred_center'] = kf2d.pos
                if 'history2d' not in tr:
                    tr['history2d'] = deque(maxlen=self.history_len)
                tr['history2d'].append((kf2d.pos))
                vx, vy = kf2d.vel
                tr['vel2d'] = np.array([vx, vy], dtype=np.float32)

                # --- 3D update ---
                # 3D KF update (if we have usable depth)
                if pos3d_d is not None and np.all(np.isfinite(pos3d_d)):
                    Xd, Yd, Zd = float(pos3d_d[0]), float(pos3d_d[1]), float(pos3d_d[2])
                    kf3d = tr.get('kf3d', None)
                    if kf3d is None:
                        kf3d = CVKalman3D(Xd, Yd, Zd)
                        tr['kf3d'] = kf3d
                    kf3d.update(Xd, Yd, Zd)
                    tr['pos3d'] = np.array(kf3d.pos, dtype=np.float32)
                    tr['pos3d_pred'] = tr['pos3d']
                    tr['vel3d'] = np.array(kf3d.vel, dtype=np.float32)
                else:
                    # if no new depth, keep previous pos3d / pos3d_pred as is
                    pass
                # update 3D history
                if pos3d_d is not None and np.all(np.isfinite(pos3d_d)):
                    tr['history3d'].append(kf3d.pos)

                    if tid not in self.trajectories:
                        self.trajectories[tid] = []
                    self.trajectories[tid].append(kf3d.pos)
                # update appearance with EMA
                if feat_d is not None:
                    feat_old = tr.get('feat', None)
                    if feat_old is None:
                        tr['feat'] = feat_d
                    else:
                        beta = 0.8
                        new_feat = {}

                        # smooth color histogram
                        col_old = feat_old.get('color', None)
                        col_new = feat_d.get('color', None)
                        if col_old is not None and col_new is not None:
                            new_feat['color'] = beta * col_old + (1.0 - beta) * col_new
                        elif col_new is not None:
                            new_feat['color'] = col_new
                        else:
                            new_feat['color'] = col_old

                        # for ORB, just take the latest descriptor if available
                        orb_new = feat_d.get('orb', None)
                        orb_old = feat_old.get('orb', None)
                        new_feat['orb'] = orb_new if orb_new is not None else orb_old

                        tr['feat'] = new_feat

                used_dets.add(best_j)

            else:
                # unmatched track -> predict forward
                tr['disappeared'] += 1
                kf2d = tr.get('kf2d', None)
                if kf2d is not None:
                    u_pred, v_pred = tr.get('pred_center', kf2d.pos)
                    x1, y1, x2, y2 = tr['bbox']
                    w = x2 - x1
                    h = y2 - y1
                    tr['bbox'] = (
                        int(u_pred - w / 2), int(v_pred - h / 2),
                        int(u_pred + w / 2), int(v_pred + h / 2)
                    )

                kf3d = tr.get('kf3d', None)
                if kf3d is not None:
                    # we've already called predict() at the top, so just copy
                    tr['pos3d'] = np.array(tr['pos3d_pred'], dtype=np.float32)
                    tr['vel3d'] = np.array(kf3d.vel, dtype=np.float32)
                # --- hallucination early-kill ---
                if self._is_likely_hallucination(tr):
                    # we are looping over a *copy* of keys (track_ids),
                    # so deleting directly is safe here
                    if tid in self.tracks:
                        del self.tracks[tid]
                    continue  # skip prediction
        # end of matching loop

        # 4) remove long-disappeared tracks
        ids_to_delete = []
        for tid, tr in self.tracks.items():
            if tr['disappeared'] > self.max_disappeared:
                ids_to_delete.append(tid)
            elif self._has_exited_frame(tr, frame_shape):
                ids_to_delete.append(tid)
        for tid in ids_to_delete:
            del self.tracks[tid]

        # 5) create new tracks for unused detections
        for j, det in enumerate(detections):
            if j in used_dets:
                continue
            bbox_d, cls_d, conf_d, pos3d_d, feat_d = det
            if pos3d_d is None:
                pos3d_d = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

            tid = self.next_id
            cx_d, cy_d = self._bbox_centroid(bbox_d)
            kf2d = CVKalman2D(cx_d, cy_d)
            kf3d = None
            if pos3d_d is not None and np.all(np.isfinite(pos3d_d)):
                X0, Y0, Z0 = float(pos3d_d[0]), float(pos3d_d[1]), float(pos3d_d[2])
                kf3d = CVKalman3D(X0, Y0, Z0)

            self.tracks[tid] = {
                'bbox': bbox_d,
                'class_id': cls_d,
                'conf': conf_d,
                'disappeared': 0,
                'pos3d': pos3d_d,
                'vel3d': np.zeros(3, dtype=np.float32),
                'history3d': deque([pos3d_d], maxlen=self.history_len),
                'history2d': deque([(cx_d, cy_d)], maxlen=self.history_len),
                'vel2d': np.zeros(2, dtype=np.float32),
                'feat': feat_d,
                'age': 1,
                'visible_count': 1,
                'kf2d': kf2d,
                'kf3d': kf3d,
                'reacquired_this_frame': False,
                'last_reacq_jump2d': None,
                'last_reacq_jump3d': None,
            }
            self.trajectories[tid] = [pos3d_d.copy()]
            self.next_id += 1
        # 6) optional: cleanup duplicate tracks
        self._cleanup_duplicate_tracks()
        # 7) LOGGING: snapshot state for this frame
        if frame_idx is not None:
            for tid, tr in self.tracks.items():
                pos3d = tr['pos3d']
                if pos3d is not None and np.all(np.isfinite(pos3d)):
                    pos3d_log = pos3d.copy()
                else:
                    pos3d_log = None

                center = self._bbox_centroid(tr['bbox'])
                rec = {
                    'frame': frame_idx,
                    'visible': (tr['disappeared'] == 0),
                    'disappeared': tr['disappeared'],
                    'center': center,
                    'pos3d': pos3d_log,
                    'reacquired': tr.get('reacquired_this_frame', False),
                    'jump2d': tr.get('last_reacq_jump2d', None),
                    'jump3d': tr.get('last_reacq_jump3d', None),
                }

                self.history.setdefault(tid, []).append(rec)

                # reset per-frame flag
                tr['reacquired_this_frame'] = False

    # ---------- public accessors ----------

    def get_tracks(self):
        """
        For 2D drawing: returns {track_id: (bbox, class_id, conf)}.
        """
        out = {}
        for tid, tr in self.tracks.items():
            if tr['disappeared'] <= self.max_disappeared:
                out[tid] = (tr['bbox'], tr['class_id'], tr['conf'])
        return out

    def get_tracks_3d(self):
        """
        Optional: returns {track_id: (pos3d, vel3d, class_id, conf)}.
        """
        out = {}
        for tid, tr in self.tracks.items():
            if tr['disappeared'] <= self.max_disappeared:
                out[tid] = (tr['pos3d'], tr['vel3d'], tr['class_id'], tr['conf'])
        return out
    def get_tracks_with_history2d(self):
        """
        Returns {track_id: (bbox, class_id, conf, history2d_list)}
        where history2d_list is a list of (x, y) points.
        """
        out = {}
        for tid, tr in self.tracks.items():
            if tr['disappeared'] <= self.max_disappeared:
                hist2d = list(tr.get('history2d', []))
                out[tid] = (tr['bbox'], tr['class_id'], tr['conf'], hist2d)
        return out
    
    def get_history(self):
        return self.history