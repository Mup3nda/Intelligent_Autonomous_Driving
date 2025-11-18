import os
import numpy as np
import cv2
from ultralytics import YOLO
from calibration import load_camera_calibration, setup_stereo_camera
from data_loader import load_sequence
from detection import detect_objects_in_frame
from DepthHandling import compute_depth_map, bbox_depth, depth_to_3d,depth_to_vis
from featureExtraction import extract_roi_features
from Tracker import SimpleTracker3D
from visualization import save_trajectories_csv, show_trajectories_topdown,draw_tracks_and_trajectories


def process_sequence(seq_path, stereo_params, detection_model, class_names):
    left_imgs, right_imgs, labels = load_sequence(seq_path)

    K_left = stereo_params['K_left']
    tracker = SimpleTracker3D(K_left,
                              max_disappeared=45,
                              max_distance_3d=5.0,   # tune
                              max_distance_2d=15.0,
                              history_len=10)  # tune
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500,
                                                       varThreshold=16,
                                                       detectShadows=True  # shadows will be marked as 127
                                                       )
    for left_name, right_name in zip(left_imgs, right_imgs):
        left_path = os.path.join(seq_path, 'left', 'data', left_name)
        right_path = os.path.join(seq_path, 'right', 'data', right_name)

        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        if left_img is None or right_img is None:
            continue
        gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        fg_mask_raw = bg_subtractor.apply(gray)  # 0=bg, 255=fg, 127=shadow

        # remove shadows (127) and clean up
        _, fg_mask_bin = cv2.threshold(fg_mask_raw, 200, 255, cv2.THRESH_BINARY)
        fg_mask_bin = cv2.morphologyEx(fg_mask_bin, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        # 1) detection (2D)
        det2d = detect_objects_in_frame(left_img, detection_model,conf_thres=0.5, max_detections=20)

        # 2) depth
        depth_map, disparity = compute_depth_map(left_img, right_img, stereo_params)

        # 3) enrich detections with 3D positions
        det3d = []
        for bbox, cls, conf in det2d:
            d = bbox_depth(bbox, depth_map, radius=5)
            if d is None:
                pos3d = None
            else:
                cx, cy = tracker._bbox_centroid(bbox)
                pos3d = depth_to_3d(cx, cy, d, K_left)
            # extract appearance features
            feat = extract_roi_features(left_img, bbox, fg_mask=fg_mask_bin)
            
            det3d.append((bbox, cls, conf, pos3d, feat))

        # 4) update tracker
        H, W = left_img.shape[:2]
        tracker.update(det3d, frame_shape=(H, W))

        # 5) visualize
        tracks2d_hist = tracker.get_tracks_with_history2d()
        tracked_img = draw_tracks_and_trajectories(left_img.copy(), tracks2d_hist, class_names)
        cv2.imshow("Tracking", tracked_img)

        # optional: disparity/depth debug
        depth_color = depth_to_vis(depth_map)
        cv2.imshow("Depth Map", depth_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return tracker


# Load calibration
calib_path = r"c:\Program Files\StartJupyter\PFAS\Final Project\Rectified Images\calib_cam_to_cam.txt"
calibration = load_camera_calibration(calib_path)

print("Calibration keys loaded:")
for key in sorted(calibration.keys()):
    print(f"  {key}: shape {calibration[key].shape}")

# Setup stereo pair (cameras 02 left, 03 right)
stereo = setup_stereo_camera(calibration, left_id='02', right_id='03')

print("\n--- Stereo Camera (02 left, 03 right) ---")
print("K_left:\n", stereo['K_left'])
print("\nK_right:\n", stereo['K_right'])
print("\nR (rotation):\n", stereo['R'])
print("\nT (translation):\n", stereo['T'])
print("\nBaseline (distance):", np.linalg.norm(stereo['T']), "m")


seq_01_path = r"c:\Program Files\StartJupyter\PFAS\Final Project\Rectified Images\seq_02"
left_imgs, right_imgs, labels = load_sequence(seq_01_path)

print(f"Loaded {len(left_imgs)} left images, {len(right_imgs)} right images")
print(f"Labels: {set(labels)}")

detection_model = YOLO("C:/Program Files/StartJupyter/PFAS/Final Project/yolo11s.pt")
class_names = {0: "person", 1: "bicycle", 2: "car"}

tracker = process_sequence(seq_01_path, stereo, detection_model, class_names)
print("Finished tracking. Trajectories:", {tid: len(t) for tid, t in tracker.trajectories.items()})
save_trajectories_csv(tracker, "C:/Program Files/StartJupyter/PFAS/Final Project/trajectories_3d.csv")
show_trajectories_topdown(tracker)