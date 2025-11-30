import cv2
import os
import numpy as np
from calibration import load_camera_calibration, setup_stereo_camera, rectify_stereo_pair
from DepthHandling import compute_depth_map, depth_to_vis
# Load calibration files
custom_calib_path = r"c:\Program Files\StartJupyter\PFAS\Final Project\Rectified Images\calib_custom.txt"
gt_calib_path     = r"c:\Program Files\StartJupyter\PFAS\Final Project\Rectified Images\calib_cam_to_cam.txt"

custom_calib = load_camera_calibration(custom_calib_path)
gt_calib     = load_camera_calibration(gt_calib_path)

# Set up stereo rigs
stereo_custom = setup_stereo_camera(custom_calib, left_id='00', right_id='01')
stereo_gt     = setup_stereo_camera(gt_calib,     left_id='02', right_id='03')
print("\n--- Ground Truth Stereo Camera (02 left, 03 right) ---")
print("K_left:\n", stereo_gt['K_left'])
print("\nK_right:\n", stereo_gt['K_right'])
print("\nR (rotation):\n", stereo_gt['R'])
print("\nT (translation):\n", stereo_gt['T'])
print("\nBaseline (distance):", np.linalg.norm(stereo_gt['T']), "m")

print("\n--- Custom Stereo Camera (00 left, 01 right) ---")
print("K_left:\n", stereo_custom['K_left'])
print("\nK_right:\n", stereo_custom['K_right'])
print("\nR (rotation):\n", stereo_custom['R'])
print("\nT (translation):\n", stereo_custom['T'])
print("\nBaseline (distance):", np.linalg.norm(stereo_custom['T']), "m")
# Load one raw stereo frame
imgL = cv2.imread(r"C:\Program Files\StartJupyter\PFAS\Final Project\34759_final_project_raw\seq_01\image_02\data\0000000000.png")
imgR = cv2.imread(r"C:\Program Files\StartJupyter\PFAS\Final Project\34759_final_project_raw\seq_01\image_03\data\0000000000.png")
assert imgL is not None and imgR is not None, "Failed to load stereo images"

# output folder
out_dir = r"c:\Program Files\StartJupyter\PFAS\Final Project\calib_comparison"
os.makedirs(out_dir, exist_ok=True)

# -------------------------
# 3. Rectify using both calibrations
# -------------------------
rectL_gt, rectR_gt = rectify_stereo_pair(imgL, imgR, stereo_gt)
rectL_cu, rectR_cu = rectify_stereo_pair(imgL, imgR, stereo_custom)

cv2.imwrite(os.path.join(out_dir, "rect_gt_left.png"), rectL_gt)
cv2.imwrite(os.path.join(out_dir, "rect_gt_right.png"), rectR_gt)
cv2.imwrite(os.path.join(out_dir, "rect_custom_left.png"), rectL_cu)
cv2.imwrite(os.path.join(out_dir, "rect_custom_right.png"), rectR_cu)

# Optional: side-by-side for the report
stack_gt    = np.hstack([rectL_gt, rectR_gt])
stack_custom = np.hstack([rectL_cu, rectR_cu])
cv2.imwrite(os.path.join(out_dir, "rect_gt_side_by_side.png"), stack_gt)
cv2.imwrite(os.path.join(out_dir, "rect_custom_side_by_side.png"), stack_custom)

# -------------------------
# 4. Depth computation
# -------------------------

depth_gt,disparity_gt = compute_depth_map(rectL_gt, rectR_gt, stereo_gt)
depth_cu,disparity_cu = compute_depth_map(rectL_cu, rectR_cu, stereo_custom)

# Ensure float32 for math later
depth_gt = depth_gt.astype(np.float32)
depth_cu = depth_cu.astype(np.float32)

# -------------------------
# 5. Visualize depth (for figures)
# -------------------------
depth_gt_vis = depth_to_vis(depth_gt)      # your function → uint8 BGR/GRAY
depth_cu_vis = depth_to_vis(depth_cu)

cv2.imwrite(os.path.join(out_dir, "depth_gt_vis.png"), depth_gt_vis)
cv2.imwrite(os.path.join(out_dir, "depth_custom_vis.png"), depth_cu_vis)

# -------------------------
# 6. Depth difference + metrics
# -------------------------
# Valid mask: both have positive / finite depth
valid = (depth_gt > 0) & (depth_cu > 0) & np.isfinite(depth_gt) & np.isfinite(depth_cu)

if np.count_nonzero(valid) > 0:
    diff = depth_cu - depth_gt
    diff_valid = diff[valid]

    mae = np.mean(np.abs(diff_valid))
    rmse = np.sqrt(np.mean(diff_valid ** 2))
    mean_diff = np.mean(diff_valid)

    print("\n--- Depth comparison (custom vs GT) ---")
    print("Valid pixels:", np.count_nonzero(valid))
    print(f"MAE  (|d_custom - d_gt|)  = {mae:.4f} [depth units]")
    print(f"RMSE (d_custom - d_gt)    = {rmse:.4f} [depth units]")
    print(f"Mean signed diff          = {mean_diff:.4f} [depth units]")

    # Heatmap of absolute depth error
    abs_err = np.zeros_like(depth_gt, dtype=np.float32)
    abs_err[valid] = np.abs(diff[valid])

    # Clip to 95th percentile for nicer contrast
    vmax = np.percentile(abs_err[valid], 95)
    if vmax <= 0:
        vmax = abs_err[valid].max() if abs_err[valid].max() > 0 else 1.0

    err_norm = np.clip(abs_err, 0, vmax) / vmax * 255.0
    err_norm = err_norm.astype(np.uint8)
    err_color = cv2.applyColorMap(err_norm, cv2.COLORMAP_MAGMA)
    err_color[~valid] = 0

    cv2.imwrite(os.path.join(out_dir, "depth_abs_error_heatmap.png"), err_color)
else:
    print("No overlapping valid depth pixels – cannot compute statistics.")
