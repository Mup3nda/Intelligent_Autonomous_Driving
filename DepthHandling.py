import cv2
import numpy as np

def compute_depth_map(left_img, right_img, stereo_params):
    """
    Compute disparity and depth from a rectified stereo pair using StereoSGBM.
    Assumes left_img and right_img are already rectified (KITTI-style).
    """
    # If your images are already rectified (which they are in "Rectified Images"),
    # you usually do NOT need to undistort again:
    left_rect = left_img
    right_rect = right_img

    # Convert to grayscale
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    # StereoSGBM parameters (you can tune these)
    window_size = 5

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 12,  # 192
        blockSize=window_size,
        P1=8 * window_size * window_size,
        P2=32 * window_size * window_size,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=50,
        speckleRange=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32)

    # SGBM returns disparity scaled by 16
    disparity /= 16.0

    # Optional: median blur to clean small speckles
    disparity = cv2.medianBlur(disparity, 5)

    # Compute depth: Z = f * B / disp
    focal_length = stereo_params['K_left'][0, 0]
    baseline = np.linalg.norm(stereo_params['T'])

    disp = disparity.copy()
    disp[disp <= 0.0] = np.nan  # invalid / infinite depth

    depth = (baseline * focal_length) / disp

    return depth, disparity

def depth_to_vis(depth_map, max_depth=80.0):
    depth = depth_map.copy()

    # Mask of valid depth values
    valid = np.isfinite(depth) & (depth > 0) & (depth < max_depth)

    # Start with all zeros
    vis = np.zeros_like(depth, dtype=np.float32)

    # Invert only valid depths so closer = brighter
    vis[valid] = max_depth - depth[valid]

    # Normalize only over valid range
    if np.any(valid):
        vmin = vis[valid].min()
        vmax = vis[valid].max()
        if vmax > vmin:
            vis[valid] = (vis[valid] - vmin) / (vmax - vmin) * 255.0

    vis = vis.astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    return color

def bbox_to_3d(bbox, depth_map, K, patch_radius=2):
        """
        Estimate a 3D point for a detection using the depth map.
        We take the median depth in a small patch around the bbox center,
        then backproject using camera intrinsics K.

        bbox: (x1, y1, x2, y2)
        depth_map: HxW array of depth (in meters)
        K: 3x3 intrinsics
        """
        x1, y1, x2, y2 = bbox
        cx = int(0.5 * (x1 + x2))
        cy = int(0.5 * (y1 + y2))

        H, W = depth_map.shape
        x0 = max(cx - patch_radius, 0)
        x1p = min(cx + patch_radius + 1, W)
        y0 = max(cy - patch_radius, 0)
        y1p = min(cy + patch_radius + 1, H)

        patch = depth_map[y0:y1p, x0:x1p]
        if patch.size == 0:
            return None

        depth = np.nanmedian(patch)
        if not np.isfinite(depth) or depth <= 0:
            return None

        fx  = K[0, 0]
        fy  = K[1, 1]
        cx0 = K[0, 2]
        cy0 = K[1, 2]

        X = (cx - cx0) / fx * depth
        Y = (cy - cy0) / fy * depth
        Z = depth

        return np.array([X, Y, Z], dtype=np.float32)  

def bbox_depth(bbox, depth_map, radius=5):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    h, w = depth_map.shape
    x0 = max(cx - radius, 0)
    x1p = min(cx + radius + 1, w)
    y0 = max(cy - radius, 0)
    y1p = min(cy + radius + 1, h)

    patch = depth_map[y0:y1p, x0:x1p]
    patch = patch[np.isfinite(patch) & (patch > 0)]

    if patch.size == 0:
        return None
    return np.median(patch)

def depth_to_3d(u, v, depth, K):
    fx = K[0,0]; fy = K[1,1]
    cx = K[0,2]; cy = K[1,2]

    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z], dtype=np.float32)
