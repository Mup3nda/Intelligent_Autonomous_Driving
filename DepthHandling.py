import cv2
import numpy as np

def preprocess_image(img, scale=0.5, blur_ksize=5):
    """
    Basic preprocessing: convert to grayscale, downsample, blur.
    - img: input image (BGR or grayscale)
    - scale: downsampling factor (0.5 = half size)
    - blur_ksize: Gaussian blur kernel size (odd int, e.g. 3,5,7)
    Returns: preprocessed grayscale image.
    """
    # Convert to grayscale if needed
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    
    # Downsample
    if scale != 1.0:
        img_gray = cv2.resize(
            img_gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
    
    # Blur
    if blur_ksize is not None and blur_ksize > 1:
        img_gray = cv2.GaussianBlur(img_gray, (blur_ksize, blur_ksize), 0)
    
    return img_gray


def preprocess_stereo_pair(left_img, right_img, scale=0.5, blur_ksize=5):
    """
    Preprocess left and right images in the same way.
    Returns: (left_proc, right_proc)
    """
    left_proc = preprocess_image(left_img, scale=scale, blur_ksize=blur_ksize)
    right_proc = preprocess_image(right_img, scale=scale, blur_ksize=blur_ksize)
    return left_proc, right_proc

# ============================================================
# 3. Disparity computation
# ============================================================

def create_stereo_matcher(method="sgbm"):
    """
    Create and configure a stereo matcher (StereoBM or StereoSGBM).
    Returns a cv2 stereo matcher object.
    """
    
    # These parameters are reasonable defaults for KITTI-like images.
    # You can tune them later.
    min_disp = 0
    num_disp = 16*12   # must be divisible by 16
    block_size = 5   # must be odd, 3..11 typical

    if method.lower() == "bm":
        stereo = cv2.StereoBM_create(
            numDisparities=num_disp,
            blockSize=block_size
        )
        # Optional BM tuning:
        stereo.setPreFilterCap(31)
        stereo.setUniquenessRatio(15)
        stereo.setSpeckleWindowSize(100)
        stereo.setSpeckleRange(32)
        stereo.setDisp12MaxDiff(1)
        stereo.setMinDisparity(min_disp)

    else:
        # StereoSGBM
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 1 * block_size * block_size,   # penalty on disparity change
            P2=32 * 1 * block_size * block_size,  # larger penalty
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=50,
            speckleRange=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    return stereo


def compute_disparity_map(left_proc, right_proc, method="sgbm"):
    """
    Compute disparity map from preprocessed grayscale stereo images.
    Returns a float32 disparity map (in pixels).
    """
    stereo = create_stereo_matcher(method=method)
    disp = stereo.compute(left_proc, right_proc).astype(np.float32)

    # For BM/SGBM in OpenCV, disparity is scaled by 16
    disp /= 16.0
    disp = cv2.medianBlur(disp, 5)

    return disp

# ============================================================
# 4. Disparity → Depth
# ============================================================

def compute_baseline_and_focal(stereo_params):
    """
    Compute focal length and baseline from stereo_params.
    Uses the rectified projection matrices P_rect_left/right:
      P_left  = [f 0 cx 0]
      P_right = [f 0 cx -f*B]
    So: B = -P_right[0,3] / f
    Returns: (f, B)
    """
    P_left = stereo_params['P_rect_left']
    P_right = stereo_params['P_rect_right']

    f = P_left[0, 0]
    # Baseline from right camera projection
    B = -P_right[0, 3] / f

    return f, B


def disparity_to_depth(disparity, stereo_params, min_disp=0.1):
    """
    Convert disparity map to depth map using:
      depth = f * B / disparity
    - disparity: float32 array (pixels)
    - stereo_params: dict with P_rect_left/right
    Returns: depth map (float32), same shape as disparity.
    """
    f, B = compute_baseline_and_focal(stereo_params)

    disp = disparity.copy().astype(np.float32)

    # Initialize depth with infinities (invalid)
    depth = np.full_like(disp, np.inf, dtype=np.float32)

    # Valid where disparity > min_disp
    valid = disp > min_disp

    depth[valid] = (f * B) / disp[valid]

    return depth

# ============================================================
# 5. Interface
# ============================================================

def compute_depth_map(left_img,
                      right_img,
                      stereo_params,
                      scale=1.0,
                      blur_ksize=5,
                      matcher_method="sgbm"):
    """
    Main pipeline for depth estimation.
    This MUST match Melvin's expected interface:

        depth, disparity = compute_depth_map(left_img, right_img, stereo_params)

    Steps:
      1) Preprocess (grayscale, downsample, blur)
      2) Compute disparity (StereoBM/StereoSGBM)
      3) Convert disparity → depth using calibration
    """
    # 1) Preprocess stereo pair returns scaled grayscale images
    left_proc, right_proc = preprocess_stereo_pair(
        left_img, right_img,
        scale=scale,
        blur_ksize=blur_ksize
    )

    # 2) Disparity
    disparity_small = compute_disparity_map(
        left_proc,
        right_proc,
        method=matcher_method
    )
   

    # Upscale disparity back to the original image resolution.
    # Note: disparities computed on a downsampled image must be scaled by 1/scale
    # to match the original image pixel coordinates.
    if scale == 0:
        raise ValueError("scale must be non-zero")

    inv_scale = 1.0 / scale

    # Preserve validity mask so invalid pixels don't produce bogus values after resize
    valid_mask = np.isfinite(disparity_small)
    disp_to_resize = disparity_small.copy()
    disp_to_resize[~valid_mask] = 0.0

    # Resize to original image size (cv2.resize uses (width, height) target)
    h_orig, w_orig = left_img.shape[:2]
    disparity = cv2.resize(disp_to_resize, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    # Resize mask with nearest interpolation to keep boolean integrity
    mask_resized = cv2.resize(valid_mask.astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Scale disparity values to original-pixel units
    disparity = disparity.astype(np.float32) * inv_scale

    # Restore invalid pixels
    disparity[~mask_resized] = np.nan
    # 3) Depth
    depth = disparity_to_depth(disparity, stereo_params)

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

#def compute_depth_map(left_img, right_img, stereo_params):
#    """
#    Compute disparity and depth from a rectified stereo pair using StereoSGBM.
#    Assumes left_img and right_img are already rectified (KITTI-style).
#    """
#    # If your images are already rectified (which they are in "Rectified Images"),
#    # you usually do NOT need to undistort again:
#    left_rect = left_img
#    right_rect = right_img
#
#    # Convert to grayscale
#    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
#    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
#
#    # StereoSGBM parameters (you can tune these)
#    window_size = 5
#
#    stereo = cv2.StereoSGBM_create(
#        minDisparity=0,
#        numDisparities=16 * 12,  # 192
#        blockSize=window_size,
#        P1=8 * window_size * window_size,
#        P2=32 * window_size * window_size,
#        disp12MaxDiff=1,
#        uniquenessRatio=5,
#        speckleWindowSize=50,
#        speckleRange=1,
#        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
#    )
#    disparity = stereo.compute(left_gray, right_gray).astype(np.float32)
#
#    # SGBM returns disparity scaled by 16
#    disparity /= 16.0
#
#    # Optional: median blur to clean small speckles
#    disparity = cv2.medianBlur(disparity, 5)
#
#    # Compute depth: Z = f * B / disp
#    focal_length = stereo_params['K_left'][0, 0]
#    baseline = np.linalg.norm(stereo_params['T'])
#
#    disp = disparity.copy()
#    disp[disp <= 0.0] = np.nan  # invalid / infinite depth
#
#    depth = (baseline * focal_length) / disp
#
#    return depth, disparity