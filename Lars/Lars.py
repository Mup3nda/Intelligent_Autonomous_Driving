import cv2
import numpy as np

# ============================================================
# 1. Calibration loading 
# ============================================================

# def load_camera_calibration(calib_file):
#     """
#     Load camera calibration from calib_cam_to_cam.txt.
#     Returns a dictionary with calibration matrices for each camera.
#     """
#     calib = {}
    
#     with open(calib_file, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith('#'):
#                 continue
            
#             if ':' in line:
#                 key, values = line.split(':', 1)
#                 key = key.strip()
#                 values = np.array([float(x) for x in values.split()])
                
#                 # Matrices
#                 if 'K_' in key or 'R_' in key or 'R_rect_' in key:
#                     if len(values) == 9:      # 3x3 matrix
#                         calib[key] = values.reshape(3, 3)
#                 elif 'P_rect_' in key:
#                     if len(values) == 12:     # 3x4 matrix
#                         calib[key] = values.reshape(3, 4)
#                 # Distortion or other vectors
#                 elif 'D_' in key or 'T_' in key or 'S_' in key or 'S_rect_' in key:
#                     calib[key] = values
#                 else:
#                     calib[key] = values
    
#     return calib

def load_camera_calibration(calib_file):
    """
    Load camera calibration from calib_cam_to_cam.txt.
    Returns a dictionary with calibration matrices for each camera.
    Designed to work with KITTI-style files that also have calib_time etc.
    """
    calib = {}
    
    with open(calib_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Expect lines on the form "K_02: ..." or "P_rect_02: ..." or "calib_time: ..."
            if ':' not in line:
                continue

            key, values = line.split(':', 1)
            key = key.strip()
            values_strs = values.split()

            # Skip known non-numeric keys (like calib_time)
            if key.lower().startswith("calib_time"):
                continue

            # Try converting the rest of the line to floats.
            # If that fails, we just ignore that line.
            try:
                vals = np.array([float(x) for x in values_strs], dtype=np.float64)
            except ValueError:
                # Line contains non-numeric tokens we don't care about
                continue

            # Store depending on the key pattern
            if key.startswith('K_') or key.startswith('R_') or key.startswith('R_rect_'):
                if len(vals) == 9:
                    calib[key] = vals.reshape(3, 3)
            elif key.startswith('P_rect_'):
                if len(vals) == 12:
                    calib[key] = vals.reshape(3, 4)
            elif key.startswith('D_') or key.startswith('T_') or key.startswith('S_') or key.startswith('S_rect_'):
                calib[key] = vals  # keep as 1D
            else:
                # Any other numeric line we just store as 1D
                calib[key] = vals

    return calib



def setup_stereo_camera(calibration, left_id='02', right_id='03'):
    """
    Extract calibration for stereo pair (left & right cameras).
    Returns a dict 'stereo_params' used by compute_depth_map.
    """
    K_left = calibration[f'K_{left_id}'].copy()
    D_left = calibration[f'D_{left_id}'].copy()
    K_right = calibration[f'K_{right_id}'].copy()
    D_right = calibration[f'D_{right_id}'].copy()
    
    R = calibration[f'R_{right_id}'].copy()
    T = calibration[f'T_{right_id}'].copy()
    
    R_rect_left = calibration[f'R_rect_{left_id}'].copy()
    R_rect_right = calibration[f'R_rect_{right_id}'].copy()
    P_rect_left = calibration[f'P_rect_{left_id}'].copy()
    P_rect_right = calibration[f'P_rect_{right_id}'].copy()
    
    return {
        'K_left': K_left, 'D_left': D_left,
        'K_right': K_right, 'D_right': D_right,
        'R': R, 'T': T,
        'R_rect_left': R_rect_left, 'R_rect_right': R_rect_right,
        'P_rect_left': P_rect_left, 'P_rect_right': P_rect_right
    }

# ============================================================
# 2. Preprocessing
# ============================================================

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
    num_disp = 128   # must be divisible by 16
    block_size = 7   # must be odd, 3..11 typical

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
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
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
                      scale=0.5,
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
    # 1) Preprocess stereo pair
    left_proc, right_proc = preprocess_stereo_pair(
        left_img, right_img,
        scale=scale,
        blur_ksize=blur_ksize
    )

    # 2) Disparity
    disparity = compute_disparity_map(
        left_proc,
        right_proc,
        method=matcher_method
    )

    # 3) Depth
    depth = disparity_to_depth(disparity, stereo_params)

    return depth, disparity

# ============================================================
# 6. Main
# ============================================================

if __name__ == "__main__":
    # Example paths – change to your own
    calib_path = r"C:\Users\larst\Downloads\34759 Perception for autonomous systems, Fall 2025 - 11182025 - 744 PM\34759_final_project_rect\34759_final_project_rect\calib_cam_to_cam.txt"
    #02
    left_img_path = r"C:\Users\larst\Downloads\34759 Perception for autonomous systems, Fall 2025 - 11182025 - 744 PM\34759_final_project_rect\34759_final_project_rect\seq_01\image_02\data\000140.png" 
    #03
    right_img_path = r"C:\Users\larst\Downloads\34759 Perception for autonomous systems, Fall 2025 - 11182025 - 744 PM\34759_final_project_rect\34759_final_project_rect\seq_01\image_03\data\000140.png"

    # Load calibration
    calibration = load_camera_calibration(calib_path)
    stereo_params = setup_stereo_camera(calibration, left_id='02', right_id='03')

    # Load images
    left = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_img_path, cv2.IMREAD_COLOR)

    if left is None or right is None:
        raise FileNotFoundError("Could not load left/right images. Check the paths.")

    # Compute depth & disparity
    depth_map, disparity = compute_depth_map(left, right, stereo_params)

    # Quick visualization inside this script (grayscale disparity)
    disp_vis = disparity.copy()
    disp_vis[~np.isfinite(disp_vis)] = 0
    disp_vis = cv2.normalize(disp_vis, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX)
    disp_vis = disp_vis.astype(np.uint8)

    cv2.imshow("Disparity (normalized)", disp_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
