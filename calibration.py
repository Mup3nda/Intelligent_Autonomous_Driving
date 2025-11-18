import numpy as np

def load_camera_calibration(calib_file):
    """
    Load camera calibration from calib_cam_to_cam.txt.
    Returns a dictionary with calibration matrices for each camera.
    """
    calib = {}
    
    with open(calib_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse lines like "K_00: 123.45 0 456.78 ..."
            if ':' in line:
                key, values = line.split(':', 1)
                key = key.strip()
                values = np.array([float(x) for x in values.split()])
                
                # Reshape to matrix
                if 'K_' in key or 'R_' in key or 'R_rect_' in key:
                    if len(values) == 9:  # 3x3 matrix
                        calib[key] = values.reshape(3, 3)
                elif 'P_rect_' in key:
                    if len(values) == 12:  # 3x4 projection matrix
                        calib[key] = values.reshape(3, 4)
                elif 'D_' in key:
                    calib[key] = values  # Keep as 1D array
                elif 'T_' in key or 'S_' in key or 'S_rect_' in key:
                    calib[key] = values
                else:
                    calib[key] = values
    
    return calib

def setup_stereo_camera(calibration, left_id='02', right_id='03'):
    """
    Extract calibration for stereo pair (left & right cameras).
    Returns: K_left, D_left, K_right, D_right, R, T
    """
    # Left camera intrinsics
    K_left = calibration[f'K_{left_id}'].copy()
    D_left = calibration[f'D_{left_id}'].copy()
    
    # Right camera intrinsics
    K_right = calibration[f'K_{right_id}'].copy()
    D_right = calibration[f'D_{right_id}'].copy()
    
    # Rotation and translation between left and right
    R = calibration[f'R_{right_id}'].copy()
    T = calibration[f'T_{right_id}'].copy()
    
    # Rectification matrices
    R_rect_left = calibration[f'R_rect_{left_id}'].copy()
    R_rect_right = calibration[f'R_rect_{right_id}'].copy()
    
    # Rectified projection matrices
    P_rect_left = calibration[f'P_rect_{left_id}'].copy()
    P_rect_right = calibration[f'P_rect_{right_id}'].copy()
    
    return {
        'K_left': K_left, 'D_left': D_left,
        'K_right': K_right, 'D_right': D_right,
        'R': R, 'T': T,
        'R_rect_left': R_rect_left, 'R_rect_right': R_rect_right,
        'P_rect_left': P_rect_left, 'P_rect_right': P_rect_right
        }