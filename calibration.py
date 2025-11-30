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

def setup_stereo_camera(calibration, left_id='00', right_id='01'):
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

import numpy as np
import cv2  # <-- add this

def load_camera_calibration(calib_file):
    """
    Load camera calibration from calib_cam_to_cam.txt or custom calib file.
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


def setup_stereo_camera(calibration, left_id='00', right_id='01'):
    """
    Extract calibration for stereo pair (left & right cameras).
    Returns a dict with intrinsics, distortion, relative pose, and rectification.
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
    
    # Optional: rectified image sizes if present (S_rect_XX)
    S_rect_left = calibration.get(f'S_rect_{left_id}', None)
    S_rect_right = calibration.get(f'S_rect_{right_id}', None)

    return {
        'K_left': K_left, 'D_left': D_left,
        'K_right': K_right, 'D_right': D_right,
        'R': R, 'T': T,
        'R_rect_left': R_rect_left, 'R_rect_right': R_rect_right,
        'P_rect_left': P_rect_left, 'P_rect_right': P_rect_right,
        'S_rect_left': S_rect_left, 'S_rect_right': S_rect_right,
    }


def get_rectification_maps(stereo, image_size):
    """
    Create rectification maps for a stereo pair.

    stereo: dict from setup_stereo_camera()
    image_size: (width, height) of the RAW images

    Returns:
        mapL1, mapL2, mapR1, mapR2 suitable for cv2.remap
    """
    w, h = image_size

    K_left  = stereo['K_left']
    D_left  = stereo['D_left']
    K_right = stereo['K_right']
    D_right = stereo['D_right']

    Rl = stereo['R_rect_left']
    Rr = stereo['R_rect_right']
    Pl = stereo['P_rect_left']
    Pr = stereo['P_rect_right']

    mapL1, mapL2 = cv2.initUndistortRectifyMap(
        K_left, D_left, Rl, Pl, (w, h), cv2.CV_32FC1
    )
    mapR1, mapR2 = cv2.initUndistortRectifyMap(
        K_right, D_right, Rr, Pr, (w, h), cv2.CV_32FC1
    )

    return mapL1, mapL2, mapR1, mapR2


def rectify_stereo_pair(left_img, right_img, stereo):
    """
    Convenience function:
    Given raw left/right images and a stereo dict (from setup_stereo_camera),
    returns rectified left/right images.

    left_img, right_img: BGR (or grayscale) images with same size
    stereo: dict from setup_stereo_camera()
    """
    h, w = left_img.shape[:2]
    mapL1, mapL2, mapR1, mapR2 = get_rectification_maps(stereo, (w, h))

    rectL = cv2.remap(left_img,  mapL1, mapL2, cv2.INTER_LINEAR)
    rectR = cv2.remap(right_img, mapR1, mapR2, cv2.INTER_LINEAR)

    return rectL, rectR
