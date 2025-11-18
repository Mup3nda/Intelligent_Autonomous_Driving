import cv2
import numpy as np
# Feature extractor (ORB = FAST + BRIEF)
orb = cv2.ORB_create(
    nfeatures=128,      # you can lower to 64 if needed
    scaleFactor=1.2,
    nlevels=8
)

def extract_roi_features(img, bbox, fg_mask=None, bins=(8, 8, 8)):
    """
    Return a dict of appearance features:
      - 'color': HSV histogram (np.array [N])
      - 'orb'  : mean ORB descriptor (np.array [128]) or None

    If fg_mask is provided (same size as img), only use foreground pixels
    when computing the color histogram and ORB (optional).
    """
    x1, y1, x2, y2 = bbox
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = max(int(x2), 0)
    y2 = max(int(y2), 0)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    # ---------- COLOR HIST ----------
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = None
    if fg_mask is not None:
        mask = fg_mask[y1:y2, x1:x2]
        if mask.size == 0:
            mask = None

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],  # H, S, V
        mask,
        bins,
        [0, 180, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)
    color_feat = hist.flatten().astype(np.float32)

    # ---------- ORB DESCRIPTORS ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y1:y2, x1:x2]

    orb_mask = mask if mask is not None else None
    kps, desc = orb.detectAndCompute(roi_gray, orb_mask)

    if desc is None or len(desc) == 0:
        orb_feat = None
    else:
        # ORB descriptors are 32-byte binary vectors; treat them as float
        # and average them to get a single 1D descriptor.
        orb_feat = desc.astype(np.float32).mean(axis=0)

    return {
        'color': color_feat,
        'orb': orb_feat
    }