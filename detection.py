import cv2
from ultralytics import YOLO

def detect_objects_in_frame(img, model, conf_thres=0.3, max_detections=20):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.predict(img_rgb, verbose=False)[0]

    allowed = {0, 1, 2}
    detections = []

    for box in results.boxes:
        cls = int(box.cls[0].cpu().numpy())
        if cls not in allowed:
            continue

        conf = float(box.conf[0].cpu().numpy())
        if conf < conf_thres:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        detections.append(((x1, y1, x2, y2), cls, conf))

    detections.sort(key=lambda d: d[2], reverse=True)
    if max_detections is not None:
        detections = detections[:max_detections]

    return detections