import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def draw_tracks_on_frame(img, tracks, class_names):
    """Draw bboxes and track IDs on frame."""
    for track_id, (bbox, class_id, conf) in tracks.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Handle dict or list for class_names
        if isinstance(class_names, dict):
            cname = class_names.get(class_id, str(class_id))
        else:
            if 0 <= class_id < len(class_names):
                cname = class_names[class_id]
            else:
                cname = str(class_id)

        label = f"ID:{track_id} {cname} {conf:.2f}"
        cv2.putText(img, label, (x1, max(y1-10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def draw_tracks_and_trajectories(img, tracks_with_hist, class_names):
    """
    Draw bboxes, IDs, and 2D trajectories on the frame.
    tracks_with_hist: {track_id: (bbox, class_id, conf, history2d_list)}
    """
    for track_id, (bbox, class_id, conf, hist2d) in tracks_with_hist.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # class name lookup as before
        if isinstance(class_names, dict):
            cname = class_names.get(class_id, str(class_id))
        else:
            if 0 <= class_id < len(class_names):
                cname = class_names[class_id]
            else:
                cname = str(class_id)

        label = f"ID:{track_id} {cname} {conf:.2f}"
        cv2.putText(img, label, (x1, max(y1-10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # draw trajectory polyline
        if len(hist2d) >= 2:
            pts = np.array(hist2d, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

    return img

import csv
def save_trajectories_csv(tracker, out_path="trajectories_3d.csv"):
    """
    Save all 3D trajectories to a CSV file.
    Columns: track_id, x, y, z
    """
    out_path = os.path.abspath(out_path)
    print(f"Saving trajectories to: {out_path}")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "x", "y", "z"])

        for tid, traj in tracker.trajectories.items():
            traj = np.array(traj)
            if traj.ndim != 2 or traj.shape[1] != 3:
                continue
            mask = np.all(np.isfinite(traj), axis=1)
            traj = traj[mask]
            for p in traj:
                writer.writerow([tid, float(p[0]), float(p[1]), float(p[2])])

    print("CSV export done.")
def show_trajectories_topdown(tracker,
                              scale=3.0,
                              img_size=800,
                              max_points_per_track=500):
    """
    Simple top-down (X-Z) trajectory visualization using OpenCV only.
    X -> horizontal, Z -> vertical (forward).
    """
    import numpy as np
    import cv2

    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # center of the canvas
    cx = img_size // 2
    cz = img_size - 50  # bottom for large Z

    colors = {}

    for tid, traj in tracker.trajectories.items():
        if len(traj) < 2:
            continue

        pts = np.array(traj)
        if pts.ndim != 2 or pts.shape[1] != 3:
            continue

        mask = np.all(np.isfinite(pts), axis=1)
        pts = pts[mask]

        if pts.shape[0] < 2:
            continue

        # Optionally subsample to avoid super dense lines
        if pts.shape[0] > max_points_per_track:
            idx = np.linspace(0, pts.shape[0] - 1, max_points_per_track).astype(int)
            pts = pts[idx]

        # X (left/right), Z (depth)
        X = pts[:, 0]
        Z = pts[:, 2]

        # Map meters to pixels
        pts_img = []
        for x, z in zip(X, Z):
            # smaller z (closer) near bottom, larger z (far) higher up
            u = int(cx + x * scale)
            v = int(cz - z * scale)
            pts_img.append((u, v))

        if len(pts_img) < 2:
            continue

        # color per track
        if tid not in colors:
            colors[tid] = (
                int(50 + (tid * 40) % 200),
                int(80 + (tid * 70) % 175),
                int(100 + (tid * 90) % 155)
            )
        col = colors[tid]

        for i in range(len(pts_img) - 1):
            cv2.line(canvas, pts_img[i], pts_img[i+1], col, 1)

    cv2.imshow("Top-down trajectories (X vs Z)", canvas)
    cv2.waitKey(0)
    cv2.destroyWindow("Top-down trajectories (X vs Z)")

# After processing, plot 3D trajectories
def plot_trajectories_3d(tracker, save_path=None):
    """
    Plot 3D trajectories stored in tracker.trajectories.
    Filters out NaNs and very short trajectories.
    Optionally saves to a PNG.
    """
    traj_dict = tracker.trajectories

    print("CWD is:", os.getcwd())
    print("Matplotlib backend:", matplotlib.get_backend())
    print("Trajectories (raw lengths):",
          {tid: len(traj) for tid, traj in traj_dict.items()})

    try:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        num_plotted = 0

        for tid, traj in traj_dict.items():
            # traj should be a list of (3,) arrays
            if not traj:
                continue

            traj = np.array(traj)

            # Guard against weird shapes
            if traj.ndim != 2 or traj.shape[1] != 3:
                print(f"Skipping track {tid}: unexpected shape {traj.shape}")
                continue

            # Keep only finite points
            mask = np.all(np.isfinite(traj), axis=1)
            traj = traj[mask]

            if traj.shape[0] < 2:
                # too short / no valid points
                continue

            num_plotted += 1
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"ID {tid}")

        ax.set_title("3D Trajectories (Camera Coordinates)")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        if num_plotted > 0:
            ax.legend()

        ax.view_init(elev=15, azim=-40)
        plt.tight_layout()

        # Always save a file, even if save_path is None
        if save_path is None:
            save_path = "trajectories_3d.png"
        save_path = os.path.abspath(save_path)
        print(f"Saving 3D plot to: {save_path}")
        plt.savefig(save_path, dpi=150)

        print(f"Plotted trajectories: {num_plotted}")
        plt.show()

    except Exception as e:
        print("ERROR while plotting trajectories:", repr(e))


