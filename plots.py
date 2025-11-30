import matplotlib.pyplot as plt
import numpy as np
import os

def plot_track_trajectories_with_occlusion(tracker, save_dir=None):
    history = tracker.get_history()
    save_dir = os.path.abspath(save_dir)
    print("plot_track_trajectories_with_occlusion: #tracks in history =", len(history))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    num_plotted = 0

    for tid, entries in history.items():
        # Extract positions
        pos3d = [e['pos3d'] for e in entries]

        # skip tracks without any 3D info
        if all(p is None for p in pos3d):
            continue

        X = np.array([p[0] if p is not None else np.nan for p in pos3d])
        Y = np.array([p[1] if p is not None else np.nan for p in pos3d])
        Z = np.array([p[2] if p is not None else np.nan for p in pos3d])

        # if everything is NaN, skip
        if np.all(np.isnan(X)) or np.all(np.isnan(Z)):
            continue

        visible_mask = np.array([e['visible'] for e in entries], dtype=bool)
        reacq_mask   = np.array([e['reacquired'] for e in entries], dtype=bool)

        plt.figure()
        plt.title(f"Track {tid} 3D trajectory (top-down X–Z)")
        plt.plot(X, Z, linestyle='-', alpha=0.5)

        # visible positions
        plt.scatter(X[visible_mask], Z[visible_mask],
                    marker='o', label='visible', s=15)

        # occluded positions (predicted only)
        plt.scatter(X[~visible_mask], Z[~visible_mask],
                    marker='x', label='occluded', s=20)

        # where track was reacquired after occlusion
        if np.any(reacq_mask):
            plt.scatter(X[reacq_mask], Z[reacq_mask],
                        marker='s', label='reacquired', s=40)

        plt.xlabel('X [m]')
        plt.ylabel('Z [m]')
        plt.legend()
        plt.grid(True)

        if save_dir is not None:
            out_path = os.path.join(save_dir, f"track_{tid}_topdown.png")
            print("Saving:", out_path)
            plt.savefig(out_path, dpi=150)
            plt.close()
        else:
            plt.show()

        num_plotted += 1

    print("plot_track_trajectories_with_occlusion: plotted", num_plotted, "tracks")

def plot_reacquisition_error(tracker, save_path=None):
    history = tracker.get_history()
    save_path = os.path.abspath(save_path)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    jumps2d = []
    jumps3d = []

    for tid, entries in history.items():
        for e in entries:
            if e['reacquired']:
                if e['jump2d'] is not None:
                    jumps2d.append(e['jump2d'])
                if e['jump3d'] is not None:
                    jumps3d.append(e['jump3d'])

    if not jumps2d and not jumps3d:
        print("No reacquisition events logged.")
        return

    plt.figure()
    if jumps2d:
        plt.hist(jumps2d, bins=20, alpha=0.7, label='2D jump [px]')
    if jumps3d:
        plt.hist(jumps3d, bins=20, alpha=0.7, label='3D jump [m]')
    plt.xlabel('Jump magnitude')
    plt.ylabel('Count')
    plt.title('Reacquisition correction magnitude')
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

import csv
import numpy as np

def save_track_history_csv(tracker, save_path):
    """
    Dump per-frame per-track history to a single CSV.

    Columns:
        track_id, frame, visible, disappeared,
        center_u, center_v,
        pos3d_x, pos3d_y, pos3d_z,
        reacquired, jump2d, jump3d
    """
    history = tracker.get_history()
    save_path = os.path.abspath(save_path)
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'track_id', 'frame', 'visible', 'disappeared',
            'center_u', 'center_v',
            'pos3d_x', 'pos3d_y', 'pos3d_z',
            'reacquired', 'jump2d', 'jump3d'
        ])

        for tid, entries in history.items():
            for e in entries:
                center_u, center_v = e['center']
                pos3d = e['pos3d']
                if pos3d is None or not np.all(np.isfinite(pos3d)):
                    x, y, z = np.nan, np.nan, np.nan
                else:
                    x, y, z = float(pos3d[0]), float(pos3d[1]), float(pos3d[2])

                writer.writerow([
                    int(tid),
                    int(e['frame']),
                    int(bool(e['visible'])),
                    int(e['disappeared']),
                    float(center_u),
                    float(center_v),
                    x, y, z,
                    int(bool(e['reacquired'])),
                    e['jump2d'] if e['jump2d'] is not None else np.nan,
                    e['jump3d'] if e['jump3d'] is not None else np.nan,
                ])
def save_reacquisition_errors_csv(tracker, save_path):
    """
    Save one row per reacquisition event:
        track_id, frame, jump2d, jump3d
    """
    history = tracker.get_history()
    save_path = os.path.abspath(save_path)
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', 'frame', 'jump2d', 'jump3d'])

        for tid, entries in history.items():
            for e in entries:
                if not e['reacquired']:
                    continue
                j2 = e['jump2d']
                j3 = e['jump3d']
                writer.writerow([
                    int(tid),
                    int(e['frame']),
                    j2 if j2 is not None else np.nan,
                    j3 if j3 is not None else np.nan,
                ])
