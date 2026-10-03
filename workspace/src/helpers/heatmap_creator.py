import pandas as pd
import numpy as np
import os
from workspace import config
from collections import deque
import cv2

from workspace.src.helpers.user_interface import UserInterface

userInterface = UserInterface()

def _odd(k: int) -> int:
    """Returns an odd number >= 1 (required by GaussianBlur)."""
    k = int(max(1, k))
    return k if k % 2 == 1 else k + 1

def render_video_with_heatmap(input_video_path: str,
                                      eyetracker_data_df: pd.DataFrame,
                                      output_video_path: str,
                                      frame_col: str = "Frame_Number",
                                      gx_col: str = "GAZE_X_VIDEO",
                                      gy_col: str = "GAZE_Y_VIDEO",
                                      window_size=20,    
                                      grid_scale=0.6,    
                                      blur_ksize=31,     
                                      gamma=1.15,       
                                      alpha=0.65):       
    """
    Creates a video with a heatmap of views in a moving time window (last N frames).
    Note: assumes that Frame_Number in CSV is 1-based -> we map to 0-based.
    """

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open the video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    gaze_by_frame = {
        int(k) - 1: v[[gx_col, gy_col]].values
        for k, v in eyetracker_data_df.groupby(frame_col)
    }

    buffer = deque(maxlen=window_size)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pts = gaze_by_frame.get(frame_idx)
        if pts is not None and len(pts) > 0:
            for x, y in pts:
                if not (np.isnan(x) or np.isnan(y)):
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= xi < w and 0 <= yi < h:
                        buffer.append((xi, yi))

        if len(buffer) > 0:
            xs = np.fromiter((p[0] for p in buffer), dtype=float, count=len(buffer))
            ys = np.fromiter((p[1] for p in buffer), dtype=float, count=len(buffer))

            m = (~np.isnan(xs)) & (~np.isnan(ys)) & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            xs, ys = xs[m], ys[m]

            if xs.size > 0:
                bins_w = max(16, int(w * grid_scale))
                bins_h = max(16, int(h * grid_scale))
                H, _, _ = np.histogram2d(xs, ys,
                                         bins=[bins_w, bins_h],
                                         range=[[0, w], [0, h]])
                H = H.T.astype(np.float32)

                k = _odd(blur_ksize)
                if k > 1:
                    H = cv2.GaussianBlur(H, (k, k), 0)

                if H.max() > 0:
                    H = (H / H.max()).astype(np.float32)
                else:
                    H = np.zeros_like(H, dtype=np.float32)

                H = np.power(H, gamma, dtype=np.float32)

                H = cv2.resize(H, (w, h), interpolation=cv2.INTER_LINEAR)

                H_255 = (H * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(H_255, cv2.COLORMAP_TURBO)

                A = (alpha * H).astype(np.float32)[..., None]
                frame = (frame.astype(np.float32) * (1 - A) +
                         heatmap_color.astype(np.float32) * A).astype(np.uint8)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    userInterface.print_info(f"HeatMap ready: {output_video_path}")

