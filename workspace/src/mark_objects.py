"""
mark_objects.py

Description:
Mark objects on video and save them to the csv file using pretrained YOLO model.

Author: Milena Borczak 
Creation date: 10.04.2025
"""

import cv2
import pandas as pd
import numpy as np
import os
from ultralytics import YOLO
from workspace import config
from workspace.src.helpers.user_interface import UserInterface
from workspace.src.helpers.video_tools import create_video, mark_objects_on_video



_NEIGHBOR_WINDOW = 10  # frames to search left/right when validating detections

userInterface = UserInterface()
video_info = None          # populated lazily by _get_video_info()


def _get_video_info():
    """Return the cached VideoInfo, creating it on first access.

    Keeping initialisation out of the module scope makes the module importable
    in test environments that do not have a real video file.
    """
    global video_info
    if video_info is None:
        video_info = create_video()
    return video_info

def iou(box_1, box_2) -> float:
    """Calculates the Intersection over Union (IoU) of two bounding boxes

    Args:
        box_1 - ground truth bounding box
        box_2 - predicted bounding box

    Returns:
        float: Measure that represents how much two bounding boxes overlap.
    """
    if any(np.isnan(box_1)) or any(np.isnan(box_2)):
        return 0.0
    x_1 = max(box_1[0], box_2[0])
    y_1 = max(box_1[1], box_2[1])
    x_2 = min(box_1[2], box_2[2])
    y_2 = min(box_1[3], box_2[3])
    inter_area = max(0, x_2 - x_1) * max(0, y_2 - y_1)
    if inter_area == 0:
        return 0.0
    box_1_area = max(0, box_1[2] - box_1[0]) * max(0, box_1[3] - box_1[1])
    box_2_area = max(0, box_2[2] - box_2[0]) * max(0, box_2[3] - box_2[1])
    union_area = box_1_area + box_2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def fill_missing_ids(df: pd.DataFrame,
                     lookahead_frames: int = 5,
                     iou_threshold: float = 0.5) -> pd.DataFrame:
    """
    Completes missing Object_IDs in rows with Object_Detected == 'Yes'
    based on matching bboxes (IoU) to subsequent frames.

    Args:
        df: DataFrame with columns:
            Frame_Number, Object_Detected, Object_ID, X1,Y1,X2,Y2
        lookahead_frames: how many frames ahead to look for a matching ID
        iou_threshold: minimum IoU to consider it the same object

    Returns:
        DataFrame with Object_ID
    """
    df_fixed = df.copy()

    # remove IDs that appear in only one frame (single-frame detections are unreliable)
    id_counts = df_fixed.loc[df_fixed["Object_Detected"] == "Yes"].groupby("Object_ID")["Frame_Number"].nunique()
    single_frame_ids = id_counts[id_counts == 1].index
    df_fixed.loc[df_fixed["Object_ID"].isin(single_frame_ids), "Object_ID"] = np.nan


    df_fixed = df_fixed.sort_values(["Frame_Number"]).reset_index(drop=True)

    df_fixed["Object_ID"] = df_fixed["Object_ID"].replace("", np.nan)

    frame_to_indices = df_fixed.groupby("Frame_Number").indices

    missing_idx = df_fixed[
        (df_fixed["Object_Detected"] == "Yes") &
        (df_fixed["Object_ID"].isna())
    ].index.tolist()

    for idx in missing_idx:
        frame = int(df_fixed.at[idx, "Frame_Number"])

        box = df_fixed.loc[idx, ["X1", "Y1", "X2", "Y2"]].astype(float).values

        best_iou = 0.0
        best_id = None

        # search both backward and forward within ±lookahead_frames
        search_frames = (
            list(range(max(1, frame - lookahead_frames), frame)) +
            list(range(frame + 1, frame + 1 + lookahead_frames))
        )
        for f in search_frames:
            if f not in frame_to_indices:
                continue

            for j in frame_to_indices[f]:
                if df_fixed.at[j, "Object_Detected"] != "Yes":
                    continue
                if pd.isna(df_fixed.at[j, "Object_ID"]):
                    continue

                box2 = df_fixed.loc[j, ["X1", "Y1", "X2", "Y2"]].astype(float).values

                score = iou(box, box2)

                if score > best_iou:
                    best_iou = score
                    best_id = df_fixed.at[j, "Object_ID"]

        if best_id is not None and best_iou >= iou_threshold:
            df_fixed.at[idx, "Object_ID"] = best_id

    mask_filtered = (df_fixed["Object_Detected"] == "Yes") & (df_fixed["Object_ID"].isna())
    df_fixed.loc[mask_filtered, "Detected_By"] = "FilteredOut"
    df_fixed.loc[mask_filtered, "Object_Detected"] = "No"
    df_fixed.loc[mask_filtered, "Object_ID"] = np.nan
    df_fixed.loc[mask_filtered, "X1"] = np.nan
    df_fixed.loc[mask_filtered, "Y1"] = np.nan
    df_fixed.loc[mask_filtered, "X2"] = np.nan
    df_fixed.loc[mask_filtered, "Y2"] = np.nan
    df_fixed.loc[mask_filtered, "Confidence"] = np.nan


    return df_fixed


def process_video(video_path=None, model_path=config.MODEL, conf=0.5, iou_threshold=0.4):
    """Run YOLO tracking on *video_path* and return one DataFrame row per detected box per frame.

    Args:
        video_path: Path to the input video file.  Defaults to the first video
            found in ``config.INPUT_VIDEO_DIR`` when *None*.
        model_path: Path to the YOLO weights file.
        conf: Minimum confidence score for a detection to be kept.
        iou_threshold: IoU threshold used by the YOLO tracker for NMS.

    Returns:
        DataFrame with columns: Frame_Number, Resolution_Width, Resolution_Height,
        Frame_Start_Time, Frame_End_Time, Object_Detected, Detected_By,
        Object_ID, X1, Y1, X2, Y2, Confidence.
    """
    if video_path is None:
        video_path = _get_video_info().video_path
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frame_number = 0
    results_list = []

    # stream=True yields results frame by frame without loading the whole video into memory
    for result in model.track(source=video_path, conf=conf, iou=iou_threshold, persist=True, stream=True):
        frame_start_time = frame_number / fps
        frame_end_time = (frame_number + 1) / fps
        frame_number += 1

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            results_list.append([
                frame_number, w, h,
                frame_start_time, frame_end_time,
                "No", "", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            ])
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        ids = None
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy()
        else:
            ids = np.full((len(xyxy),), np.nan)

        for (x1, y1, x2, y2), c, obj_id in zip(xyxy, confs, ids):
            results_list.append([
                frame_number, w, h,
                frame_start_time, frame_end_time,
                "Yes", "YOLO", float(obj_id) if not np.isnan(obj_id) else np.nan,
                float(x1), float(y1), float(x2), float(y2), float(c)
            ])

    return pd.DataFrame(results_list, columns=[
        "Frame_Number", "Resolution_Width", "Resolution_Height",
        "Frame_Start_Time", "Frame_End_Time",
        "Object_Detected", "Detected_By", "Object_ID",
        "X1", "Y1", "X2", "Y2", "Confidence"
    ])


def match_boxes_by_iou(current_boxes: np.ndarray,
                       next_boxes: np.ndarray,
                       iou_threshold: float = 0.4):
    """Match current-frame boxes to next-frame boxes by IoU (greedy, one-to-one).

    Args:
        current_boxes: (N, 5) array — [x1, y1, x2, y2, Object_ID]
        next_boxes:    (M, 5) array — [x1, y1, x2, y2, Object_ID]
        iou_threshold: minimum IoU score to accept a match.

    Returns:
        List of (curr_row, next_row) pairs from the input arrays.
    """
    if len(current_boxes) == 0 or len(next_boxes) == 0:
        return []

    pairs = []
    used_next = set()

    for c in current_boxes:
        best_iou = 0.0
        best_j = None
        for j, n in enumerate(next_boxes):
            if j in used_next:
                continue
            score = iou(c[:4], n[:4])
            if score > best_iou:
                best_iou = score
                best_j = j

        if best_j is not None and best_iou >= iou_threshold:
            used_next.add(best_j)
            pairs.append((c, next_boxes[best_j]))

    return pairs

def interpolate_missing_per_id(df: pd.DataFrame,
                               interpolation_threshold: int = 15,
                               iou_match_threshold: float = 0.3,
                               fps: float = None) -> list[dict]:
    """Linearly interpolate missing frames for each Object_ID independently.

    Handles the case where an object disappears for a few frames while other
    objects remain visible (so the frame is not globally "empty").

    Args:
        df: DataFrame after the first global interpolation pass.
        interpolation_threshold: maximum gap size (in frames) to interpolate across.
        iou_match_threshold: minimum IoU between the boundary bboxes to allow interpolation.
        fps: video frame rate used to compute frame timestamps.  Falls back to the
            module-level ``video_info.fps`` when not provided.

    Returns:
        List of row dicts with ``Detected_By='Interpolated_ID'`` to be appended.
    """
    effective_fps = fps if fps is not None else _get_video_info().fps

    frames_to_add = []

    df_yes = df[df["Object_Detected"] == "Yes"].copy()
    df_yes = df_yes.dropna(subset=["Object_ID"])
    df_yes["Frame_Number"] = df_yes["Frame_Number"].astype(int)

    # process each object ID independently
    for obj_id, g in df_yes.groupby("Object_ID"):
        g = g.sort_values("Frame_Number")

        frames = g["Frame_Number"].values

        # walk through consecutive detections for this ID
        for i in range(len(frames) - 1):
            f1 = int(frames[i])
            f2 = int(frames[i + 1])
            gap = f2 - f1 - 1

            if gap < 1 or gap > interpolation_threshold:
                continue

            row_start = g[g["Frame_Number"] == f1].iloc[0]
            row_end = g[g["Frame_Number"] == f2].iloc[0]

            bbox_start = row_start[["X1", "Y1", "X2", "Y2"]].astype(float).values
            bbox_end = row_end[["X1", "Y1", "X2", "Y2"]].astype(float).values

            # skip if the boundary bboxes are too dissimilar (object likely moved out of frame)
            if iou(bbox_start, bbox_end) < iou_match_threshold:
                continue

            for k in range(1, gap + 1):
                interp_frame = f1 + k
                alpha = k / (gap + 1)
                interpolated_bbox = (1 - alpha) * bbox_start + alpha * bbox_end

                new_row = {
                    "Frame_Number": interp_frame,
                    "Resolution_Width": row_start["Resolution_Width"],
                    "Resolution_Height": row_start["Resolution_Height"],
                    "Frame_Start_Time": (interp_frame - 1) / effective_fps,
                    "Frame_End_Time": interp_frame / effective_fps,
                    "Object_Detected": "Yes",
                    "Detected_By": "Interpolated_ID",
                    "Object_ID": obj_id,
                    "X1": int(interpolated_bbox[0]),
                    "Y1": int(interpolated_bbox[1]),
                    "X2": int(interpolated_bbox[2]),
                    "Y2": int(interpolated_bbox[3]),
                    "Confidence": ""
                }
                frames_to_add.append(new_row)

    return frames_to_add

def clean_data(df: pd.DataFrame, iou_threshold: float = 0.5, gap_threshold: int = 15, interpolation_threshold: int = 30) -> pd.DataFrame:
    """
    Processes detected objects DataFrame and clean data. 
    - Remove false positives 
    - Add false negatives
    - Add objects interpolation

    Args:
        df: Detected objects data frame
        iou_threshold : The minimum IoU required to consider a detected object the same as previous. It's used to add new object_id. 
        gap_threshold : Minimum interval of frames to consider the object as new
        interpolation_threshold : Maximum number of frames without detection to interpolate object

    Returns:
        DataFrame: A list of detection results for each frame.
    """

    userInterface.print_info("Data cleaning...")

    df_cleaned = df.copy()  # work on a copy so the caller's DataFrame is never mutated
    grouped = df_cleaned.groupby("Frame_Number")
    rows_to_remove: set = set()   # index values of individual detections to invalidate
    frames_to_add = []
    all_frames = sorted(df_cleaned["Frame_Number"].unique())

    for frame in all_frames:
        frame_data = grouped.get_group(frame)
        yes_rows = frame_data[frame_data["Object_Detected"] == "Yes"]
        if yes_rows.empty:
            continue

        # pre-compute neighbor bboxes once per frame (shared across all detections in this frame)
        neighbor_bboxes = []
        for nf in range(frame - _NEIGHBOR_WINDOW, frame + _NEIGHBOR_WINDOW + 1):
            if nf != frame and nf in grouped.groups:
                ndf = grouped.get_group(nf)
                if "Yes" in ndf["Object_Detected"].values:
                    neighbor_bboxes.extend(ndf[["X1", "Y1", "X2", "Y2"]].dropna().values)

        # validate each detection independently
        for idx, row in yes_rows.iterrows():
            bbox = row[["X1", "Y1", "X2", "Y2"]].astype(float).values
            if any(np.isnan(v) for v in bbox):
                continue  # NaN-coord rows are handled downstream
            similar_found = any(iou(bbox, nbbox) > iou_threshold for nbbox in neighbor_bboxes)
            if not similar_found:
                rows_to_remove.add(idx)


    mask_filtered = df_cleaned.index.isin(rows_to_remove)
    df_cleaned.loc[mask_filtered, "Object_Detected"] = "No"
    df_cleaned.loc[mask_filtered, "Detected_By"] = "FilteredOut"
    df_cleaned.loc[mask_filtered, "Object_ID"] = np.nan
    df_cleaned.loc[mask_filtered, "X1"] = np.nan
    df_cleaned.loc[mask_filtered, "Y1"] = np.nan
    df_cleaned.loc[mask_filtered, "X2"] = np.nan
    df_cleaned.loc[mask_filtered, "Y2"] = np.nan
    df_cleaned.loc[mask_filtered, "Confidence"] = np.nan

    df_cleaned_sorted = df_cleaned.sort_values(["Frame_Number", "Object_ID"])
    detected_frames = df_cleaned_sorted[df_cleaned_sorted["Object_Detected"] == "Yes"]["Frame_Number"].unique()

    for i in range(len(detected_frames) - 1):
        current_f = detected_frames[i]
        next_f = detected_frames[i + 1]
        gap = next_f - current_f - 1
        
        if 1 <= gap <= interpolation_threshold:
            current_boxes = df_cleaned_sorted[df_cleaned_sorted["Frame_Number"] == current_f][["X1", "Y1", "X2", "Y2", "Object_ID"]].dropna().values
            next_boxes = df_cleaned_sorted[df_cleaned_sorted["Frame_Number"] == next_f][["X1", "Y1", "X2", "Y2", "Object_ID"]].dropna().values

            matched_pairs = match_boxes_by_iou(current_boxes, next_boxes, iou_threshold=0.3)

            for curr, nxt in matched_pairs:
                bbox_start = curr[:4].astype(float)
                object_id = int(curr[4])
                bbox_end = nxt[:4].astype(float)

                for k in range(1, gap + 1):
                    interp_frame = current_f + k
                    alpha = k / (gap + 1)
                    interpolated_bbox = (1 - alpha) * bbox_start + alpha * bbox_end

                    new_row = {
                        "Frame_Number": interp_frame,
                        "Resolution_Width": _get_video_info().frame_width,
                        "Resolution_Height": _get_video_info().frame_height,
                        "Frame_Start_Time": (interp_frame - 1) / _get_video_info().fps,
                        "Frame_End_Time": interp_frame / _get_video_info().fps,
                        "Object_Detected": "Yes",
                        "Detected_By": "Interpolated",
                        "Object_ID": object_id,
                        "X1": int(interpolated_bbox[0]),
                        "Y1": int(interpolated_bbox[1]),
                        "X2": int(interpolated_bbox[2]),
                        "Y2": int(interpolated_bbox[3]),
                        "Confidence": ""
                    }
                    frames_to_add.append(new_row)


    if frames_to_add:
        interpolated_frames = {row["Frame_Number"] for row in frames_to_add}
        df_cleaned = df_cleaned[
            ~((df_cleaned["Frame_Number"].isin(interpolated_frames)) & (df_cleaned["Object_Detected"] == "No"))
        ]

    df_cleaned_final = pd.concat([df_cleaned, pd.DataFrame(frames_to_add)], ignore_index=True)
    df_cleaned_final = df_cleaned_final.sort_values(["Frame_Number", "Object_ID"]).reset_index(drop=True)

    # second pass: per-ID interpolation for gaps where other objects are still visible
    frames_to_add_id = interpolate_missing_per_id(
        df_cleaned_final,
        interpolation_threshold=interpolation_threshold,
        iou_match_threshold=0.3,
        fps=_get_video_info().fps
    )

    if frames_to_add_id:
        # append interpolated rows without removing existing 'No' rows for other objects
        df_cleaned_final = pd.concat([df_cleaned_final, pd.DataFrame(frames_to_add_id)], ignore_index=True)
        df_cleaned_final = df_cleaned_final.sort_values(["Frame_Number", "Object_ID"]).reset_index(drop=True)


    df_detected = df_cleaned_final[df_cleaned_final["Object_Detected"] == "Yes"].copy()

    df_detected.sort_values(by=["Object_ID", "Frame_Number"], inplace=True)

    new_id = 1
    new_ids = []

    last_seen = {}  # Object_ID -> (last_frame, new_assigned_id)

    for _, row in df_detected.iterrows():
        old_id = row["Object_ID"]
        frame = row["Frame_Number"]

        if old_id not in last_seen:
            last_seen[old_id] = (frame, new_id)
            new_ids.append(new_id)
            new_id += 1
        else:
            last_frame, assigned_id = last_seen[old_id]
            if frame - last_frame >= gap_threshold:
                last_seen[old_id] = (frame, new_id)
                new_ids.append(new_id)
                new_id += 1
            else:
                last_seen[old_id] = (frame, assigned_id)
                new_ids.append(assigned_id)

    df_detected["Object_ID"] = new_ids

    df_cleaned = df_cleaned_final[~df_cleaned_final.index.isin(df_detected.index)]

    df_cleaned = pd.concat([df_cleaned, df_detected], ignore_index=True)
    df_cleaned.sort_values(["Frame_Number", "Object_ID"], inplace=True)

    return df_cleaned

def save_to_csv(df: pd.DataFrame) -> None:
    path = config.CSV_LOGO_DETECTIONS
    df.to_csv(path, sep=';', index=False)
    userInterface.print_info("Detected objects saved to file: " + path)


def main():
    os.makedirs(config.DETECTIONS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.DETECTED_IMG_DIR, exist_ok=True)
    userInterface.print_info("MARKING OBJECTS MODULE, PRESS CTRL+C TO EXIT")
    try:
        detected_objects = process_video()
        detected_objects = fill_missing_ids(detected_objects, lookahead_frames=5, iou_threshold=0.5)
        df_cleaned = clean_data(detected_objects)
        mark_objects_on_video(df_cleaned)
        save_to_csv(df_cleaned)
    except FileNotFoundError as e:
        userInterface.print_error("FileNotFoundError: " + str(e))
    except Exception as e:
        userInterface.print_error("Exception: "+ str(e))

if __name__ == "__main__":
    main()




