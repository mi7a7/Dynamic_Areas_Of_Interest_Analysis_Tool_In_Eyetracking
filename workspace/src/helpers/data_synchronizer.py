import pandas as pd
import numpy as np
import os
from workspace import config
from workspace.src.helpers.user_interface import UserInterface
from workspace.src.helpers.utilities import extract_user_id_from_filename, list_user_files
from workspace.src.helpers.video_tools import create_video

userInterface = UserInterface()


def select_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects only those columns from the eye tracker that are needed for analysis:
    - time and synchronization (TIME, CS)
    - gazes (FPOG, BPOG, LPOG, RPOG)
    Returns a new DataFrame with the selected columns.
    Raises:
        ValueError: if any required column is absent, with a readable list of
            the missing column names.
    """
    required_cols = [
        "TIME", "CS",
        "FPOGX", "FPOGY", "FPOGD", "FPOGID", "FPOGV",
        "BPOGX", "BPOGY", "BPOGV",
    ]

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required eye-tracker columns: {sorted(missing)}"
        )

    return df[required_cols].copy()


def compute_frame_time_ranges(objects_data_df,
                    col_frame_number="Frame_Number",
                    col_frame_start="Frame_Start_Time",
                    col_frame_end="Frame_End_Time"):
    """
    Collects unique frames and their time intervals: [min start, max end] per Frame_Number.
    """

    # time intervals for frames, frames could repeat themselves when there were several objects 
    
    frames_df = (
        objects_data_df
        .groupby(col_frame_number, as_index=False)
        .agg({col_frame_start: "min", col_frame_end: "max"})
        .sort_values(col_frame_start)
        .reset_index(drop=True)
    )
    return frames_df

def sync_eyetracker_to_frames(eyetracker_data_processed_df,
                              frames_df,
                              col_time="TIME",
                              col_time_relative="TIME_RELATIVE_S",
                              col_frame_number="Frame_Number",
                              col_frame_start="Frame_Start_Time",
                              col_frame_end="Frame_End_Time"):
    """
    1) determine the start of the video after the event code,
    2) calculate the relative time and trim to the video time,
    3) merge_asof with frame intervals,
    4) count the frame length.
    
    Returns: processed eye tracker DataFrame (sorted by index).
    """

    SYNC_DELAY_S = 0.277  # VLC delay

    # determine the beginning of the video (first occurrence of the event: CS == event_code_start)
    # CS=3 release the mouse button    
    start_video_t0_idx = eyetracker_data_processed_df.index[eyetracker_data_processed_df["CS"] == 3].min() 
    start_video_t0_time = eyetracker_data_processed_df.loc[start_video_t0_idx, col_time]

    start_video_t0_time += SYNC_DELAY_S

    # determination of relative time
    eyetracker_data_processed_df[col_time_relative] = eyetracker_data_processed_df[col_time] - start_video_t0_time

    # end of the video (from objects)
    end_video_tmax_time = frames_df[col_frame_end].max()

    # trim to [0, end_video_tmax_time]
    # remove records before the start of the video and after the end of the video
    eyetracker_data_processed_df = eyetracker_data_processed_df[
        (eyetracker_data_processed_df[col_time_relative] >= 0) &
        (eyetracker_data_processed_df[col_time_relative] <= end_video_tmax_time)
    ].copy()

    # sorting eye tracker data by relative time to use merge_asof
    eyetracker_data_processed_df = eyetracker_data_processed_df.sort_values(col_time_relative).reset_index(drop=True)

    # pinning frames (last frame whose start <= sample time)
    # Assigning frames using the “merge_asof” method:
    # for each time, we take the last frame whose Start_Time <= TIME_RELATIVE_S,
    # and filter out cases where the time does not fit within END

    eyetracker_data_processed_df = pd.merge_asof(
        eyetracker_data_processed_df,
        frames_df[[col_frame_number, col_frame_start, col_frame_end]],
        left_on=col_time_relative,
        right_on=col_frame_start,
        direction="backward",
        allow_exact_matches=True
    )

    
    # Drop samples whose timestamp overshoots the assigned frame's end boundary.
    # merge_asof uses backward matching (last frame whose start <= sample time),
    # so without this guard a sample at t=0.099 s could be kept in a frame that
    # ended at t=0.04 s if the next frame starts at t=0.10 s,
    # this probelm should not exist.
    eyetracker_data_processed_df = eyetracker_data_processed_df[
        eyetracker_data_processed_df[col_time_relative]
        <= eyetracker_data_processed_df[col_frame_end]
    ].copy()

    # Remove rows where no frame could be assigned (NaN Frame_Number).
    eyetracker_data_processed_df = eyetracker_data_processed_df.dropna(
        subset=[col_frame_number]
    )

    # frame length
    eyetracker_data_processed_df['FRAME_DURATION'] = eyetracker_data_processed_df[col_frame_end] - eyetracker_data_processed_df[col_frame_start]
    eyetracker_data_processed_df = eyetracker_data_processed_df.sort_index()

    return eyetracker_data_processed_df


def pog_fractions_to_screen_pixels(eyetracker_data_df: pd.DataFrame, screen_w: int, screen_h: int) -> pd.DataFrame:
    """
    Converts POG (Point of Gaze) coordinates in fractions [0..1] to screen pixels.
    Supported pog_types: FPOG, BPOG, LPOG, RPOG.

    Adds columns:
      - <POG>X_px, <POG>Y_px, <POG>V_bool
    e.g. FPOGX_px, FPOGY_px, FPOGV_bool
    """
    eyetracker_data_df = eyetracker_data_df.copy()
    pog_types = ["FPOG", "BPOG"]
    raw_cols_to_drop: list = []  # accumulated across all iterations

    for pog in pog_types:
        x_col = f"{pog}X"
        y_col = f"{pog}Y"
        v_col = f"{pog}V"
        raw_cols_to_drop.extend([x_col, y_col, v_col])

        if x_col not in eyetracker_data_df.columns: eyetracker_data_df[x_col] = np.nan
        if y_col not in eyetracker_data_df.columns: eyetracker_data_df[y_col] = np.nan
        if v_col not in eyetracker_data_df.columns: eyetracker_data_df[v_col] = np.nan

        x_fraction = pd.to_numeric(eyetracker_data_df[x_col], errors="coerce")
        y_fraction = pd.to_numeric(eyetracker_data_df[y_col], errors="coerce")
        valid  = pd.to_numeric(eyetracker_data_df[v_col], errors="coerce")

        eyetracker_data_df[f"{pog}X_PX"] = x_fraction * float(screen_w)
        eyetracker_data_df[f"{pog}Y_PX"] = y_fraction * float(screen_h)
        eyetracker_data_df[f"{pog}V_BOOL"] = (valid == 1)

    eyetracker_data_df = eyetracker_data_df.drop(columns=raw_cols_to_drop, errors="ignore")

    final_cols_order = [
    "TIME", "CS", "TIME_RELATIVE_S",
    "Frame_Number", "Frame_Start_Time", "Frame_End_Time", "FRAME_DURATION",
    "FPOGX_PX", "FPOGY_PX", "FPOGV_BOOL", "FPOGD", "FPOGID",
    "BPOGX_PX", "BPOGY_PX", "BPOGV_BOOL",
    ]

    eyetracker_data_df = eyetracker_data_df[final_cols_order]


    return eyetracker_data_df


def select_final_gaze(eyetracker_data_df: pd.DataFrame,
                      priority=("FPOG","BPOG"),
                      gaze_x_col="GAZE_X_SCREEN",
                      gaze_y_col="GAZE_Y_SCREEN",
                      src_gaze_col="GAZE_SOURCE") -> pd.DataFrame:
    """
    Selects the final coordinates of the view in screen pixels
    based on the priority of POG types. Assumes columns:
    <POG>X_px, <POG>Y_px, <POG>V_bool  (e.g., FPOGX_px, FPOGV_bool, ...)
    """
    eyetracker_data_df = eyetracker_data_df.copy()

    gaze_x = pd.Series(np.nan, index=eyetracker_data_df.index, dtype="float64")
    gaze_y = pd.Series(np.nan, index=eyetracker_data_df.index, dtype="float64")
    src_gaze = pd.Series(np.nan, index=eyetracker_data_df.index, dtype="object")

    for pog in priority:
        v = eyetracker_data_df.get(f"{pog}V_BOOL")
        x = eyetracker_data_df.get(f"{pog}X_PX")
        y = eyetracker_data_df.get(f"{pog}Y_PX")
        if v is None or x is None or y is None:
            continue

        pick = v.fillna(False) & gaze_x.isna() & gaze_y.isna()
        gaze_x = gaze_x.where(~pick, x)
        gaze_y = gaze_y.where(~pick, y)
        src_gaze = src_gaze.where(~pick, pog)

    eyetracker_data_df[gaze_x_col] = gaze_x
    eyetracker_data_df[gaze_y_col] = gaze_y
    eyetracker_data_df[src_gaze_col] = src_gaze

    return eyetracker_data_df


def gaze_to_video_coords(eyetracker_data_df: pd.DataFrame,
                           screen_w: int,
                           screen_h: int,
                           video_w: int,
                           video_h: int,
                           x_in="GAZE_X_SCREEN",
                           y_in="GAZE_Y_SCREEN",
                           x_out="GAZE_X_VIDEO",
                           y_out="GAZE_Y_VIDEO") -> pd.DataFrame:
    """
    Converts view coordinates from screen layout (px) to video layout (px).
    
    Cases:
      - video <= screen (both dims): --no-autoscale used in VLC, 
        video displayed pixel-perfect, centered → subtract offset only
      - video > screen (any dim): VLC scales to fit preserving aspect ratio (letterbox)
        → subtract offset and divide by scale
    """
    eyetracker_data_df = eyetracker_data_df.copy()

    needs_scaling = (video_w > screen_w) or (video_h > screen_h)

    if needs_scaling:
        # VLC scales to fit screen preserving aspect ratio (letterbox)
        scale = min(screen_w / video_w, screen_h / video_h)
        scaled_w = video_w * scale
        scaled_h = video_h * scale
        offset_x = (screen_w - scaled_w) / 2.0
        offset_y = (screen_h - scaled_h) / 2.0
    else:
        # VLC displays pixel-perfect, video centered on screen
        scale = 1.0
        offset_x = (screen_w - video_w) / 2.0
        offset_y = (screen_h - video_h) / 2.0

    eyetracker_data_df[x_out] = (eyetracker_data_df[x_in] - offset_x) / scale
    eyetracker_data_df[y_out] = (eyetracker_data_df[y_in] - offset_y) / scale

    return eyetracker_data_df

def transform_one_user(user_file_path, objects_data_path, screen_w, screen_h, video_w, video_h):
    """
    Apply synchronization methods to single user data.
    """
    eyetracker_data_raw_df = pd.read_csv(user_file_path, sep=";")
    objects_data_orginal_df = pd.read_csv(objects_data_path, sep=';')
    eyetracker_data_processed_df = eyetracker_data_raw_df.copy()
    objects_data_df = objects_data_orginal_df.copy()

    eyetracker_data_processed_df = select_relevant_columns(eyetracker_data_processed_df)
    frames_df = compute_frame_time_ranges(objects_data_df)

    eyetracker_data_processed_df = sync_eyetracker_to_frames(eyetracker_data_processed_df, frames_df)

    eyetracker_data_processed_df = pog_fractions_to_screen_pixels(eyetracker_data_processed_df, screen_w, screen_h)

    eyetracker_data_processed_df = select_final_gaze(eyetracker_data_processed_df)

    eyetracker_data_processed_df = gaze_to_video_coords(eyetracker_data_processed_df, screen_w, screen_h, video_w, video_h)

    return eyetracker_data_processed_df  

def synchronize_all():
    """
    Synchronizes collected gazepoint users data with object detection results.
    """
    all_processed = []

    user_files = list_user_files(config.EYE_TRACKER_DATA_DIR)
    for fpath in user_files:
        user_id = extract_user_id_from_filename(fpath)
        try:
            video_info = create_video()
            df_proceed = transform_one_user(fpath, config.CSV_LOGO_DETECTIONS, config.SCREEN_WIDTH, config.SCREEN_HEIGHT, video_info.frame_width, video_info.frame_height)
            df_proceed = df_proceed.copy()
            df_proceed["user_id"] = user_id
            os.makedirs(config.EYE_TRACKER_DATA_SYNC_DIR, exist_ok=True)
            out_user = os.path.join(config.EYE_TRACKER_DATA_SYNC_DIR, f"user_{user_id}.csv")
            df_proceed.to_csv(out_user, sep=";")
            userInterface.print_info(f"User data saved: {out_user}")

            all_processed.append(df_proceed)
        except Exception as e:
            userInterface.print_error(f"An error occured while processing user data: {fpath}: {e}")

    if all_processed:
        df_all = pd.concat(all_processed, ignore_index=True)

        all_out = config.CSV_USERS_SYNC
        df_all.to_csv(all_out, sep=";")
        userInterface.print_info(f"Converted file for all users saved to: {all_out}")

