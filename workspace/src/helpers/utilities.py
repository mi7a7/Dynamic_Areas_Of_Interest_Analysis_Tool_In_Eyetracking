import os

import numpy as np
import pandas as pd


def list_user_files(folder):
    users_files = []
    for name in os.listdir(folder):
        if name.startswith("user_") and name.endswith(".csv") and name != "users.csv":
            users_files.append(os.path.join(folder, name))
    users_files.sort()
    return users_files

def extract_user_id_from_filename(filepath):
    base = os.path.basename(filepath)
    if base.startswith("user_"):
        rest = base[len("user_"):]
        if "_" in rest:
            return rest.split("_", 1)[0]
        else:
            return rest.split(".", 1)[0]
    return os.path.splitext(base)[0]

def get_middle_frame_paths_per_object(objects_df: pd.DataFrame,
                                      frames_dir: str,
                                      object_id_col="Object_ID",
                                      frame_number_col="Frame_Number") -> dict:
    """
    For each object, it selects the middle frame and then maps it to a JPG file.
    Returns a dictionary: {Object_ID: full_path_to_frame_N.jpg}
    """
    middle_frame_paths = {}

    for obj_id, group in objects_df.groupby(object_id_col):
        frames = np.sort(group[frame_number_col].astype(int).unique())
        mid_idx = len(frames) // 2
        middle_frame = int(frames[mid_idx])

        filename = f"frame_{middle_frame}.jpg"
        path = os.path.join(frames_dir, filename)
        middle_frame_paths[int(obj_id)] = path

    return middle_frame_paths