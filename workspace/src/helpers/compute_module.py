import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

from workspace.src.helpers.utilities import extract_user_id_from_filename

def compute_logo_attention_per_user(objects_df: pd.DataFrame, eyetracker_df: pd.DataFrame,
                                    user_id: int,
                           frame_col="Frame_Number",
                           gx_col="GAZE_X_VIDEO", gy_col="GAZE_Y_VIDEO",
                           conscious_threshold: float = 0.2):
    """
    Combines objects and eye tracker data.
    Returns a table:
        "object_ID"
        "object_duration_s"
        "gaze_time_s"
        "view_time_ratio"
        "consciously_seen"
        "user_id"
    """
    results = []

    for obj_id, object_rows in objects_df.groupby("Object_ID"):
        start = float(object_rows["Frame_Start_Time"].min())
        end   = float(object_rows["Frame_End_Time"].max())
        object_duration_s = end - start
        

        frame_duration = (eyetracker_df.groupby(frame_col)["FRAME_DURATION"].first().to_dict())

        frames_for_logo = object_rows[frame_col]
        gaze_subset = eyetracker_df.merge(frames_for_logo, on=frame_col, how="inner")


        hit_frames = set()  

        for _, row in gaze_subset.iterrows():
            gx, gy, fn = row[gx_col], row[gy_col], row[frame_col]
            if pd.isna(gx) or pd.isna(gy):
                continue

            bboxes = object_rows.loc[object_rows[frame_col] == fn, ["X1", "Y1", "X2", "Y2"]].values

            for (x1, y1, x2, y2) in bboxes:

                if (x1 <= gx <= x2) and (y1 <= gy <= y2):
                    hit_frames.add(fn)  
                    break  


        gaze_time_s = float(sum(frame_duration.get(fn, 0.0) for fn in hit_frames))

        consciously_seen = gaze_time_s > conscious_threshold 
        view_time_ratio = (gaze_time_s / object_duration_s) if object_duration_s > 0 else 0.0

        results.append({
            "object_ID": obj_id,
            "object_duration_s": object_duration_s,
            "gaze_time_s": gaze_time_s,
            "view_time_ratio": round(view_time_ratio, 3),
            "consciously_seen": consciously_seen,
            "user_id": user_id
        })

    return pd.DataFrame(results)

def get_object_stats_all_users(user_files, objects_data_df: pd.DataFrame, users_info_df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []

    for fpath in user_files:
        uid = extract_user_id_from_filename(fpath)
        et_df = pd.read_csv(fpath, sep=';')
        per_user_df = compute_logo_attention_per_user(objects_data_df, et_df, uid)

        all_rows.append(per_user_df)

    object_stats_all_users = pd.concat(all_rows, ignore_index=True)

    object_stats_all_users['user_id']= object_stats_all_users['user_id'].astype('Int64')
    object_stats_all_users = pd.merge(object_stats_all_users, users_info_df, on="user_id", how="left")

    return object_stats_all_users

def aggregate_object_stats_all_users(object_stats_all_users: pd.DataFrame) -> pd.DataFrame:
 
    df = object_stats_all_users.copy()

    df["consciously_seen_num"] = df["consciously_seen"].astype(int)

    df = df.groupby("object_ID", as_index=False).agg(
        object_duration_s=("object_duration_s", "first"),    
        mean_gaze_time_s=("gaze_time_s", "mean"),
        mean_view_time_ratio=("view_time_ratio", "mean"),
        n_users=("user_id", "nunique"),
        consciously_seen_pct=("consciously_seen_num", lambda s: 100.0 * s.mean())
    )

    df["object_duration_s"] = df["object_duration_s"].round(3)
    df["mean_gaze_time_s"] = df["mean_gaze_time_s"].round(3)
    df["mean_view_time_ratio"] = df["mean_view_time_ratio"].round(3)
    df["consciously_seen_pct"] = df["consciously_seen_pct"].round(1)

    return df

def summarize_logo_attention_all_users(object_stats_all_users: pd.DataFrame, objects_data_df: pd.DataFrame):

    video_duration_s = objects_data_df["Frame_End_Time"].max() - objects_data_df["Frame_Start_Time"].min()
    
    total_objects = int(object_stats_all_users["object_ID"].nunique())

    object_stats_all_users["seen_num"] = object_stats_all_users["consciously_seen"].astype(int)

    per_user = object_stats_all_users.groupby("user_id").agg(
        seen_count=("seen_num", "sum"),
        total_count=("object_ID", "count")
    ).reset_index()

    avg_seen_count_per_user = float(per_user["seen_count"].mean()) if len(per_user) else 0.0

    per_user["seen_pct"] = np.where(
        per_user["total_count"] > 0,
        100.0 * per_user["seen_count"] / per_user["total_count"],
        0.0
    )
    avg_seen_pct_per_user = float(per_user["seen_pct"].mean()) if len(per_user) else 0.0

    avg_seen_pct_per_user = float(per_user["seen_pct"].mean()) if len(per_user) else 0.0

    seen_only = object_stats_all_users[object_stats_all_users["seen_num"] == 1]
    mean_gaze_all_users  = float(seen_only["gaze_time_s"].mean())      if not seen_only.empty else 0.0
    mean_ratio_all_users = float(seen_only["view_time_ratio"].mean())  if not seen_only.empty else 0.0

    stats_summary = {
        "Video duration [s]": round(float(video_duration_s), 3),
        "Total objects": total_objects,
        "Avg consciously seen logos per user": round(avg_seen_count_per_user, 2),
        "Avg % of logos seen per user": round(avg_seen_pct_per_user, 2),
        "Mean gaze time [s] (seen only, all users)": round(mean_gaze_all_users, 3),
        "Mean view-time ratio (seen only, all users)": round(mean_ratio_all_users, 3),
    }

    return stats_summary

def generate_user_stats(users_df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    n_users = int(len(users_df))
    rows.append(["Number of users", n_users])

    mean_age = round(users_df['age'].mean(), 1)
    min_age  = int(users_df['age'].min())
    max_age  = int(users_df['age'].max())

    rows.append(["Mean age", mean_age])
    rows.append(["Minimum age", min_age])
    rows.append(["Maximum age", max_age])

    gender_counts = users_df["gender"].astype(str).str.strip().value_counts(dropna=False)
    total = gender_counts.sum()

    for g, v in gender_counts.items():
        percent = v / total * 100
        rows.append([f"Gender: {g}", f"{v} ({percent:.1f}%)"])

    return pd.DataFrame(rows, columns=["Metric", "Value"])

def compute_age_attention_correlation_per_user(object_stats_all_users):
   
    df = object_stats_all_users.copy()

    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    per_user = (
        df.groupby("user_id")["consciously_seen"]
        .mean()
        .reset_index()
    )
    per_user["seen_pct"] = per_user["consciously_seen"] * 100

    per_user = per_user.merge(df[["user_id", "age"]].drop_duplicates(), on="user_id", how="left")

    return per_user

def compute_age_correlation_stats(per_user_age: pd.DataFrame):
   
    ages = per_user_age["age"].astype(float)
    seen = per_user_age["seen_pct"].astype(float)

    pearson_r, pearson_p = pearsonr(ages, seen)
    spearman_rho, spearman_p = spearmanr(ages, seen)

    return pearson_r, pearson_p, spearman_rho, spearman_p

def compute_age_regression(per_user_age):
    X = per_user_age["age"].values.reshape(-1, 1)
    y = per_user_age["seen_pct"].values

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)

    return slope, intercept, r2