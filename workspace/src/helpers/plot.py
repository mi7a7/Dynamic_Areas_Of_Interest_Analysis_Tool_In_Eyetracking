import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from workspace import config


def setup_matplotlib_style():

    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[
        "#774c60",
        "#576ca8",
        "#b75d69",
        "#1a1423",
        "#eacdc2",
        "#372549",
        "#74a57f"
    ])

    plt.rcParams.update({
        "axes.edgecolor": "#aaaaaa",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.7,
        "font.size": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "#fcfcfc",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.axisbelow": True 
    })



def plot_gaze_time_bar(object_stats_df, output_dir=config.PLOTS_DIR):
    df_sorted = object_stats_df.sort_values("gaze_time_s", ascending=False)
    x_ids = df_sorted["object_ID"].astype(str)

    plt.figure(figsize=(8, 4))
    plt.bar(x_ids, df_sorted["gaze_time_s"])
    plt.xlabel("Object ID")
    plt.ylabel("Gaze time (s)")
    plt.title("Gaze time per object")
    plt.tight_layout()

    path = os.path.join(output_dir, "bar_gaze_time.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_view_time_ratio_bar(object_stats_df, output_dir=config.PLOTS_DIR):
    df_sorted = object_stats_df.sort_values("view_time_ratio", ascending=False)
    x_ids = df_sorted["object_ID"].astype(str)
    y_vals = df_sorted["view_time_ratio"] * 100.0

    plt.figure(figsize=(8, 4))
    plt.bar(x_ids, y_vals)
    plt.xlabel("Object ID")
    plt.ylabel("View-time ratio (%)")
    plt.title("View-time ratio per object")
    plt.tight_layout()

    path = os.path.join(output_dir, "bar_view_time_ratio.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_seen_pie(object_stats_df, output_dir=config.PLOTS_DIR):
    seen = int(object_stats_df["consciously_seen"].sum())
    not_seen = int(len(object_stats_df) - seen)

    plt.figure(figsize=(5, 5))
    wedges, texts, autotexts = plt.pie(
        [seen, not_seen],
        autopct="%1.1f%%"
    )
    plt.title("Share of consciously seen objects")
    plt.legend(wedges, ["Seen", "Not seen"], title="Legend", loc="upper right")
    plt.tight_layout()

    path = os.path.join(output_dir, "pie_seen.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_duration_vs_gaze_scatter(object_stats_df, output_dir=config.PLOTS_DIR):
    plt.figure(figsize=(6, 6))
    plt.scatter(object_stats_df["object_duration_s"], object_stats_df["gaze_time_s"])

    for _, row in object_stats_df.iterrows():
        plt.text(
            row["object_duration_s"],
            row["gaze_time_s"] + 0.05,
            str(int(row["object_ID"])),
            ha='center', va='bottom', fontsize=9, color='black'
        )

    max_val = max(
        object_stats_df["object_duration_s"].max(),
        object_stats_df["gaze_time_s"].max()
    )
    plt.plot([0, max_val], [0, max_val], color='gray', linestyle='--', linewidth=1)

    plt.axis("equal")
    plt.xlabel("Object duration (s)")
    plt.ylabel("Gaze time (s)")
    plt.title("Object duration vs gaze time")
    plt.tight_layout()

    path = os.path.join(output_dir, "scatter_duration_vs_gaze.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def generate_all_plots_per_user(object_stats_df, output_dir=config.PLOTS_DIR):
    """
    Generuje wszystkie wykresy i zwraca dict z nazwami plików:
    {
      "bar_gaze_time": ".../bar_gaze_time.png",
      "bar_view_time_ratio": ".../bar_view_time_ratio.png",
      "pie_seen": ".../pie_seen.png",
      "scatter_duration_vs_gaze": ".../scatter_duration_vs_gaze.png",
    }
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_matplotlib_style()

    paths = {}
    paths["bar_gaze_time"] = plot_gaze_time_bar(object_stats_df, output_dir)
    paths["bar_view_time_ratio"] = plot_view_time_ratio_bar(object_stats_df, output_dir)
    paths["pie_seen"] = plot_seen_pie(object_stats_df, output_dir)
    paths["scatter_duration_vs_gaze"] = plot_duration_vs_gaze_scatter(object_stats_df, output_dir)
    return paths

def plot_seen_pie_all_users(object_stats_df, output_dir=config.PLOTS_DIR):
    seen = int(object_stats_df["consciously_seen"].sum())
    not_seen = int(len(object_stats_df) - seen)

    plt.figure(figsize=(5, 5))
    wedges, texts, autotexts = plt.pie(
        [seen, not_seen],
        autopct="%1.1f%%"
    )
    plt.title("Share of consciously seen objects")
    plt.legend(wedges, ["Seen", "Not seen"], title="Legend", loc="upper right")
    plt.tight_layout()

    path = os.path.join(output_dir, "pie_seen_all.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_avg_gaze_time_bar(object_stats_df, output_dir=config.PLOTS_DIR):
    df_sorted = object_stats_df.sort_values("mean_gaze_time_s", ascending=False)
    x_ids = df_sorted["object_ID"].astype(str)

    plt.figure(figsize=(8, 4))
    plt.bar(x_ids, df_sorted["mean_gaze_time_s"])
    plt.xlabel("Object ID")
    plt.ylabel("Gaze time (s)")
    plt.title("Mean gaze time per object")
    plt.tight_layout()

    path = os.path.join(output_dir, "bar_mean_gaze_time.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_avg_view_time_ratio_bar(object_stats_df, output_dir=config.PLOTS_DIR):
    df_sorted = object_stats_df.sort_values("mean_view_time_ratio", ascending=False)
    x_ids = df_sorted["object_ID"].astype(str)
    y_vals = df_sorted["mean_view_time_ratio"] * 100.0

    plt.figure(figsize=(8, 4))
    plt.bar(x_ids, y_vals)
    plt.xlabel("Object ID")
    plt.ylabel("Mean view-time ratio (%)")
    plt.title("Mean view-time ratio per object")
    plt.tight_layout()

    path = os.path.join(output_dir, "bar_mean_view_time_ratio.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path



def plot_duration_vs_avg_gaze_scatter(object_stats_df, output_dir=config.PLOTS_DIR):
    plt.figure(figsize=(6, 6))
    plt.scatter(object_stats_df["object_duration_s"], object_stats_df["mean_gaze_time_s"])

    for _, row in object_stats_df.iterrows():
        plt.text(
            row["object_duration_s"],
            row["mean_gaze_time_s"] + 0.05,
            str(int(row["object_ID"])),
            ha='center', va='bottom', fontsize=9, color='black'
        )

    max_val = max(
        object_stats_df["object_duration_s"].max(),
        object_stats_df["mean_gaze_time_s"].max()
    )
    plt.plot([0, max_val], [0, max_val], color='gray', linestyle='--', linewidth=1)

    plt.axis("equal")
    plt.xlabel("Object duration (s)")
    plt.ylabel("Mean gaze time (s)")
    plt.title("Object duration vs mean gaze time")
    plt.tight_layout()

    path = os.path.join(output_dir, "scatter_duration_vs_mean_gaze.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path

def plot_genders_pie(users_df, output_dir=config.PLOTS_DIR):
 
    gender_counts = users_df["gender"].astype(str).str.strip().value_counts(dropna=False)
    labels = gender_counts.index.tolist()
    sizes = gender_counts.values.tolist()

    plt.figure(figsize=(5, 5))
    wedges, texts, autotexts = plt.pie(
        sizes,
        autopct="%1.1f%%",
    )

    plt.title("Users gender distribution")
    plt.legend(wedges, labels, title="Legend", loc="upper right")
    plt.tight_layout()

    path = os.path.join(output_dir, "pie_gender_distribution.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path

def plot_seen_percentage_by_gender_bar(object_stats_all_users, output_dir=config.PLOTS_DIR):

    per_user = (
        object_stats_all_users.groupby(["user_id", "gender"])["consciously_seen"]
        .mean()
        .reset_index()
    )
    per_user["seen_pct"] = per_user["consciously_seen"] * 100

    avg_seen_by_gender = per_user.groupby("gender")["seen_pct"].mean().sort_values(ascending=False)

    plt.figure(figsize=(6, 4))
    bars = plt.bar(avg_seen_by_gender.index, avg_seen_by_gender.values)
    plt.ylabel("Average % of logos consciously seen")
    plt.title("Average percentage of logos seen by gender")

    for bar, val in zip(bars, avg_seen_by_gender.values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "bar_seen_pct_by_gender.png")
    plt.savefig(path, dpi=200)
    plt.close()

    return path

def plot_view_ratio_by_gender_boxplot(object_stats_all_users, output_dir=config.PLOTS_DIR):
    plt.figure(figsize=(6, 4))
    object_stats_all_users.boxplot(column="view_time_ratio", by="gender",
                   grid=True, patch_artist=True,)

    plt.suptitle("")
    plt.title("Distribution of view-time ratio by gender")
    plt.xlabel("Gender")
    plt.ylabel("View-time ratio")
    plt.tight_layout()

    path = os.path.join(output_dir, "box_view_ratio_by_gender.png")
    plt.savefig(path, dpi=200)
    plt.close()

    return path



def plot_age_hist(users_df, output_dir=config.PLOTS_DIR):
    """
    Histogram rozkładu wieku użytkowników.
    """

    ages = pd.to_numeric(users_df["age"], errors="coerce").dropna()
    bins = [0, 25, 35, 50, 120]
    labels = ["<25", "25–35", "35–50", "50+"]

    plt.figure(figsize=(7, 4))
    plt.hist(ages, bins=bins, edgecolor="white")

    centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
    plt.xticks(centers, labels, rotation=0)

    plt.xlabel("Age [years]")
    plt.ylabel("Count")
    plt.title("Age distribution (histogram)")
    plt.tight_layout()

    path = os.path.join(output_dir, "hist_age_distribution.png")
    plt.savefig(path, dpi=200)
    plt.close()

    return path


def plot_seen_percentage_by_age_group_bar(object_stats_all_users,
                                          output_dir=config.PLOTS_DIR):
    """
    Wykres słupkowy:
    średni % świadomie zauważonych logo w grupach wiekowych.
    """

    df = object_stats_all_users.copy()

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    bins = [0, 25, 35, 50, 120]
    labels = ["<25", "25–35", "35–50", "50+"]

    df["age_group"] = pd.cut(
        df["age"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False
    )

    per_user = (
        df.groupby(["user_id", "age_group"])["consciously_seen"]
        .mean()
        .reset_index()
    )

    per_user["seen_pct"] = per_user["consciously_seen"] * 100.0

    avg_seen_by_age_group = (
        per_user
        .dropna(subset=["age_group"])
        .groupby("age_group")["seen_pct"]
        .mean()
        .reindex(labels)
    )

    plt.figure(figsize=(7, 4))
    bars = plt.bar(avg_seen_by_age_group.index.astype(str),
                   avg_seen_by_age_group.values)

    plt.ylabel("Average % of logos consciously seen")
    plt.xlabel("Age group")
    plt.title("Average percentage of logos seen by age group")

    for bar, val in zip(bars, avg_seen_by_age_group.values):
        if pd.isna(val):
            continue
        plt.text(
            bar.get_x() + bar.get_width()/2,
            val + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()

    path = os.path.join(output_dir, "bar_seen_pct_by_age_group.png")
    plt.savefig(path, dpi=200)
    plt.close()

    return path

def plot_age_regression(per_user_age, slope, intercept,
                        output_dir=config.PLOTS_DIR):

    os.makedirs(output_dir, exist_ok=True)

    x = per_user_age["age"].values
    y = per_user_age["seen_pct"].values

    plt.figure(figsize=(6,4))
    plt.scatter(x, y, s=30, alpha=0.8)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Age [years]")
    plt.ylabel("% of logos consciously seen")
    plt.title("Correlation between age and attention")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "age_vs_attention_regression.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path



def generate_all_plots_all_users(object_stats_df, object_stats_all_users, users_info_df, per_user_age, slope, intercept, output_dir=config.PLOTS_DIR):
    """
    Generuje wszystkie wykresy i zwraca dict z nazwami plików:
    {
      "bar_gaze_time": ".../bar_gaze_time.png",
      "bar_view_time_ratio": ".../bar_view_time_ratio.png",
      "pie_seen": ".../pie_seen.png",
      "scatter_duration_vs_gaze": ".../scatter_duration_vs_gaze.png",
    }
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_matplotlib_style()

    paths = {}
    paths["bar_gaze_time_all"] = plot_avg_gaze_time_bar(object_stats_df, output_dir)
    paths["bar_view_time_ratio_all"] = plot_avg_view_time_ratio_bar(object_stats_df, output_dir)
    paths["pie_seen_all"] = plot_seen_pie_all_users(object_stats_all_users, output_dir)
    paths["scatter_duration_vs_gaze_all"] = plot_duration_vs_avg_gaze_scatter(object_stats_df, output_dir)
    paths["pie_gender_distribution"] = plot_genders_pie(users_info_df, output_dir)
    paths["bar_seen_pct_by_gender"] = plot_seen_percentage_by_gender_bar(object_stats_all_users, output_dir)
    paths["box_view_ratio_by_gender"] = plot_view_ratio_by_gender_boxplot(object_stats_all_users, output_dir)
    paths["hist_age_distribution"]= plot_age_hist(users_info_df, output_dir)
    paths["bar_seen_pct_by_age_group"]= plot_seen_percentage_by_age_group_bar(object_stats_all_users, output_dir)
    paths["age_vs_attention_regression"]= plot_age_regression(per_user_age, slope, intercept)

    return paths


