"""
generate_result.py

Description:
Creates final result of the application. It combines data from eye-tracker with object detections results. 
Final report is in PDF format and it contains tables, plots and detailed analysis. 
It also generates heatmap based on eye-tracker data.

Author: Milena Borczak 
Creation date: 11.01.2026
"""

import os
import traceback
import pandas as pd
from workspace.src.helpers.compute_module import aggregate_object_stats_all_users, compute_age_attention_correlation_per_user, compute_age_regression, compute_logo_attention_per_user, get_object_stats_all_users
from workspace.src.helpers.data_synchronizer import synchronize_all
from workspace.src.helpers.heatmap_creator import render_video_with_heatmap
from workspace.src.helpers.pdf_generator import generate_pdf_report_all_users, generate_pdf_report_per_user
from workspace.src.helpers.plot import generate_all_plots_all_users, generate_all_plots_per_user
from workspace.src.helpers.user_interface import UserInterface
from workspace import config
from workspace.src.helpers.utilities import extract_user_id_from_filename, get_middle_frame_paths_per_object, list_user_files
from workspace.src.helpers.video_tools import create_video

userInterface = UserInterface()

def get_input() -> str:
    user_files = list_user_files(config.EYE_TRACKER_DATA_SYNC_DIR)
    max_id = len(user_files)
    user_input = userInterface.get_input(f"Enter user_id (1–{max_id} or ALL): ")

    if user_input.isdigit():
        user_num = int(user_input)
        if 1 <= user_num <= max_id:
            return user_input
        else:
            userInterface.print_error(f"User id out of range. It must be in 1–{max_id}.")
            return ""
    elif user_input.upper() == "ALL":
        return "ALL"
    else:
        userInterface.print_error(f"Wrong input. Enter a number between 1–{max_id} or ALL.")
        return ""

def main():
    userInterface.print_info("GENERATE RESULT MODULE, PRESS CTRL+C TO EXIT")
    try:
        synchronize_all()
        os.makedirs(config.RESULT_DIR, exist_ok=True)
        os.makedirs(config.HEATMAPS_DIR, exist_ok=True)
        while True:

            USER_ID = get_input()

            if USER_ID == "": break

            video_info = create_video()
            eyetracker_data_processed_df = pd.DataFrame()
            objects_data_df = pd.read_csv(config.CSV_LOGO_DETECTIONS, sep=';')
            middle_frames_paths = get_middle_frame_paths_per_object(
                objects_df=objects_data_df,
                frames_dir=config.DETECTED_IMG_DIR
            )
            users_info_df = pd.read_csv(config.CSV_USERS, sep=';')
            
            userInterface.print_info("Generating PDF reports. Please wait.")

            if USER_ID == "ALL":
                eyetracker_data_processed_df = pd.read_csv(config.CSV_USERS_SYNC, sep=';')
                object_stats_all_users = get_object_stats_all_users(list_user_files(config.EYE_TRACKER_DATA_SYNC_DIR), objects_data_df, users_info_df)
                per_object_agg = aggregate_object_stats_all_users(object_stats_all_users)
                
                per_user_age = compute_age_attention_correlation_per_user(
                    object_stats_all_users
                )
                slope, intercept, r2 = compute_age_regression(per_user_age)
                generate_all_plots_all_users(
                    per_object_agg, 
                    object_stats_all_users,
                    users_info_df, 
                    per_user_age, 
                    slope, 
                    intercept)
                generate_pdf_report_all_users(
                    object_stats_all_users,
                    objects_data_df,
                    users_info_df = users_info_df,
                    user_id="ALL",
                    output_dir=config.RESULT_DIR,
                    thumbnails_of_objects = middle_frames_paths,
                    thumbs_per_row = 2
                )

            else:
                eyetracker_data_processed_path = os.path.join(config.EYE_TRACKER_DATA_SYNC_DIR, f"user_{USER_ID}.csv")
                eyetracker_data_processed_df = pd.read_csv(eyetracker_data_processed_path, sep=";")
                object_stats_df = compute_logo_attention_per_user(objects_data_df, eyetracker_data_processed_df, USER_ID)
                generate_all_plots_per_user(object_stats_df)
                generate_pdf_report_per_user(
                    object_stats_df,
                    objects_data_df,
                    users_info = users_info_df,
                    user_id=USER_ID,
                    thumbnails_of_objects = middle_frames_paths,
                    thumbs_per_row = 2
                )
           
            userInterface.print_info("Rendering heatmap video. Please wait.")
           
            render_video_with_heatmap(
                input_video_path=video_info.video_path,
                eyetracker_data_df=eyetracker_data_processed_df,
                output_video_path=os.path.join(config.HEATMAPS_DIR, f"user_{USER_ID}.mp4"),
                window_size=30,  
                grid_scale=0.4, 
                blur_ksize=31,
                alpha=1,
                gamma=0.4,
            )

    except Exception as e:
        userInterface.print_error("Unexpected exception occured while running the application: "+ str(e))
        traceback.print_exc() 
        

if __name__ == "__main__":
    main()
