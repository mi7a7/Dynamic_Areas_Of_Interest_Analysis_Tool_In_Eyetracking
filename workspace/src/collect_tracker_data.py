"""
collect_tracker_data.py

Description:
Collect eye-tracker gaze points on video using Gazepoint Api Client (gazepoint_api_client.py).

Author: Milena Borczak 
Creation date: 17.10.2025
"""

import sys
from workspace.src.helpers.gazepoint_api_client import OpenGazeTracker
import os
import time
import subprocess
from workspace.src.helpers.video_tools import create_video
from workspace import config
from workspace.src.helpers.user_interface import UserInterface
from workspace.src.helpers.user import Gender, User
import pandas as pd

userInterface = UserInterface()
video_info = create_video()
os.makedirs(config.EYE_TRACKER_DATA_DIR, exist_ok=True)

def record_data(user_id: int) -> None:

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    fname = os.path.join(
        config.EYE_TRACKER_DATA_DIR,
        f"user_{user_id}_{timestamp}.csv"
    )

    # Open the connection to the tracker.
    tracker = OpenGazeTracker(ip='127.0.0.1',logfile=fname, debug=False)

    # Calibrate the tracker.
    tracker.calibrate()

    # Start recording data.
    tracker.start_recording()
    tracker.log("START=%d" % (round(time.time()*1000)))

    screen_width = config.SCREEN_WIDTH
    screen_height = config.SCREEN_HEIGHT

    vlc_path = config.VLC_PATH
    video_path =  video_info.video_path
    video_width = video_info.frame_width
    video_height = video_info.frame_height

    vlc_args = [
        vlc_path,
        video_path,
        "--start-paused",
        "--video-on-top",
        "--fullscreen",
        "--play-and-exit",
        "--no-video-title-show"
    ]

    # Add --no-autoscale only if video doesn't exceed screen resolution
    if video_width <= screen_width and video_height <= screen_height:
        vlc_args.append("--no-autoscale")

    # Run VLC
    vlc_proc = subprocess.Popen(vlc_args)
    vlc_proc.wait() 

    # Stop recording.
    tracker.log("STOP=%d" % (round(time.time()*1000)))
    tracker.stop_recording()

    # Close the connection.
    tracker.close()
    vlc_proc.kill()

def main():
    userInterface.print_info("COLLECT TRACKER DATA MODULE, PRESS CTRL+C TO EXIT")
    while userInterface.get_confirmation("Do you want to record new data entry?"):
        try:
            user = User()
            user.first_name = userInterface.get_input("Enter the first name", str)
            user.last_name = userInterface.get_input("Enter the last name", str)
            user.age = int(userInterface.get_input("Enter age", int))
            user.gender = Gender[userInterface.get_selection("Select gender from the list: ", [g.name for g in Gender])]
            user_id = user.save_to_csv()
            userInterface.print_info(f"User with id {user_id} succesfully created.")
            if userInterface.get_confirmation("Do you want to start eye-tracking?"):
                record_data(user_id)

        except Exception as e:
            userInterface.print_error(f"An error occured while recording new data: {str(e)}")

if __name__ == "__main__":
    main()

