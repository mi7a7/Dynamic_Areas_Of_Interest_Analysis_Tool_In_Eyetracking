"""
delete_objects.py

Description:

This script deletes objects on specified video frames via console interface. 
User enters frames to be removed based on ../workspace/detections_output/detected_images directory.
Frames are removed from this directory as well as from the csv file. Video output is then updated.


Author: Milena Borczak 
Creation date: 20.05.2025
"""

from workspace.src.helpers.user_interface import UserInterface
from workspace import config
from pathlib import Path
import pandas as pd
from workspace.src.helpers.video_tools import mark_objects_on_video
import sys

userInterface = UserInterface()
csv_path = config.CSV_LOGO_DETECTIONS
img_path = Path(config.DETECTED_IMG_DIR)

def valid_frames_range(start: int, end: int, df: pd.DataFrame) -> bool:
    """
    This function validates frame range to be removed entered by user.
    Arguments:
    start - start index of frames range to remove
    end - end index of frames range to remove 
    """

    #Check if both values are greater than 0
    if start < 1 or end < 1:
        userInterface.print_error("Start and End indexes must be greater than 0. Please try again.")
        return False
    
    #Check if start <= end
    if start > end:
        userInterface.print_error("Start index must be lower than End index. Please try again.")
        return False
    
    #Check if frames exist in csv
    frame_numbers = df["Frame_Number"].values
    if start not in frame_numbers or end not in frame_numbers:
        userInterface.print_error(f"Start or end frame does not exist in CSV file: {csv_path}. Please try again.")
        return False

    return True

def valid_object_id(object_id: int, df: pd.DataFrame) -> bool:
    #Check if object exists in the data frame
    object_ids = df["Object_ID"].values
    if object_id not in object_ids:
        userInterface.print_error(f"Object with id {object_id} doesn't exist. Please try again.")
        return False
    return True

def remove_frames_img(start: int, end: int) -> None:
    """
    This function removes images with detections from /detections_output/detected_images directory based on frames to remove range. 
    Arguments:
    start - start index of frames range to remove
    end - end index of frames range to remove 
    """
    for file in img_path.iterdir():
        if file.is_file():
            # Check if filename matches frame_XXXXX pattern
            match = file.name.split('.')[0] 
            if match.startswith("frame_"):
                try:
                    index = int(match.replace("frame_", ""))
                    if start <= index <= end:
                        userInterface.print_info(f"Removing {file.name}")  
                        file.unlink()  # delete the file
                except ValueError:
                    continue  # skip files that don’t match pattern
    
    return None

def remove_frames_csv(start: int, end: int, object_id: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    This function removes detections from saved csv file. 
    Arguments:
    start - start index of frames range to remove
    end - end index of frames range to remove 
    """

    mask = (df["Frame_Number"] >= start) & (df["Frame_Number"] <= end) & (df["Object_ID"] == object_id)
    df.loc[mask, "Object_Detected"] = "No"
    df.loc[mask, "Detected_By"] = pd.NA
    df.loc[mask, "Object_ID"] = pd.NA
    df.loc[mask, "X1"] = pd.NA
    df.loc[mask, "X2"] = pd.NA
    df.loc[mask, "Y1"] = pd.NA
    df.loc[mask, "Y2"] = pd.NA
    df.loc[mask, "Confidence"] = pd.NA

    df.to_csv(csv_path, sep=';', index=False)

    userInterface.print_info(f"Rows from frame {start} to {end} succesfully updated in CSV file.")

    return df

def file_is_open(filepath: str) -> bool:
    """
    Helper function to check if the file is opened.
    Arguments:
    filepath - path to a file
    """

    try:
        with open(filepath, 'a'):
            pass
        return False  
    except OSError:
        return True  

def main():
    userInterface.print_info("DELETING OBJECTS MODULE, PRESS CTRL+C TO EXIT")
    try:
        df = pd.read_csv(csv_path, sep=';')
        while True:
        
            start_index = userInterface.get_input("Enter frame start index", cast_type=int)
            end_index = userInterface.get_input("Enter frame end index", cast_type=int)

            if not valid_frames_range(int(start_index), int(end_index), df):
                continue

            object_id = userInterface.get_input("Enter object id to be removed", cast_type=int)

            if not valid_object_id(int(object_id), df):
                continue

            confirmed = userInterface.get_confirmation(f"Do you want to remove object with id {object_id} from frames {start_index} to {end_index} ?")

            while file_is_open(csv_path):
                userInterface.get_confirmation("CSV file is locked. Did you close the file?")

            if confirmed:
                remove_frames_img(int(start_index), int(end_index))
                df = remove_frames_csv(int(start_index), int(end_index), int(object_id), df)
                mark_objects_on_video(df)
                
                userInterface.print_info("Objects removed succesfully!")
                repeat = userInterface.get_confirmation(f"Do you want to remove more objects ?")

                if not repeat:
                    sys.exit(0)

    except Exception as e:
        userInterface.print_error(f"An error occured with the message: {str(e)}")
        sys.exit(0)

if __name__ == "__main__":
    main()