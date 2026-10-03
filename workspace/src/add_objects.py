"""
add_objects.py

Description:
User interface for manually adding object markers to video using an auto tracker.

Author: Milena Borczak 
Creation date: 20.05.2025
"""

import cv2
from workspace import config
from workspace.src.helpers.user_interface import UserInterface
import pandas as pd
from workspace.src.helpers.video_tools import mark_objects_on_video

userInterface = UserInterface()

folder_path_detections_output = config.DETECTIONS_OUTPUT_DIR
csv_path = config.CSV_LOGO_DETECTIONS

def draw_rectangle(frame, bbox) -> None:
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)


def findMaxObjectId() -> int:
    """
    Find max object id in the loaded data frame.

    Returns:
        int: max object id.
    """
    if not csv_path:
        raise FileNotFoundError("Logo detections csv file not found.")
    
    existing_df = pd.read_csv(csv_path, sep=';')
    if not existing_df.empty and "Object_ID" in existing_df.columns:
        max_existing_id = existing_df["Object_ID"].max()
        max_existing_id = int(max_existing_id) if not pd.isna(max_existing_id) else 0
    else:
        max_existing_id = 0

    return max_existing_id

def process_frame(frame, current_frame, fps, tracking_data, written_frames, trackers):
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    frame_start_time = (current_frame-1) / fps
    frame_end_time = current_frame / fps
    new_trackers = []

    for tracker, bbox, obj_id in trackers:
        ok, new_bbox = tracker.update(frame)
        if ok:
            draw_rectangle(frame, new_bbox)
            new_trackers.append((tracker, new_bbox, obj_id))

            if (current_frame, obj_id) not in written_frames:
                tracking_data.append([
                    current_frame,
                    frame_width,
                    frame_height,
                    frame_start_time,
                    frame_end_time,
                    "Yes",
                    "KCF",
                    obj_id,
                    int(new_bbox[0]),
                    int(new_bbox[1]),
                    int(new_bbox[0] + new_bbox[2]),
                    int(new_bbox[1] + new_bbox[3]),
                    ""
                ])
                written_frames.add((current_frame, obj_id))

    return new_trackers

def process_video(maxObjectId: int) -> pd.DataFrame:
    """
    Open input video and provide interface for a user to track new objects.
    Keyboard commands:
    q - quit 
    n - next frame 
    b - previous frame
    f - go 10 frames ahead
    r - go 10 frames back
    t - select object
    d - keep the tracker

    Returns:
        DataFrame: Data frame with existing and new objects marked on the video. 
    """

    video_path = config.VIDEO_WITH_DETECTIONS

    if not video_path:
        raise FileNotFoundError("Video with detections path not found.")

    cap = cv2.VideoCapture(video_path)

    object_id = maxObjectId + 1 

    if not cap.isOpened():
        raise Exception("Video with detecions cannot be opened.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 1
    tracking_data = []  #  [Frame_Number, Resolution_Width, Resolution_Height, Video_Time, Object_Detected, Detected_By, X1, Y1, X2, Y2, Confidence]
    written_frames = set()  # pairs of (frame, obj_id) to avoid duplicates
    trackers = []  # (tracker, bbox, object_id)

    userInterface.print_info("Use the keyboard to navigate: " \
    """    
    q - quit 
    n - next frame 
    b - previous frame
    f - go 10 frames ahead
    r - go 10 frames back
    t - select object
    d - stop tracking
    """)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
        ret, frame = cap.read()
        if not ret:
            break #end of the video

        cv2.putText(frame, f"Frame: {current_frame}/{frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        trackers = process_frame(frame, current_frame, fps, tracking_data, written_frames, trackers)

        cv2.imshow("Video", frame)
        key = cv2.waitKey(0) & 0xFF

        # q - exit the video
        if key == ord('q'):
            break
        
        # n - one frame ahead
        elif key == ord('n'):
            current_frame = min(current_frame + 1, frame_count)
        # b - one frame back
        elif key == ord('b'):
            current_frame = max(current_frame - 1, 1)
        # f - 10 frames ahead
        elif key == ord('f'):
            if len(trackers) == 0:
                current_frame = min(current_frame + 10, frame_count)
            else:
                for _ in range(10):
                    if current_frame < frame_count:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
                        ret, frame = cap.read()
                        if not ret:
                            break
                        trackers = process_frame(frame, current_frame, fps, tracking_data, written_frames, trackers)
                        current_frame += 1
        # r - 10 frames back
        elif key == ord('r'):
            if len(trackers) == 0:
                current_frame = max(current_frame - 10, 1)
            else:
                for _ in range(10):
                    if current_frame > 1:
                        current_frame -= 1
                        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
                        ret, frame = cap.read()
                        if not ret:
                            break
                        trackers = process_frame(frame, current_frame, fps, tracking_data, written_frames, trackers)
        # t - tracker - select the object with the mouse, press enter
        elif key == ord('t'):
            tracker = cv2.TrackerKCF_create()
            bbox = cv2.selectROI("Video", frame, False)
            if bbox[2] > 0 and bbox[3] > 0:
                tracker.init(frame, bbox)
                trackers.append((tracker, bbox, object_id))
                object_id += 1
        # d - Keep the tracker, then it won't save the objects on the frame we're stopping at.
        elif key == ord('d'):
            # Remove data from the current frame from tracking_data so that the last frame is not saved.
            tracking_data = [row for row in tracking_data if row[0] != current_frame]
            written_frames = {(frm, oid) for (frm, oid) in written_frames if frm != current_frame}
            trackers = []
    
    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(tracking_data, columns=[
        "Frame_Number",
        "Resolution_Width",
        "Resolution_Height",
        "Frame_Start_Time",
        "Frame_End_Time",
        "Object_Detected",
        "Detected_By",
        "Object_ID",
        "X1", "Y1", "X2", "Y2",
        "Confidence"
    ])

    existing_df = pd.read_csv(csv_path, sep=';')

    # Remove ‘No’ entries from the existing dataframe if there are ‘Yes’ entries for the same frames in the new one.
    frames_with_yes = df[df["Object_Detected"] == "Yes"]["Frame_Number"].unique()
    existing_df = existing_df[~((existing_df["Frame_Number"].isin(frames_with_yes)) & (existing_df["Object_Detected"] == "No"))]

    # Only now combine the data
    updated_df = pd.concat([existing_df, df], ignore_index=True)
    updated_df = updated_df.sort_values(["Frame_Number", "Object_ID"])

    return updated_df


def main():
    userInterface.print_info("ADDING OBJECTS MODULE, PRESS CTRL+C TO EXIT")
    try:
        updated_df = process_video(findMaxObjectId())
        updated_df.to_csv(csv_path, sep=';', index=False)
        mark_objects_on_video(updated_df)
    except FileNotFoundError as e:
        userInterface.print_info("FileNotFoundError: " + str(e))
    except Exception as e:
        userInterface.print_info("Exception: " + str(e))

if __name__ == "__main__":
    main()
