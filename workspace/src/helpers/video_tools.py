from dataclasses import dataclass
import os
import cv2
from workspace import config
from workspace.src.helpers.user_interface import UserInterface
import pandas as pd
from pathlib import Path

userInterface = UserInterface()

@dataclass
class VideoInfo:
    video_path: str = ""
    fps: float = 0.0
    frame_width: int = 0
    frame_height: int = 0

def create_video() -> VideoInfo:
    """Searches for the video and return VideoInfo object.

    Args:
        None

    Returns:
        VideoInfo object
    """

    video_info = VideoInfo()

    folder_path_video = config.INPUT_VIDEO_DIR
    if not folder_path_video:
        raise FileNotFoundError("Video path not found")
    video_files = [f for f in os.listdir(folder_path_video) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        raise FileNotFoundError("Video not found")
    
    video_info.video_path = os.path.join(folder_path_video, video_files[0])

    cap = cv2.VideoCapture(video_info.video_path)
    video_info.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_info.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_info.fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return video_info

def mark_objects_on_video(df: pd.DataFrame) -> None:
    """
    Mark existing and new detections on video and save it to the output folder.
    Also saves full frames with detected objects as images.
    """
    
    video_info = create_video()
    userInterface.print_info(f"Creating output video... Please wait.")

    detected_images_folder = config.DETECTED_IMG_DIR
    os.makedirs(detected_images_folder, exist_ok=True)

    input_video = video_info.video_path
    output_video = config.VIDEO_WITH_DETECTIONS

    existing_video_path = Path(output_video)

    if existing_video_path.exists() and existing_video_path.is_file():
        existing_video_path.unlink() 

    capture = cv2.VideoCapture(input_video)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    frame_idx = 0

    detections_per_frame = df[df["Object_Detected"] == "Yes"].groupby("Frame_Number")

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame_idx += 1
        any_detected = False

        if frame_idx in detections_per_frame.groups:
            detections = detections_per_frame.get_group(frame_idx)
            any_detected = True

            for _, row in detections.iterrows():
                x1, y1, x2, y2 = int(row["X1"]), int(row["Y1"]), int(row["X2"]), int(row["Y2"])
                object_id = row["Object_ID"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID {int(object_id)}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

        if any_detected:
            filename = f"frame_{frame_idx}.jpg"
            filepath = os.path.join(detected_images_folder, filename)
            cv2.imwrite(filepath, frame)

    capture.release()
    out.release()

    userInterface.print_info("New output video saved to file: " + output_video)
    userInterface.print_info("Full-frame detection images saved in: " + detected_images_folder)