import os

#PATHS
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTIONS_OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "output")
RESULT_DIR = os.path.join(WORKSPACE_DIR, "result")
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")
HEATMAPS_DIR = os.path.join(RESULT_DIR, "heatmaps")
EYE_TRACKER_DATA_DIR = os.path.join(DETECTIONS_OUTPUT_DIR, "eye_tracker")
EYE_TRACKER_DATA_SYNC_DIR = os.path.join(EYE_TRACKER_DATA_DIR, "synchronized")
CSV_USERS = os.path.join(EYE_TRACKER_DATA_DIR, 'users.csv')
CSV_USERS_SYNC = os.path.join(EYE_TRACKER_DATA_SYNC_DIR, 'users.csv')
INPUT_VIDEO_DIR = os.path.join(WORKSPACE_DIR, "input/video")
INPUT_LOGO_DIR = os.path.join(WORKSPACE_DIR, "input/logo")
VIDEO_WITH_DETECTIONS = os.path.join(DETECTIONS_OUTPUT_DIR, 'video_with_detections.mp4')
CSV_LOGO_DETECTIONS = os.path.join(DETECTIONS_OUTPUT_DIR, 'logo_detections.csv')
DETECTED_IMG_DIR = os.path.join(DETECTIONS_OUTPUT_DIR, 'detected_images')
VLC_PATH = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
MODEL_DIR = os.path.join(WORKSPACE_DIR, 'model')
MODEL = os.path.join(MODEL_DIR, "runs/detect/logo_detection/weights/best.pt")

#PARAMETERS
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1200

#TRAINING MODEL PARAMETERS
COCO_ANNATOATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
NUMBER_0F_CLASSES =1  
CLASS_NAMES=["logo"] 
PRETRAINED_MODEL_NAME ='yolov8n.pt' 
EPOCHS=50
IMGSZ=960
BATCH=4
MODEL_NAME='logo_detection'
PATIENCE=5
MOSAIC =0.2
FLIPUD=0  
FLIPLR=0   
HSV_H=0   
HSV_S=0   
HSV_V=0   