# Dynamic Areas Of Interest Analysis Tool In Eyetracking 
## Narzędzie do analizy dynamicznego obszaru zainteresowania w technologii śledzenia ruchu gałek ocznych

### About

A tool developed as part of a master's thesis at Warsaw University of Technology.

The tool enables automated analysis of dynamic areas of interest (DAOI) in eye tracking research on video materials. It provides a complete end-to-end pipeline: from training a custom logo detection model (YOLOv8n) based on a single reference image, through eye tracking data collection (Gazepoint GP3), data synchronization, to generating a PDF report with visual attention statistics and a gaze heatmap video.

1. Project structure 

```plaintext
workspace/
│
├── src/                                # Source code of the application
|   |── train_model.py                  # Module for training YOLO model 
|   |── mark_objects.py                 # Module for marking detctions on video using trained YOLO model
│   |── add_objects.py                  # Module for manually adding objects on video
│   ├── delete_objects.py               # Module for deleting specified objects
│   ├── collect_tracker_data.py         # Module responsible for collecting data from eye-tracker
│   ├── generate_result.py              # Module responsible for generating final result for one user or for all
│   └── helpers/                        # Helper functions and classes
│       ├── gazepoint_api_client.py     # API client for Gazpoint software
│       ├── user_interface.py           # Module for interacting with the user via console
│       ├── user.py                     # Class which provides properties and methods for eye-tracker users 
│       ├── compute_module.py           # Computes data displayed in generated reports
│       ├── data_synchronizer.py        # Synchronizes data between eye tracker and logo detections
│       ├── heatmap_creator.py          # Creates heatmap based on eyetracker data
│       ├── pdf_generator.py            # Generates pdf result
│       ├── plot.py                     # Draws plots for reports
│       ├── utilities.py                # File handling helper functions used in modules
│       └── video_tools.py              # Helper class to load and store video data and mark objects on video
│
├── config.py                           # Global configuration of the application
│
├── model/                              # Trained YOLO model files directory
│
├── input/                              # Directory for input user's files
│   ├── video/                          # Directory for video input
│   └── logo/                           # Directory for logo input
│
├── output/                             # Directory for output files (csv, detection results, eye_tracker data and videos)
│
└── results/                            # Directory for final result files (heatmaps, PDFs, plots)
```

### Setup 

1. Install python (3.11.10)
https://www.python.org/downloads/

2. Open terminal (bash is preffered). 

If you are using VS code: Terminal (from the top bar) -> New Terminal 

3. Run following commands: 
- cd path/to/root/directory
- python -m venv venv (if venv folder doesnt exist)
- source venv/Scripts/activate 
- py pip install -r requirements.txt (if installed for the first time)
- py -m workspace.src.train_model
- py -m workspace.src.mark_objects
- py -m workspace.src.add_objects (optional)
- py -m workspace.src.delete_objects (optional)
- py -m workspace.src.collect_tracker_data
- py -m workspace.src.generate_result

## Contact 

Milena Borczak
www.linkedin.com/in/milena-borczak-590a81252 
