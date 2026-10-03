"""
train_model.py

Description:

This module trains pretrained YOLO model on input logo image so it can be used to mark objects on the video automatically.
It uses COCO dataset.
It can traing the model on both GPU and CPU. Training settings can be adjusted in config.py file. 
ModelTrainer class provides overlay_images method which adds logo transformations. 
Modules uses user_interface class to print some info and get input in the console. 

Author: Milena Borczak 
Creation date: 15.02.2025
"""


import os
import sys
import time
import traceback
from pycocotools.coco import COCO
import requests
import zipfile
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO 
from workspace import config
from workspace.src.helpers.user_interface import UserInterface
import torch

class ModelTrainer:

    def __init__(self):
        self.user_interface = UserInterface()

    def start_training(self) -> None:
        model = YOLO(config.PRETRAINED_MODEL_NAME) 

        device = 0 if torch.cuda.is_available() else "cpu"
        self.user_interface.print_info(f"Training on {'GPU' if device == 0 else 'CPU'}")

        print("CONFIG FILE:", config.__file__)
        print("CONFIG VALUES:", config.PRETRAINED_MODEL_NAME, config.IMGSZ, config.BATCH, config.MODEL_NAME)


        model.train(
            data=self.dataset_config,
            epochs=config.EPOCHS,
            imgsz=config.IMGSZ,
            batch=config.BATCH,
            name=config.MODEL_NAME,
            device=device,
            patience=config.PATIENCE,
            mosaic = config.MOSAIC,
            flipud=config.FLIPUD,  
            fliplr=config.FLIPLR,   
            hsv_h=config.HSV_H,   
            hsv_s=config.HSV_S,   
            hsv_v=config.HSV_V,  
        )


    def gpu_ready(self) -> bool:
        self.user_interface.print_info("Graphic card: " + str(torch.cuda.get_device_name(0)))  
        self.user_interface.print_info("PyTorch version: " + str(torch.__version__))
        self.user_interface.print_info("CUDA version: " + str(torch.version.cuda))  
        self.user_interface.print_info("GPU memory used: " + str(torch.cuda.memory_allocated())) 
        self.user_interface.print_info("Reserved GPU memory: " + str(torch.cuda.memory_reserved())) 
        return torch.cuda.is_available()

    def prepare_dataset(self, download: bool = True) -> None:

        self.__create_train_model_directories()

        if download is True:
            self.__download_annotations()
            self.__unzip_annotations()
            self.__download_coco()

        self.__overlay_images(self.folder_path_dataset_coco_images,
                            self.folder_path_logo,
                            self.folder_path_dataset_train_images, 
                            self.folder_path_dataset_train_labels, 
                            self.folder_path_dataset_val_images,
                            self.folder_path_dataset_val_labels)
        
        self.__create_yaml_file()
        
    def __create_yaml_file(self):
        """
        Create dataset.yaml file.
        
        Params:
            output_path: Path to save the dataset.yaml file
            train_path: Path to the train/images folder
            val_path: Path to the val/images folder
            num_classes: Number of classes in the dataset
            class_names: List of class names
        """
        yaml_content = f"""train: {self.folder_path_dataset_train_images}
val: {self.folder_path_dataset_val_images}

nc: {config.NUMBER_0F_CLASSES}
names: {config.CLASS_NAMES}
"""
        
        self.dataset_config = os.path.join(self.model_dir, "dataset.yaml")

        with open(self.dataset_config, 'w') as yaml_file:
            yaml_file.write(yaml_content)

    def __create_train_model_directories(self) -> None:

        self.model_dir = config.MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)

        self.user_interface.print_info("Creating directories for training...")

        # create dataset directory
        self.folder_path_dataset_coco = os.path.join(self.model_dir, "dataset_coco")
        os.makedirs(self.folder_path_dataset_coco, exist_ok=True)

        # create images directory
        self.folder_path_dataset_coco_images = os.path.join(self.folder_path_dataset_coco, "images_coco")
        os.makedirs(self.folder_path_dataset_coco_images, exist_ok=True)

        # create annotations directory
        self.folder_path_dataset_coco_annotations = os.path.join(self.folder_path_dataset_coco, "annotations_coco")
        os.makedirs(self.folder_path_dataset_coco_annotations, exist_ok=True)

        # create new directory
        self.folder_path_dataset = os.path.join(self.model_dir, "dataset")
        os.makedirs(self.folder_path_dataset, exist_ok=True)

        # create train data directory
        self.folder_path_dataset_train = os.path.join(self.folder_path_dataset, "train")
        os.makedirs(self.folder_path_dataset_train, exist_ok=True)

        # create validation data directory
        self.folder_path_dataset_val = os.path.join(self.folder_path_dataset, "val")
        os.makedirs(self.folder_path_dataset_val, exist_ok=True)

        # create training images directory
        self.folder_path_dataset_train_images = os.path.join(self.folder_path_dataset_train, "images")
        os.makedirs(self.folder_path_dataset_train_images, exist_ok=True)

        # create labels directory
        self.folder_path_dataset_train_labels = os.path.join(self.folder_path_dataset_train, "labels")
        os.makedirs(self.folder_path_dataset_train_labels, exist_ok=True)

        # create validation images directory
        self.folder_path_dataset_val_images = os.path.join(self.folder_path_dataset_val, "images")
        os.makedirs(self.folder_path_dataset_val_images, exist_ok=True)

        # create validation labels directory
        self.folder_path_dataset_val_labels = os.path.join(self.folder_path_dataset_val, "labels")
        os.makedirs(self.folder_path_dataset_val_labels, exist_ok=True)

        self.folder_path_logo = config.INPUT_LOGO_DIR
        os.makedirs(self.folder_path_logo, exist_ok=True)

        self.user_interface.print_info("Directories created successfully.")


    def __download_annotations(self) -> None:

        coco_annotation_url = config.COCO_ANNATOATIONS_URL
        self.folder_path_dataset_coco_annotations_zip = os.path.join(self.folder_path_dataset_coco_annotations, "annotations_trainval2017.zip")

        self.user_interface.print_info("Downloading annotations...")

        r = requests.get(coco_annotation_url)
        with open(self.folder_path_dataset_coco_annotations_zip, "wb") as f:
            f.write(r.content)

        self.user_interface.print_info(f"Annotations downloaded to: {self.folder_path_dataset_coco_annotations_zip}")

    def __unzip_annotations(self) -> None:

        self.user_interface.print_info("Opening annotations...")
        with zipfile.ZipFile(self.folder_path_dataset_coco_annotations_zip, "r") as zip_ref:
            zip_ref.extractall(self.folder_path_dataset_coco_annotations)  
        self.user_interface.print_info(f"Annotations unzipped to: {self.folder_path_dataset_coco_annotations}")

    def __download_coco(self) -> None:
        self.user_interface.print_info("Downloading COCO images...")

        start_time = time.time()

        # Load annotations
        coco = COCO(os.path.join(self.folder_path_dataset_coco_annotations, 'annotations/instances_train2017.json'))

        # Select random 1000 images
        image_ids = random.sample(coco.getImgIds(), 1000)
        total_images = len(image_ids)

        self.user_interface.print_info(f"Preparing to download {total_images} images...")

        for idx, img_id in enumerate(image_ids, start=1):
            try:
                img_info = coco.loadImgs(img_id)[0]
                img_url = img_info['coco_url']
                img_name = img_info['file_name']
                img_path = os.path.join(self.folder_path_dataset_coco_images, img_name)

                # Skip if already exists
                if os.path.exists(img_path):
                    if idx % 50 == 0 or idx == total_images:
                        self.user_interface.print_info(f"[{idx}/{total_images}] Skipping existing image: {img_name}")
                    continue

                # Download image
                response = requests.get(img_url, timeout=15)
                response.raise_for_status()

                with open(img_path, 'wb') as handler:
                    handler.write(response.content)

                # Log progress every 20 images
                if idx % 20 == 0 or idx == total_images:
                    elapsed = time.time() - start_time
                    self.user_interface.print_info(
                        f"[{idx}/{total_images}] Downloaded: {img_name} "
                        f"({elapsed:.1f}s elapsed)"
                    )

            except Exception as e:
                self.user_interface.print_error(f"Error downloading image {img_id}: {e}")

        total_time = time.time() - start_time
        self.user_interface.print_info(f"COCO images saved successfully in {total_time:.1f} seconds.")

    def __overlay_images(self, dataset_coco_images_dir, logo_images_dir, output_folder_train_images, output_folder_train_labels, output_folder_val_images, output_folder_val_labels, train_ratio=0.8):
        """Create dataset with YOLO annotations where each image has its own annotation file, with train/val split."""
        images1 = os.listdir(dataset_coco_images_dir)
        images2 = os.listdir(logo_images_dir)

        total_images = len(images1)
        train_count = int(total_images * train_ratio)  # 80% for training
        train_images = images1[:train_count]
        val_images = images1[train_count:]

        for img1_path in images1:
            base_image = cv2.imread(os.path.join(dataset_coco_images_dir, img1_path))
            random_image_path = random.choice(images2)
            overlay_image = cv2.imread(os.path.join(logo_images_dir, random_image_path), cv2.IMREAD_UNCHANGED)

            base_image = cv2.resize(base_image, (960, 960))

            overlay_image = self.__random_adjust_brightness_contrast_saturation(overlay_image)
            
            if random.random() < 0.1:  
                overlay_image, map_x, map_y = self.__apply_distortion(overlay_image, 0.0001)
                overlay_image = self.__crop_distorted_image(overlay_image, map_x, map_y)
            else:  
                overlay_image = self.__apply_perspective_transform(overlay_image, intensity=random.uniform(0.2, 0.7))

            overlay_image = self.__rotate_image_randomly(overlay_image)
            overlay_image = self.__scale_to_percentage(base_image, overlay_image)

            max_x = base_image.shape[1] - overlay_image.shape[1]
            max_y = base_image.shape[0] - overlay_image.shape[0]
            position_x = random.randint(0, max_x)
            position_y = random.randint(0, max_y)

            overlay_bbox = {
                'x_min': position_x,
                'y_min': position_y,
                'width': overlay_image.shape[1],
                'height': overlay_image.shape[0],
                'image_width': base_image.shape[1],
                'image_height': base_image.shape[0]
            }

            overlay_image_rgb = overlay_image[:, :, :3]
            mask = overlay_image[:, :, 3] if overlay_image.shape[2] == 4 else np.ones_like(overlay_image_rgb[:, :, 0]) * 255
            for c in range(0, 3):
                base_image[position_y:position_y + overlay_image.shape[0], position_x:position_x + overlay_image.shape[1], c] = \
                    base_image[position_y:position_y + overlay_image.shape[0], position_x:position_x + overlay_image.shape[1], c] * (1 - mask / 255) + \
                    overlay_image_rgb[:, :, c] * (mask / 255)

            # Determine train or val split
            if img1_path in train_images:
                output_folder_images = output_folder_train_images
                output_folder_labels = output_folder_train_labels
            else:
                output_folder_images = output_folder_val_images
                output_folder_labels = output_folder_val_labels

            # Save the image
            output_image_path = os.path.join(output_folder_images, f"output_{os.path.basename(img1_path)}")
            cv2.imwrite(output_image_path, base_image)

            # Save the annotation
            output_annotation_path = os.path.join(output_folder_labels, f"output_{os.path.splitext(os.path.basename(img1_path))[0]}.txt")
            self.__save_yolo_annotation(output_annotation_path, overlay_bbox, class_id=0)

    def __random_adjust_brightness_contrast_saturation(self, image, brightness_range=(0.7, 1.1), contrast_range=(0.7, 1.1), saturation_range=(0.7, 1.1)):
        """
        Randomly changes the brightness, contrast, and saturation of the image, supporting the alpha channel.
        """
        if image.shape[2] == 4:
            bgr, alpha = image[:, :, :3], image[:, :, 3]
        else:
            bgr, alpha = image, None

        brightness_factor = random.uniform(*brightness_range)
        bgr = np.clip(bgr * brightness_factor, 0, 255).astype(np.uint8)

        contrast_factor = random.uniform(*contrast_range)
        mean = np.mean(bgr, axis=(0, 1), keepdims=True)
        bgr = np.clip((bgr - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

        hsv_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_image[..., 1] *= random.uniform(*saturation_range)
        hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 255)
        bgr = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if alpha is not None:
            return cv2.merge((bgr, alpha))

        return bgr
    
    def __apply_perspective_transform(self, image, intensity):
        """
        Applies perspective transformations and crops the image accordingly.
        """
        
        h, w = image.shape[:2]
        src_points = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)
        
        max_shift_x = int(w * intensity)
        max_shift_y = int(h * intensity)

        dst_points = np.array([
            [np.random.randint(0, max_shift_x // 2), np.random.randint(0, max_shift_y // 2)],  # Top-left
            [np.random.randint(0, max_shift_x // 2), h - np.random.randint(0, max_shift_y // 2)],  # Bottom-left
            [w - np.random.randint(0, max_shift_x // 2), h - np.random.randint(0, max_shift_y // 2)],  # Bottom-right
            [w - np.random.randint(0, max_shift_x // 2), np.random.randint(0, max_shift_y // 2)],  # Top-right
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        dst_points_min = dst_points.min(axis=0).astype(int) 
        dst_points_max = dst_points.max(axis=0).astype(int) 

        new_width = dst_points_max[0] - dst_points_min[0]
        new_height = dst_points_max[1] - dst_points_min[1]

        translation_matrix = np.array([
            [1, 0, -dst_points_min[0]],
            [0, 1, -dst_points_min[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        new_matrix = np.dot(translation_matrix, matrix)

        transformed_image = cv2.warpPerspective(image, new_matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        return transformed_image
    
    def __apply_distortion(self, image, a):
        """
        Applies barrel distortion to an image.
        """
        h, w = image.shape[:2]

        xi, yi = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))

        imid_x = w / 2
        imid_y = h / 2

        xt = xi - imid_x
        yt = yi - imid_y

        r = np.sqrt(xt**2 + yt**2)
        theta = np.arctan2(yt, xt)

        r_distorted = r + a * r**3

        xt_distorted = r_distorted * np.cos(theta)
        yt_distorted = r_distorted * np.sin(theta)

        u = xt_distorted + imid_x
        v = yt_distorted + imid_y

        map_x = u.astype(np.float32)
        map_y = v.astype(np.float32)

        distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return distorted_image, map_x, map_y

    def __crop_distorted_image(self, image, map_x, map_y):
        non_zero_x = np.where((map_x >= 0) & (map_x < image.shape[1]))[1]
        non_zero_y = np.where((map_y >= 0) & (map_y < image.shape[0]))[0]

        x_min, x_max = non_zero_x.min(), non_zero_x.max()
        y_min, y_max = non_zero_y.min(), non_zero_y.max()

        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
        return cropped_image
    
    def __rotate_image_randomly(self, image):
        """
        Rotates the image by a random angle between -90 and +90 degrees, stretching the canvas to avoid cropping.
        """

        angle = random.uniform(-90, 90)

        h, w = image.shape[:2]

        center = (w // 2, h // 2)

        radians = np.deg2rad(angle)
        new_w = int(abs(w * np.cos(radians)) + abs(h * np.sin(radians)))
        new_h = int(abs(w * np.sin(radians)) + abs(h * np.cos(radians)))

        new_center = (new_w // 2, new_h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotation_matrix[0, 2] += (new_center[0] - center[0])
        rotation_matrix[1, 2] += (new_center[1] - center[1])

        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        return rotated_image
    
    def __scale_to_percentage(self, base_image, overlay_image, min_percent=0.1, max_percent=40):
        """
        Scale the overlay image to a percentage of the base image size, maintaining aspect ratio.
        Ensures the scaled overlay fits within the dimensions of the base image.
        """
        base_area = base_image.shape[0] * base_image.shape[1]
        scale_factor = random.uniform(min_percent / 100, max_percent / 100)
        overlay_area = base_area * scale_factor

        aspect_ratio = overlay_image.shape[1] / overlay_image.shape[0]
        new_width = int((overlay_area * aspect_ratio) ** 0.5)
        new_height = int((overlay_area / aspect_ratio) ** 0.5)

        if new_width > base_image.shape[1]:
            new_width = base_image.shape[1]
            new_height = int(new_width / aspect_ratio)
        if new_height > base_image.shape[0]:
            new_height = base_image.shape[0]
            new_width = int(new_height * aspect_ratio)

        return cv2.resize(overlay_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def __save_yolo_annotation(self, output_file, overlay_bbox, class_id=0):
        """Save YOLOv8 annotations to a separate file for each image."""

        image_width, image_height = overlay_bbox['image_width'], overlay_bbox['image_height']

        x_center = (overlay_bbox['x_min'] + overlay_bbox['width'] / 2) / image_width
        y_center = (overlay_bbox['y_min'] + overlay_bbox['height'] / 2) / image_height
        width = overlay_bbox['width'] / image_width
        height = overlay_bbox['height'] / image_height

        with open(output_file, "w") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def main():
    userInterface = UserInterface()
    userInterface.print_info("TRAINING MODEL MODULE, PRESS CTRL+C TO EXIT")
    try:
        model_trainer = ModelTrainer()
        download = userInterface.get_input("Do you want to download coco images? y/n")
        model_trainer.prepare_dataset(download)
        if not model_trainer.gpu_ready():
            cpu = userInterface.get_confirmation("GPU is not available. Do you want to train model on CPU? y/n")
            if not cpu:
                sys.exit(0)

        if userInterface.get_confirmation("Do you want to start training? y/n"):
            model_trainer.start_training()
        else:
            sys.exit(0)

    except Exception as e:
        userInterface.print_error("Exception: " + str(e))
        traceback.print_exc() 

if __name__ == "__main__":
    main()
