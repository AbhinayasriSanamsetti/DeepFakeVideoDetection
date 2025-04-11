import os
import cv2
import torch
from ultralytics import YOLO

# Define directories
REAL_VIDEOS_DIR = 'C:/Users/HP/Downloads/archive/Celeb-real'
SYNTHESIS_VIDEOS_DIR = 'C:/Users/HP/Downloads/archive/Celeb-synthesis'
YOUTUBE_REAL_VIDEOS_DIR = 'C:/Users/HP/Downloads/archive/YouTube-real'
OUTPUT_DIR = 'C:/Users/HP/Desktop/deepfake/frames_labels'
LABELS_FILE = 'C:/Users/HP/Desktop/deepfake/frames_labels/labels.txt'  # File to store labels and coordinates

# Define constants
FRAME_INTERVAL = 10  # Process every 10th frame
IMG_SIZE = (416, 416)  # Resize images
MODEL_PATH = "C:/Users/HP/Desktop/deepfake/models/model.pt"  # Your YOLOv8 model path

# Load YOLOv8 face detection model
model = YOLO(MODEL_PATH)

def extract_faces_from_video(video_path, label, output_dir, labels_file, frame_interval=10):
    """
    Extracts faces from a video using YOLOv8, processes every 10th frame, 
    saves only frames with detected faces, and stores coordinates in a text file.
    
    :param video_path: Path to the video file.
    :param label: Label (0 for real, 1 for fake).
    :param output_dir: Directory to save frames.
    :param labels_file: File to store bounding box coordinates & labels.
    :param frame_interval: Interval for frame extraction.
    """
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frame_count = 0
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    with open(labels_file, 'a') as f:  # Append mode to store labels
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break  # Stop when video ends

            frame_count += 1

            if frame_count % frame_interval == 0:
                # Perform face detection
                results = model(frame)
                detections = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes

                if len(detections) > 0:  # If at least one face is detected
                    saved_frame_count += 1
                    for (x1, y1, x2, y2) in detections:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        face = frame[y1:y2, x1:x2]  # Crop the face
                        face_resized = cv2.resize(face, IMG_SIZE)  # Resize for consistency

                        # Save the processed face image
                        face_filename = os.path.join(video_output_dir, f"frame_{saved_frame_count}_face.jpg")
                        cv2.imwrite(face_filename, face_resized)

                        # Store bounding box coordinates and label
                        f.write(f"{face_filename} {label} {x1} {y1} {x2} {y2}\n")

    video_capture.release()
    print(f"Processed {frame_count} frames, saved {saved_frame_count} frames from {video_path}")

def process_videos(video_dirs, output_dir, labels_file, frame_interval=10):
    """
    Processes all videos from the given directories and stores face bounding box info.
    
    :param video_dirs: List of directories with video files and their labels.
    :param output_dir: Directory to save processed frames.
    :param labels_file: File to store bounding box coordinates & labels.
    :param frame_interval: Interval for frame extraction.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Clear the labels file before starting
    with open(labels_file, 'a') as f:
        f.write("Image_Path Label x1 y1 x2 y2\n")  # Header for label file

    for video_dir, label in video_dirs:
        if not os.path.exists(video_dir):
            print(f"Warning: Directory {video_dir} not found.")
            continue
        
        for video_file in os.listdir(video_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(video_dir, video_file)
                print(f"Processing {video_file}...")
                extract_faces_from_video(video_path, label, output_dir, labels_file, frame_interval)

# List of video directories and their labels (0 for real, 1 for fake)
video_dirs = [
    (REAL_VIDEOS_DIR, 0),  # Real videos -> label 0
   # (SYNTHESIS_VIDEOS_DIR, 1),  # Fake (synthesized) videos -> label 1
    (YOUTUBE_REAL_VIDEOS_DIR, 0)  # YouTube real videos -> label 0 (assuming they are manipulated)
]

# Process all videos
process_videos(video_dirs, OUTPUT_DIR, LABELS_FILE, frame_interval=FRAME_INTERVAL)


