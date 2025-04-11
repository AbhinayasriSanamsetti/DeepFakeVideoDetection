import os
from huggingface_hub import hf_hub_download

# Define local path to save the model
MODEL_DIR = "C:/Users/HP/Desktop/deepfake/models"  # Change this to your preferred directory
MODEL_FILENAME = "yolov8n-face.pt"
LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model only if it doesn't exist
if not os.path.exists(LOCAL_MODEL_PATH):
    print(f"Downloading YOLOv8 model and saving to {LOCAL_MODEL_PATH}...")
    LOCAL_MODEL_PATH = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt", local_dir=MODEL_DIR)
    print("Download completed successfully.")
else:
    print(f"Model already exists at {LOCAL_MODEL_PATH}, skipping download.")
