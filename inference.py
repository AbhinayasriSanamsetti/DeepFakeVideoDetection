import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from training import CNNFeatureExtractor, DeepfakeLSTM  # Update with your actual module if needed

# Configs
VIDEO_PATH = r"C:\Users\HP\Downloads\WhatsApp Video 2025-04-11 at 7.08.01 PM.mp4"
MODEL_PATH = "best_lstm_model.pth"
SEQ_LENGTH = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load models
cnn = CNNFeatureExtractor().to(DEVICE)
lstm = DeepfakeLSTM().to(DEVICE)
lstm.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
cnn.eval()
lstm.eval()

# Extract 16 frames uniformly
def extract_frames(video_path, seq_len=SEQ_LENGTH):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, seq_len, dtype=int)

    tensor_frames = []
    raw_frames = []

    count = 0
    i = 0
    while cap.isOpened() and i < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if count == indices[i]:
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # for matplotlib
            img = Image.fromarray(raw_frames[-1])
            tensor_frames.append(transform(img))
            i += 1
        count += 1
    cap.release()

    return torch.stack(tensor_frames), raw_frames

# Predict + visualize
def predict_and_plot(video_path):
    with torch.no_grad():
        frames_tensor, raw_frames = extract_frames(video_path)
        frames_tensor = frames_tensor.unsqueeze(0).to(DEVICE)
        b, t, c, h, w = frames_tensor.shape
        features = cnn(frames_tensor.view(b * t, c, h, w)).view(b, t, -1)
        outputs = lstm(features)

        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs.squeeze()[pred].item()

        label_map = {0: "REAL", 1: "FAKE"}
        pred_label = label_map[pred]

        print(f"Prediction: {pred_label} (Confidence: {confidence:.4f})")

        # Plotting
        plt.figure(figsize=(16, 8))
        for i, frame in enumerate(raw_frames):
            plt.subplot(4, 4, i + 1)
            plt.imshow(frame)
            plt.axis("off")
            plt.title(f"{pred_label}\nConf: {confidence:.2f}", fontsize=10,
                      color='green' if pred_label == "REAL" else 'red')
        plt.suptitle("Deepfake Detection Inference", fontsize=16)
        plt.tight_layout()
        plt.show()

# Run
if __name__ == "__main__":
    predict_and_plot(VIDEO_PATH)
