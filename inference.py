import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO  # Requires: pip install ultralytics
from training import CNNFeatureExtractor, DeepfakeLSTM

# Configs
VIDEO_PATH = r"C:\Users\HP\Downloads\WhatsApp Video 2025-04-11 at 7.01.14 PM.mp4"
MODEL_PATH = "best_lstm_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.3  # YOLO detection threshold

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

# Load YOLOv8 face detection model
yolo = YOLO("yolov8n.pt")  # Lightweight; you can also try yolov8s.pt or custom face model

# Extract frames and detect faces
def extract_faces_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    face_crops = []
    raw_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame)[0]
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            face = frame[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            raw_frames.append(face)
            face_img = Image.fromarray(face)
            face_tensor = transform(face_img)
            face_crops.append(face_tensor)
            break  # Process only one face per frame

        frame_idx += 1

    cap.release()
    return face_crops, raw_frames

# Predict + visualize
def predict_and_visualize(video_path):
    with torch.no_grad():
        face_tensors, raw_faces = extract_faces_from_video(video_path)
        if not face_tensors:
            print("⚠️ No faces detected.")
            return

        all_real_conf = []
        all_fake_conf = []

        print("\n🧪 Frame-by-frame predictions:")
        for idx, face_tensor in enumerate(face_tensors):
            input_tensor = face_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,C,H,W)
            b, t, c, h, w = input_tensor.shape
            features = cnn(input_tensor.view(b * t, c, h, w)).view(b, t, -1)
            output = lstm(features)

            probs = torch.softmax(output, dim=1).squeeze()
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()

            label = "REAL" if pred == 0 else "FAKE"
            print(f"Frame {idx+1}: {label} (Confidence: {confidence:.4f})")

            if pred == 0:
                all_real_conf.append(confidence)
            else:
                all_fake_conf.append(confidence)

        avg_real = np.mean(all_real_conf) if all_real_conf else 0
        avg_fake = np.mean(all_fake_conf) if all_fake_conf else 0

        final_label = "REAL" if avg_real >= avg_fake else "FAKE"
        final_confidence = max(avg_real, avg_fake)

        print("\n📊 Summary:")
        print(f"Average REAL confidence: {avg_real:.4f}")
        print(f"Average FAKE confidence: {avg_fake:.4f}")
        print(f"🧠 Final Prediction: {final_label} (Confidence: {final_confidence:.4f})")

        # Plot
        plt.figure(figsize=(16, 8))
        for i, face in enumerate(raw_faces[:16]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(face)
            plt.axis("off")
        plt.suptitle(f"Final Prediction: {final_label} ({final_confidence:.2f})", fontsize=16,
                     color='green' if final_label == "REAL" else 'red')
        plt.tight_layout()
        plt.show()

# Run
if __name__ == "__main__":
    predict_and_visualize(VIDEO_PATH)
