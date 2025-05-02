
import streamlit as st
import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from training import CNNFeatureExtractor, DeepfakeLSTM
from sklearn.metrics.pairwise import cosine_similarity

# === Configs ===
MODEL_PATH = r"C:\Users\HP\Desktop\deepfake\model_main.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.4
NUM_LSTM_LAYERS = 2
SIMILARITY_THRESHOLD = 0.7
MAX_FRAMES_TO_DISPLAY = 20

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Load models ===
@st.cache_resource
def load_models():
    cnn = CNNFeatureExtractor().to(DEVICE)
    lstm = DeepfakeLSTM(num_layers=NUM_LSTM_LAYERS).to(DEVICE)

    try:
        model_state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        lstm.load_state_dict(model_state_dict)
        cnn.eval()
        lstm.eval()
    except Exception as e:
        st.error(f"Error loading model: {e}")

    yolo = YOLO("yolov8n.pt")
    return cnn, lstm, yolo

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration

def extract_faces_from_video(video_path, yolo):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    face_crops = []
    raw_frames = []
    timestamps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract every 10th frame
        if frame_idx % 10 == 0:
            results = yolo(frame)[0]
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        face = frame[y1:y2, x1:x2]

                        vis_frame = frame.copy()
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)

                        raw_frames.append(vis_frame)
                        timestamps.append(frame_idx / fps)

                        face_img = Image.fromarray(face)
                        face_tensor = transform(face_img)
                        face_crops.append(face_tensor)
                        break
        frame_idx += 1

    cap.release()
    return face_crops, raw_frames, timestamps

def threshold_based_classification(probs, threshold=0.05):
    real_conf = probs[0].item()
    fake_conf = probs[1].item()
    if real_conf > fake_conf and real_conf > 0.3:
        return "REAL", real_conf
    elif fake_conf > real_conf and fake_conf > 0.55:
        return "FAKE", fake_conf
    elif abs(real_conf - fake_conf) < threshold:
        return "UNCERTAIN", max(real_conf, fake_conf)
    else:
        return "UNCERTAIN", max(real_conf, fake_conf)

def custom_consistency_check(face1, face2, cnn):
    with torch.no_grad():
        face1 = transform(Image.fromarray(face1)).unsqueeze(0).to(DEVICE)
        face2 = transform(Image.fromarray(face2)).unsqueeze(0).to(DEVICE)
        emb1 = cnn(face1).cpu().numpy()
        emb2 = cnn(face2).cpu().numpy()
        sim = cosine_similarity(emb1, emb2)[0][0]
        return sim >= SIMILARITY_THRESHOLD

def process_video(video_path, cnn, lstm, yolo):
    with torch.no_grad():
        video_duration = get_video_duration(video_path)

        st.write(f"📹 Video Duration: {video_duration:.1f} seconds")
        st.write(f"⏱ Analyzing every 10th frame...")

        face_tensors, raw_faces, timestamps = extract_faces_from_video(video_path, yolo)
        if not face_tensors:
            st.error("⚠ No faces detected in the video.")
            return None, None, None, None, None

        st.write(f"✅ Total frames processed: {len(face_tensors)}")
        frame_preds = []
        previous_frame = None

        for idx, face_tensor in enumerate(face_tensors):
            input_tensor = face_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)
            b, t, c, h, w = input_tensor.shape
            features = cnn(input_tensor.view(b * t, c, h, w)).view(b, t, -1)
            output = lstm(features)
            probs = torch.softmax(output, dim=1).squeeze()

            label, confidence = threshold_based_classification(probs)

            if previous_frame is not None:
                consistent = custom_consistency_check(previous_frame, raw_faces[idx], cnn)
                if not consistent:
                    label = "UNCERTAIN"
                    st.warning(f"⚠ Frame at {timestamps[idx]:.1f}s: Inconsistent (embedding mismatch)")

            frame_preds.append((label, confidence))
            previous_frame = raw_faces[idx]

        confident_preds = [(label, conf) for label, conf in frame_preds if label in ["REAL", "FAKE"]]
        confident_real = [conf for label, conf in confident_preds if label == "REAL"]
        confident_fake = [conf for label, conf in confident_preds if label == "FAKE"]

        total_confident = len(confident_real) + len(confident_fake)
        avg_real = np.mean(confident_real) if confident_real else 0
        avg_fake = np.mean(confident_fake) if confident_fake else 0

        real_score = len(confident_real) * avg_real
        fake_score = len(confident_fake) * avg_fake

        if total_confident == 0:
            final_label = "INSUFFICIENT DATA"
            final_confidence = 0
        else:
            final_label = "REAL" if real_score > fake_score else "FAKE"
            final_confidence = max(real_score, fake_score)
            final_confidence = final_confidence / total_confident  # Normalize score


        return final_label, final_confidence, raw_faces, frame_preds, timestamps

def process_image(image_path, cnn, lstm, yolo):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = yolo(image)[0]
    if results.boxes is None:
        st.error("⚠ No faces detected in the image.")
        return None, None, None

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            face = image[y1:y2, x1:x2]

            vis_frame = image.copy()
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face_img = Image.fromarray(face)
            face_tensor = transform(face_img).unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                b, t, c, h, w = face_tensor.shape
                features = cnn(face_tensor.view(b * t, c, h, w)).view(b, t, -1)
                output = lstm(features)
                probs = torch.softmax(output, dim=1).squeeze()
                label, confidence = threshold_based_classification(probs)

            return label, confidence, vis_frame

def main():
    st.set_page_config(page_title="DeepFake Detector", layout="wide")
    st.title("🔍 DeepFake Detection")

    detection_type = st.sidebar.radio(
        "Select Detection Type",
        ["Image Detection", "Video Detection"]
    )

    cnn, lstm, yolo = load_models()

    if detection_type == "Image Detection":
        st.header("📷 Image Detection")
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            temp_path = "temp_image.jpg"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Analyzing image..."):
                label, confidence, vis_frame = process_image(temp_path, cnn, lstm, yolo)

                if label is not None:
                    st.header("📊 Analysis Results")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Prediction", label)
                        st.metric("Confidence", f"{confidence * 100:.2f}")

                    with col2:
                        st.image(vis_frame, caption="Detected Face")

            os.remove(temp_path)

    else:
        st.header("🎥 Video Detection")
        uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Analyzing video..."):
                final_label, final_confidence, raw_faces, frame_preds, timestamps = process_video(temp_path, cnn, lstm, yolo)

                if final_label is not None:
                    st.header("📊 Analysis Results")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Final Prediction", final_label)
                        st.metric("Confidence", f"{final_confidence * 100:.2f}")

                    with col2:
                        real_frames = sum(1 for label, _ in frame_preds if label == "REAL")
                        fake_frames = sum(1 for label, _ in frame_preds if label == "FAKE")
                        st.metric("REAL Frames", real_frames)
                        st.metric("FAKE Frames", fake_frames)

                    st.header("📸 Processed Frames")

                    num_frames = min(len(raw_faces), MAX_FRAMES_TO_DISPLAY)
                    num_cols = min(4, num_frames)
                    cols = st.columns(num_cols)

                    for i in range(num_frames):
                        frame_idx = i
                        with cols[i % num_cols]:
                            timestamp = timestamps[frame_idx]
                            label, confidence = frame_preds[frame_idx]
                            st.image(
                                raw_faces[frame_idx],
                                caption=f"Time: {timestamp:.1f}s\n{label} ({confidence * 100:.2f})",
                            )

            os.remove(temp_path)

if _name_ == "_main_":
    main()