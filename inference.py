import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageTk
from torchvision import transforms
from ultralytics import YOLO
from training import CNNFeatureExtractor, DeepfakeLSTM
import tkinter as tk
from tkinter import Canvas, Scrollbar, Frame, Label
from sklearn.metrics.pairwise import cosine_similarity

# === Configs ===
VIDEO_PATH = r"C:\Users\HP\Desktop\deepfake\archive\Celeb-synthesis\id0_id2_0007.mp4"
MODEL_PATH = r"C:\Users\HP\Desktop\deepfake\model_main.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.4
NUM_LSTM_LAYERS = 2
FRAME_INTERVAL = 5
SIMILARITY_THRESHOLD = 0.7

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Load models ===
cnn = CNNFeatureExtractor().to(DEVICE)
lstm = DeepfakeLSTM(num_layers=NUM_LSTM_LAYERS).to(DEVICE)

def load_model(model_path, model, device):
    try:
        model_state_dict = torch.load(model_path, map_location=device)
        print(f"Loaded model from {model_path}")
        model.load_state_dict(model_state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
    return model

lstm = load_model(MODEL_PATH, lstm, DEVICE)
cnn.eval()
lstm.eval()

yolo = YOLO("yolov8n.pt")

def extract_faces_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    face_crops = []
    raw_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_INTERVAL == 0:
            results = yolo(frame)[0]
            if results.boxes is not None:
                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        face = frame[y1:y2, x1:x2]

                        vis_frame = frame.copy()
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)

                        raw_frames.append(vis_frame)

                        face_img = Image.fromarray(face)
                        face_tensor = transform(face_img)
                        face_crops.append(face_tensor)
                        break
        frame_idx += 1

    cap.release()
    return face_crops, raw_frames

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

def custom_consistency_check(face1, face2):
    with torch.no_grad():
        face1 = transform(Image.fromarray(face1)).unsqueeze(0).to(DEVICE)
        face2 = transform(Image.fromarray(face2)).unsqueeze(0).to(DEVICE)
        emb1 = cnn(face1).cpu().numpy()
        emb2 = cnn(face2).cpu().numpy()
        sim = cosine_similarity(emb1, emb2)[0][0]
        return sim >= SIMILARITY_THRESHOLD

def predict_and_visualize(video_path):
    with torch.no_grad():
        face_tensors, raw_faces = extract_faces_from_video(video_path)
        if not face_tensors:
            print("⚠️ No faces detected.")
            return

        print(f"\n✅ Total frames processed: {len(face_tensors)}")
        frame_preds = []
        previous_frame = None

        print("\n🧪 Frame-by-frame predictions:")

        for idx, face_tensor in enumerate(face_tensors):
            input_tensor = face_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)
            b, t, c, h, w = input_tensor.shape
            features = cnn(input_tensor.view(b * t, c, h, w)).view(b, t, -1)
            output = lstm(features)
            probs = torch.softmax(output, dim=1).squeeze()

            print(f"Probabilities for Frame {idx + 1}: {probs}")
            label, confidence = threshold_based_classification(probs)

            if previous_frame is not None:
                consistent = custom_consistency_check(previous_frame, raw_faces[idx])
                if not consistent:
                    label = "UNCERTAIN"
                    print(f"⚠️ Frame {idx + 1}: Inconsistent (embedding mismatch)")

            frame_preds.append((label, confidence))
            print(f"Frame {idx + 1}: {label} (Confidence: {confidence:.4f})")
            previous_frame = raw_faces[idx]

        # === Only consider confident (REAL or FAKE) frames for final output ===
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

        summary_text = (
            f"📊 Summary:\n"
            f"REAL frames: {len(confident_real)} | Avg confidence: {avg_real:.4f}\n"
            f"FAKE frames: {len(confident_fake)} | Avg confidence: {avg_fake:.4f}\n"
            f"🔢 REAL Ratio: {len(confident_real) / total_confident if total_confident else 0:.2f}, Score: {real_score:.4f}\n"
            f"🔢 FAKE Ratio: {len(confident_fake) / total_confident if total_confident else 0:.2f}, Score: {fake_score:.4f}\n"
            f"🧠 Final Prediction: {final_label} (Confidence: {final_confidence:.2f})"
        )

        print("\n" + summary_text)

        # === UI ===
        root = tk.Tk()
        root.title("Deepfake Frame Classification")

        canvas = Canvas(root)
        scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        final_label_text = f"🧠 Final Prediction: {final_label} (Confidence: {final_confidence:.2f})"
        final_label_widget = Label(root, text=final_label_text, font=("Arial", 24, "bold"), bg="lightgrey", padx=10, pady=5)
        final_label_widget.pack(fill="x")

        for i, (frame, (label, confidence)) in enumerate(zip(raw_faces, frame_preds)):
            img = Image.fromarray(frame)
            img = img.resize((400, 400))
            img_tk = ImageTk.PhotoImage(img)
            frame_label = Label(scrollable_frame, image=img_tk)
            frame_label.image = img_tk
            frame_label.grid(row=i // 4 * 2, column=i % 4)

            frame_status_label = Label(scrollable_frame, text=f"{label} ({confidence:.2f})", font=("Arial", 8))
            frame_status_label.grid(row=i // 4 * 2 + 1, column=i % 4)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        root.mainloop()

# Run the process
predict_and_visualize(VIDEO_PATH)