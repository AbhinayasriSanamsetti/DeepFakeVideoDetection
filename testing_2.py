import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import cv2
from tqdm import tqdm
import logging
import random

# ---------------------- Configuration ----------------------
LABELS_FILE = r"C:\Users\HP\Desktop\deepfake\frames_labels\labels.txt"
MODEL_PATH = r"C:\Users\HP\Desktop\deepfake\model_main.pth"
SEQ_LENGTH = 10
BATCH_SIZE = 4
TEST_SPLIT_RATIO = 0.1
LOG_FILE = 'test_results.log'

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------------- Transform ----------------------
imagenet_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])


# ---------------------- Dataset ----------------------
class DeepfakeDataset(Dataset):
    def __init__(self, label_file, seq_length=10):
        self.seq_length = seq_length
        self.data = self.load_sequences(label_file)

    def load_sequences(self, label_file):
        sequences = {}
        with open(label_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                frame_path, label, x1, y1, x2, y2 = parts
                try:
                    label = int(label)
                except ValueError:
                    continue
                video_name = os.path.basename(os.path.dirname(frame_path))
                if video_name not in sequences:
                    sequences[video_name] = {"frames": [], "labels": []}
                sequences[video_name]["frames"].append(frame_path)
                sequences[video_name]["labels"].append(label)

        # Create clips
        data = []
        for video_data in sequences.values():
            frames = video_data["frames"]
            labels = video_data["labels"]
            if len(frames) >= self.seq_length:
                for i in range(0, len(frames) - self.seq_length + 1, self.seq_length):
                    clip_frames = frames[i:i + self.seq_length]
                    clip_labels = labels[i:i + self.seq_length]
                    data.append((clip_frames, clip_labels))
        random.shuffle(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_paths, labels = self.data[idx]
        frames = []

        for path in frame_paths:
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            img = imagenet_transform(img)
            frames.append(img)

        return torch.stack(frames), torch.tensor(labels[-1], dtype=torch.long)


# ---------------------- Models ----------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        return x.view(x.size(0), -1)


class DeepfakeLSTM(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_layers=2, num_classes=2):
        super(DeepfakeLSTM, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


# ---------------------- Testing Function ----------------------
def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DeepfakeDataset(label_file=LABELS_FILE, seq_length=SEQ_LENGTH)

    test_size = int(len(dataset) * TEST_SPLIT_RATIO)
    train_size = len(dataset) - test_size
    _, test_dataset = random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    cnn = CNNFeatureExtractor().to(device)
    cnn.eval()

    lstm = DeepfakeLSTM().to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Check if checkpoint is a full dict (likely when using torch.save({...}))
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            lstm.load_state_dict(checkpoint["model_state_dict"])
        else:
            lstm.load_state_dict(checkpoint)
    else:
        lstm.load_state_dict(checkpoint)

    all_preds, all_labels, confidences = [], [], []

    print("🔍 Testing started...")
    for frames, labels in tqdm(test_loader, desc="Testing"):
        frames, labels = frames.to(device), labels.to(device)
        b, t, c, h, w = frames.shape
        frames = frames.view(b * t, c, h, w)

        features = cnn(frames).view(b, t, -1)
        outputs = lstm(features)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        confidences.extend(conf.detach().cpu().numpy())


    acc = accuracy_score(all_labels, all_preds)
    avg_conf = np.mean(confidences)

    print(f"✅ Test Accuracy: {acc:.2%}")
    print(f"📈 Average Confidence: {avg_conf:.4f}")
    print("📋 Classification Report:\n", classification_report(all_labels, all_preds))

    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Average Confidence: {avg_conf:.4f}")
    logging.info("Classification Report:\n" + classification_report(all_labels, all_preds))


# ---------------------- Main ----------------------
if __name__ == "__main__":
    test_model()
