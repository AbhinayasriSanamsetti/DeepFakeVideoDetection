import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import cv2
import numpy as np
from tqdm import tqdm
import copy

# Config
LABELS_FILE = 'C:/Users/HP/Desktop/deepfake/frames_labels/labels.txt'
BATCH_SIZE = 4
SEQ_LENGTH = 10
EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
PATIENCE = 3  # For early stopping

# ImageNet normalization
imagenet_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

# Dataset
class DeepfakeDataset(Dataset):
    def __init__(self, labels_file, seq_length=10):
        self.seq_length = seq_length
        self.data = self.load_labels(labels_file)

    def load_labels(self, labels_file):
        sequences = {}
        with open(labels_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                continue

            frame_path, label, x1, y1, x2, y2 = parts
            if not label.isdigit():
                continue

            label = int(label)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            video_name = os.path.basename(os.path.dirname(frame_path))

            if video_name not in sequences:
                sequences[video_name] = {"frames": [], "labels": [], "bboxes": []}

            sequences[video_name]["frames"].append(frame_path)
            sequences[video_name]["labels"].append(label)
            sequences[video_name]["bboxes"].append([x1, y1, x2, y2])

        valid_sequences = []
        for video, data in sequences.items():
            frames = sorted(data["frames"])
            labels = data["labels"]
            bboxes = data["bboxes"]

            if len(frames) >= self.seq_length:
                for i in range(0, len(frames) - self.seq_length + 1, self.seq_length):
                    valid_sequences.append((frames[i:i + self.seq_length],
                                            labels[i:i + self.seq_length],
                                            bboxes[i:i + self.seq_length]))
        return valid_sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_paths, labels, _ = self.data[idx]
        frames = []

        for frame_path in frame_paths:
            img = cv2.imread(frame_path)
            if img is None:
                print(f"⚠️ Failed to load image: {frame_path}")
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))

            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            img = imagenet_transform(img)
            frames.append(img)

        frames = torch.stack(frames)
        return frames, torch.tensor(labels[-1], dtype=torch.long)

# Models
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

# Training
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DeepfakeDataset(LABELS_FILE, SEQ_LENGTH)
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    cnn = CNNFeatureExtractor().to(device)
    lstm = DeepfakeLSTM().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_model_wts = None
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\n🔁 Epoch {epoch+1}/{EPOCHS}")
        cnn.eval()
        lstm.train()

        total_loss = 0
        all_preds, all_labels = [], []

        for frames, labels in tqdm(train_loader, desc="Training"):
            frames, labels = frames.to(device), labels.to(device)
            b, t, c, h, w = frames.shape
            frames = frames.view(b * t, c, h, w)
            features = cnn(frames).view(b, t, -1)

            outputs = lstm(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds, average='weighted')
        print(f"📊 Training - Loss: {total_loss:.4f}, Acc: {train_acc:.2%}, Precision: {train_precision:.4f}")

        # Validation phase
        cnn.eval()
        lstm.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                b, t, c, h, w = frames.shape
                frames = frames.view(b * t, c, h, w)
                features = cnn(frames).view(b, t, -1)
                outputs = lstm(features)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='weighted')
        print(f"✅ Validation - Acc: {val_acc:.2%}, Precision: {val_precision:.4f}")
        print(confusion_matrix(val_labels, val_preds))
        print(classification_report(val_labels, val_preds))

        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(lstm.state_dict())
            torch.save(best_model_wts, "best_lstm_model.pth")
            print("💾 Best model updated!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("🛑 Early stopping triggered.")
                break

    print(f"🏁 Training finished. Best validation accuracy: {best_val_acc:.2%}")

if __name__ == "__main__":
    train_model()