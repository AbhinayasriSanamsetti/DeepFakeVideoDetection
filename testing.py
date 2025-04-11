import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import numpy as np
import cv2
from tqdm import tqdm
import logging

# --------------------- Configuration ---------------------
LABELS_FILE = 'C:/Users/HP/Desktop/deepfake/frames_labels/labels.txt'
BATCH_SIZE = 4
SEQ_LENGTH = 10
TEST_SPLIT = 0.2
MODEL_PATH = "C:/Users/HP/Desktop/deepfake/best_lstm_model.pth"
LOG_FILE = "test_results.log"

# --------------------- Logging Setup ---------------------
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --------------------- Transform ---------------------
imagenet_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

# --------------------- Dataset ---------------------
class DeepfakeDataset(Dataset):
    def __init__(self, labels_file, seq_length=10):
        self.seq_length = seq_length
        self.data = self.load_labels(labels_file)

    def load_labels(self, labels_file):
        sequences = {}
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 6 or not parts[1].isdigit():
                    continue
                frame_path, label, x1, y1, x2, y2 = parts
                video = os.path.basename(os.path.dirname(frame_path))
                if video not in sequences:
                    sequences[video] = {"frames": [], "labels": []}
                sequences[video]["frames"].append(frame_path)
                sequences[video]["labels"].append(int(label))

        valid = []
        for frames in sequences.values():
            if len(frames["frames"]) >= self.seq_length:
                for i in range(0, len(frames["frames"]) - self.seq_length + 1, self.seq_length):
                    valid.append((frames["frames"][i:i+SEQ_LENGTH],
                                  frames["labels"][i:i+SEQ_LENGTH]))
        return valid

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

# --------------------- Models ---------------------
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

# --------------------- Testing ---------------------
def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DeepfakeDataset(LABELS_FILE, SEQ_LENGTH)
    test_size = int(len(dataset) * TEST_SPLIT)
    _, test_dataset = random_split(dataset, [len(dataset) - test_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    cnn = CNNFeatureExtractor().to(device)
    cnn.eval()

    lstm = DeepfakeLSTM().to(device)
    lstm.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    lstm.eval()

    all_preds, all_labels = [], []

    print("🧪 Starting testing...")
    for frames, labels in tqdm(test_loader, desc="Testing"):
        frames, labels = frames.to(device), labels.to(device)
        b, t, c, h, w = frames.shape
        frames = frames.view(b * t, c, h, w)

        # ⏱️ Time optional
        start = time.time()
        features = cnn(frames).view(b, t, -1)
        end = time.time()
        logging.info(f"Batch CNN feature time: {end - start:.2f}s")

        outputs = lstm(features)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    # 🎯 Print to console
    print(f"✅ Test Accuracy: {acc:.2%}")
    print(f"✅ Test Precision: {precision:.4f}")
    print("📌 Confusion Matrix:\n", cm)
    print("📋 Classification Report:\n", report)

    # 📝 Log results
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Test Precision: {precision:.4f}")
    logging.info("Confusion Matrix:\n" + str(cm))
    logging.info("Classification Report:\n" + report)

if __name__ == "__main__":
    test_model()
