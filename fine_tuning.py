import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
from sklearn.metrics import accuracy_score
import logging
from collections import Counter
from tqdm import tqdm
import copy

# 🔧 Config
LABELS_FILE = "C:/Users/HP/Desktop/deepfake/frames_labels/labels.txt"  # Update the file path accordingly
SEQ_LENGTH = 16
BATCH_SIZE = 4
LEARNING_RATE_CNN = 1e-5
LEARNING_RATE_LSTM = 1e-4
EPOCHS = 2
VALIDATION_SPLIT = 0.2
PATIENCE = 5
LOG_FILE = "training_output_log.txt"
VAL_ACC_FILE = "validation_accuracy.txt"

# 🧠 CNN Feature Extractor (Updated to use `weights`)
from torchvision.models import resnet18, ResNet18_Weights

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Use weights argument to avoid deprecated warning
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)

# 🧠 LSTM Classifier
class DeepfakeLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=1, num_classes=2):
        super(DeepfakeLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# 📦 Dataset class with missing frame handling
class DeepfakeDataset(Dataset):
    def __init__(self, labels_file, seq_length):
        self.seq_length = seq_length
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        with open(labels_file, "r") as f:
            lines = f.readlines()
            current_clip = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 2 or parts[1] == "Label":
                    continue
                path = parts[0]
                label = int(parts[1])
                current_clip.append((path, label))
                if len(current_clip) == seq_length:
                    labels = [lbl for _, lbl in current_clip]
                    if all(lbl == labels[0] for lbl in labels):
                        self.samples.append(([p for p, _ in current_clip], labels[0]))
                    current_clip = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        frames = []

        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                frames.append(self.transform(img))
            except Exception as e:
                print(f"⚠️ Skipping missing/unreadable frame: {p} ({e})")
                continue

        if len(frames) < self.seq_length:
            # Try another valid sample (next index)
            return self.__getitem__((idx + 1) % len(self))

        return torch.stack(frames), label

# 🚂 Training
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')
    with open(VAL_ACC_FILE, "w") as val_log:
        val_log.write("Validation Accuracy per Epoch\n")

    dataset = DeepfakeDataset(LABELS_FILE, SEQ_LENGTH)
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    cnn = CNNFeatureExtractor().to(device)
    lstm = DeepfakeLSTM().to(device)

    # 🔓 Unfreeze all CNN layers for full fine-tuning
    for param in cnn.parameters():
        param.requires_grad = True

    # 🔹 Class weights for imbalance
    all_labels = [label for _, label in dataset.samples]
    label_counts = Counter(all_labels)
    total = sum(label_counts.values())
    weights = [total / label_counts[i] for i in range(len(label_counts))]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    # 🔹 Loss & Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam([
        {'params': cnn.parameters(), 'lr': LEARNING_RATE_CNN},
        {'params': lstm.parameters(), 'lr': LEARNING_RATE_LSTM}
    ])

    # 🔧 Load pre-trained model weights if available
    PRETRAINED_MODEL_PATH =r'C:\Users\HP\Desktop\deepfake\best_lstm_model.pth'  # Provide the path to your saved model here

    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"🔧 Loading pre-trained model from {PRETRAINED_MODEL_PATH}")
        checkpoint = torch.load(PRETRAINED_MODEL_PATH)
        # Load state_dict with strict=False to ignore mismatched keys
        try:
            lstm.load_state_dict(checkpoint, strict=False)
            print("🔧 Pre-trained model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Error loading model state dict: {e}")

    best_val_acc = 0.0
    best_model_wts = None
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\n🔁 Epoch {epoch + 1}/{EPOCHS}")
        cnn.train()
        lstm.train()
        total_loss = 0
        all_preds, all_true = [], []

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
            all_true.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_true, all_preds)
        print(f"📊 Train Loss: {total_loss:.4f}, Acc: {train_acc:.2%}")

        # 🔍 Validation
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
        print(f"✅ Val Acc: {val_acc:.2%}")

        # 📝 Logging
        logging.info(f"Epoch {epoch+1}")
        logging.info(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        with open(VAL_ACC_FILE, "a") as val_log:
            val_log.write(f"Epoch {epoch+1}: {val_acc:.4f}\n")

        # 💾 Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(lstm.state_dict())
            torch.save(best_model_wts, "best_lstm_model.pth")
            print("💾 Best model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("🛑 Early stopping triggered.")
                break

    print(f"🏁 Training complete. Best Val Accuracy: {best_val_acc:.2%}")

if __name__ == "__main__":
    train_model()
