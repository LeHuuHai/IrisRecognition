import os
import random

import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import Dataset, DataLoader
from fully_connected import FullyConnected

class FeatureDataset(Dataset):
    def __init__(self, feature_path):
        self.feature_path = feature_path
        self.features = [f for f in os.listdir(feature_path) if f.endswith('.npy')]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        anchor_dir = self.features[idx]
        anchor_path = os.path.join(self.feature_path, anchor_dir)
        anchor = np.load(anchor_path)

        while True:
            positive_dir = random.choice(self.features)
            if positive_dir[0:3] == anchor_dir[0:3] and positive_dir[6] == anchor_dir[6]:
                break
        positive_path = os.path.join(self.feature_path, positive_dir)
        positive = np.load(positive_path)

        while True:
            negative_dir = random.choice(self.features)
            if negative_dir[0:3] != anchor_dir[0:3] or negative_dir[6] == anchor_dir[6]:
                break
        negative_path = os.path.join(self.feature_path, negative_dir)
        negative = np.load(negative_path)

        return anchor, positive, negative

features_path = "features"

def tripetLoss(anchor, positive, negative, margin=1.0):
    # Euclid distance
    pos_distance = F.pairwise_distance(anchor, positive, p=2)
    neg_distance = F.pairwise_distance(anchor, negative, p=2)

    # Tính triplet loss
    losses = F.relu(
        pos_distance - neg_distance + margin)  # Điều kiện: D(anchor, positive) + margin < D(anchor, negative)
    return losses.mean()  # Tính giá trị trung bình loss


def train_triplet_model(model_path, feature_path, batch_size=32, margin=1.0, num_epochs=10):
    dataset = FeatureDataset(feature_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")  # Hiển thị CPU hoặc GPU
    model = FullyConnected(input_dim=7 * 7 * 512, output_dim=128).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for anchor, positive, negative in train_loader:
            optimizer.zero_grad()
            vector_a = model(anchor)
            vector_p = model(positive)
            vector_c = model(negative)
            loss = tripetLoss(vector_a, vector_p, vector_c, margin)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                optimizer.zero_grad()
                vector_a = model(anchor)
                vector_p = model(positive)
                vector_c = model(negative)
                val_loss = tripetLoss(vector_a, vector_p, vector_c, margin)
                running_val_loss += val_loss.item()
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Vẽ biểu đồ Loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Time')
    plt.show()

    # Lưu mô hình
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train_triplet_model("../../module/", 'features/00101_L.npy', num_epochs=10)
