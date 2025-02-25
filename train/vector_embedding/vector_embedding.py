import os
import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import Dataset, DataLoader
from fully_connected import FullyConnected

from sklearn.decomposition import PCA


class FeatureDataset(Dataset):
    def __init__(self, feature_path):
        self.features = [f for f in os.listdir(feature_path) if f.endswith('.npy')]
        # Tải các vector đặc trưng từ các tệp .npy
        self.vector_feature = torch.tensor(
            [np.load(os.path.join(feature_path, f)) for f in self.features], dtype=torch.float32
        )
        pca = PCA(n_components=1500)
        self.vector_feature = pca.fit_transform(self.vector_feature)
        self.vector_feature = torch.tensor(self.vector_feature, dtype=torch.float32)

        self.distance_matrix = torch.cdist(self.vector_feature, self.vector_feature, p=2)
        self.label = [f[0:3] + f[6] for f in self.features]
        self.tripleSet = self.createTripleSet()

    def createTripleSet(self):
        tripleSet = []
        for idx, vector_a in enumerate(self.vector_feature):
            distance_vector_map = [[self.distance_matrix[idx][i], self.label[i], i] for i in range(len(self.distance_matrix[idx]))]

            # Tìm các chỉ số positive candidates dựa trên nhãn
            positive_mask = torch.tensor(
                [self.label[idx] == dvm[1] for dvm in distance_vector_map]
            )
            positive_candidate = torch.where(positive_mask)[0]
            distance_vector_map_p = [distance_vector_map[i] for i in positive_candidate]
            if len(distance_vector_map_p) > 0:
                distance_vector_map_p.sort(key=lambda x: x[0], reverse=True)
                idxs_p = distance_vector_map_p[0][2]
                vectors_p = [self.vector_feature[idxs_p]]
            else:
                continue

            # Tìm các chỉ số negative candidates dựa trên nhãn
            negative_mask = torch.tensor(
                [self.label[idx] != dvm[1] for dvm in distance_vector_map]
            )
            negative_candidate = torch.where(negative_mask)[0]
            distance_vector_map_n = [distance_vector_map[i] for i in negative_candidate]
            if len(distance_vector_map_n) > 0:
                distance_vector_map_n.sort(key=lambda x: x[0])
                idxs_n = [distance_vector_map_n[0][2],
                            distance_vector_map_n[1][2],
                            distance_vector_map_n[2][2],
                            distance_vector_map_n[3][2],
                            distance_vector_map_n[4][2]]
                vectors_n = self.vector_feature[idxs_n]
            else:
                continue

            # 1 positive, 5 negative
            for i in range(5):
                tripleSet.append((vector_a, vectors_p[0], vectors_n[i]))

        return tripleSet

    def __len__(self):
        return len(self.tripleSet)

    def __getitem__(self, idx):
        return self.tripleSet[idx]

def tripetLoss(anchor, positive, negative, margin=0.2):
    # Euclid distance
    pos_distance = F.pairwise_distance(anchor, positive, p=2)
    neg_distance = F.pairwise_distance(anchor, negative, p=2)

    # Tính triplet loss
    losses = F.relu(
        pos_distance - neg_distance + margin)  # Điều kiện: D(anchor, positive) + margin < D(anchor, negative)
    return losses.mean()  # Tính giá trị trung bình loss


def train_triplet_model(feature_path, batch_size=32, margin=0.2, num_epochs=10):
    dataset = FeatureDataset(feature_path)
    print(len(dataset))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")  # Hiển thị CPU hoặc GPU
    model = FullyConnected(input_dim= 1500, output_dim=128).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
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
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
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
    torch.save(model.state_dict(), "vector_embedding_14-1_pca.pth")

if __name__ == "__main__":
    train_triplet_model("./features", num_epochs=100)
