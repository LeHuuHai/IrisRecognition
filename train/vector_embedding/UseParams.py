import os

import numpy as np
import torch
import torch.nn.functional as F

from fully_connected import FullyConnected


class UseParam:
    def __init__(self, model_path, pca_model_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FullyConnected(input_dim=1500).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.pca_model = torch.load(pca_model_path)

    def embedding(self, vector_feature):
        mean = self.pca_model["mean"]
        components = self.pca_model["components"]
        vector_feature_mean = vector_feature - mean
        vector_feature_pca = np.dot(vector_feature_mean, components.T)
        vector_feature_pca = torch.tensor(vector_feature_pca, dtype=torch.float32)
        vector_feature_pca = vector_feature_pca.to(self.device)
        return self.model(vector_feature_pca)


def embedding(embedded_all_file):
    embed = UseParam("vector_embedding_14-1_pca.pth", "pca_pretrained_model_17-1.pth")
    all_vector_label_pairs = []
    num_vector_successfully = 0
    for i in range(1, 200):
        for j in range(1, 11):
            for suffix in ['L', 'R']:
                vector_feature_path = f'./features/{i:03}{j:02}_{suffix}.npy'
                if not os.path.exists(vector_feature_path):
                    print(f"Warning: The file {vector_feature_path} does not exist.")
                    continue
                vector_feature = torch.tensor(np.load(vector_feature_path), dtype=torch.float32).reshape(1, -1)
                embedded_vector = embed.embedding(vector_feature)
                if isinstance(embedded_vector, torch.Tensor):
                    embedded_vector = embedded_vector.cpu().detach().numpy()
                label = f"{i:03}{j:02}{suffix}"
                vector_label_pair = np.column_stack((embedded_vector, label))
                all_vector_label_pairs.append(vector_label_pair)
                print(f"Append {vector_feature_path} successfully")
                num_vector_successfully+=1
    all_vector_label_pairs = np.vstack(all_vector_label_pairs)
    np.save(embedded_all_file, all_vector_label_pairs)
    print(f"Saved {num_vector_successfully} vector and label pairs to {embedded_all_file}")


def calculator(embedded_all_file):
    vector_label_pairs = np.load(embedded_all_file)
    print(f"Loaded {vector_label_pairs.shape[0]} pairs of vectors and labels.")
    vectors = vector_label_pairs[:, :-1]
    labels = vector_label_pairs[:, -1]
    vectors = vectors.astype(np.float32)
    for i in range(200):
        tensor_0 = torch.tensor(vectors[0]).unsqueeze(0)
        tensor_i = torch.tensor(vectors[i]).unsqueeze(0)
        distance =  F.pairwise_distance(tensor_0, tensor_i, p=2)
        print(f"distance between vector 0 ({labels[0]}) and vector {i} ({labels[i]}) {labels[0][0:3]==labels[i][0:3] and labels[0][-1]==labels[i][-1]}: {distance.item()}")


if __name__ == "__main__":
    # embedding("./db.npy")
    calculator("./db.npy")

