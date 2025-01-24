import os

from sklearn.decomposition import PCA
import numpy as np
import torch

features = [f for f in os.listdir("./features") if f.endswith('.npy')]
# Tải các vector đặc trưng từ các tệp .npy
vector_feature = torch.tensor(
    [np.load(os.path.join("./features", f)) for f in features], dtype=torch.float32
)
print(vector_feature.shape)

pca = PCA(n_components=1500)
X_train_centered = vector_feature - torch.mean(vector_feature)
X_train_pca = pca.fit_transform(X_train_centered)

mean = torch.mean(vector_feature)
components = pca.components_

# Lưu vào tệp
torch.save({"mean": mean, "components": components}, "pca_pretrained_model_17-1.pth")
print(mean, components)
print("PCA pre-trained model saved!")
