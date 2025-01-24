import os

import numpy as np
import torch
import torch.nn.functional as F

from fully_connected import FullyConnected
from features_extraction import FeaturesExtraction


class Embedd:
    def __init__(self, model_path, pca_model_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FullyConnected(input_dim=1500).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.pca_model = torch.load(pca_model_path)
        self.extraction = FeaturesExtraction()

    def embedding(self, normalization_img):
        feature_vector = torch.tensor(self.extraction.extraction(normalization_img))
        mean = self.pca_model["mean"]
        components = self.pca_model["components"]
        vector_feature_mean = feature_vector - mean
        vector_feature_pca = np.dot(vector_feature_mean, components.T)
        vector_feature_pca = torch.tensor(vector_feature_pca, dtype=torch.float32)
        vector_feature_pca = vector_feature_pca.to(self.device)
        return self.model(vector_feature_pca)


