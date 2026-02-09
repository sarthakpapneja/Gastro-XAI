import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm

class SimilaritySearch:
    def __init__(self, model, train_loader, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        self.train_loader = train_loader
        self.features = []
        self.paths = [] # In a real scenario, we'd store paths. Here we store indices.
        self.labels = []
        
        # Hook to get features
        self.feature_extractor = torch.nn.Sequential(*list(model.backbone.children())[:-1]) # ResNet50 minus fc
        
        self._fit()

    def _fit(self):
        print("Extracting features for similarity search...")
        all_features = []
        with torch.no_grad():
            for inputs, targets in tqdm(self.train_loader):
                inputs = inputs.to(self.device)
                feats = self.feature_extractor(inputs)
                feats = feats.view(feats.size(0), -1)
                all_features.append(feats.cpu().numpy())
                self.labels.extend(targets.numpy())
        
        self.features = np.concatenate(all_features, axis=0)
        self.knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.knn.fit(self.features)

    def find_nearest(self, input_tensor, k=5):
        """
        input_tensor: [1, C, H, W]
        """
        with torch.no_grad():
            feat = self.feature_extractor(input_tensor.to(self.device))
            feat = feat.view(feat.size(0), -1).cpu().numpy()
        
        distances, indices = self.knn.kneighbors(feat, n_neighbors=k)
        return indices[0], distances[0]
