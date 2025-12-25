import torch
import numpy as np
import pandas as pd
from PIL import Image


class ArcFaceRetriever:
    def __init__(self, model, model_path, embedding_csv, image_csv, device):
        self.device = device
        self.model = model.to(device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        self.model.eval()

        self.embeddings = pd.read_csv(embedding_csv).values
        self.image_paths = pd.read_csv(image_csv).values.flatten()

    def get_top_k(self, image, transform, k=5):
        image = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            query_emb = self.model(image).cpu().numpy()

        distances = np.linalg.norm(self.embeddings - query_emb, axis=1)
        topk_idx = np.argsort(distances)[:k]

        return [self.image_paths[i] for i in topk_idx]
