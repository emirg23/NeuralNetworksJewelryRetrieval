import torch
import pandas as pd
import os
from torchvision import transforms
from PIL import Image

from models.arcface_model import ArcFaceModel
from dataset import ClassFolderDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models_output/arcface_epoch_10.pth"
DATASET_ROOT = "jewelry_dataset/train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = ClassFolderDataset(DATASET_ROOT, transform)
model = ArcFaceModel(embedding_dim=128).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

embeddings = []
paths = []

with torch.no_grad():
    for path, label in dataset.samples:
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).to(DEVICE)

        emb = model(img).cpu().numpy()[0]
        embeddings.append(emb)
        paths.append(path)

pd.DataFrame(embeddings).to_csv("embeddings.csv", index=False)
pd.DataFrame(paths).to_csv("image_paths.csv", index=False)

print("arcface embeddings generated.")
