from PIL import Image
import torchvision.transforms as transforms
import torch
import subprocess

from models.arcface_model import ArcFaceModel
from retrieve import ArcFaceRetriever


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

model = ArcFaceModel(embedding_dim=128)

retriever = ArcFaceRetriever(
    model=model,
    model_path="models_output/arcface_epoch_10.pth",
    embedding_csv="embeddings.csv",
    image_csv="image_paths.csv",
    device=DEVICE
)

query_img = Image.open(
    "/Users/emirg/Desktop/test.jpg"
).convert("RGB")

results = retriever.get_top_k(query_img, transform, k=5)

print("🔍 Similar items:")

for i, path in enumerate(results, 1):
    print(f"{i}. {path}")
    subprocess.run(["open", path])

print("opened images")
