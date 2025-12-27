import torch
from torchvision import transforms
from PIL import Image
import os

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "models/triplet/triplet.pth"
EMBEDDING_DIM = 128
EMBEDDINGS_PATH = "embeddings.pt"
IMAGE_PATH = "/Users/emirg/Desktop/test.jpg"
TOP_K = 5

# MODEL (TripletNet)
import torch.nn as nn
from torchvision import models

class TripletNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

# MODELİ YÜKLE
model = TripletNet(EMBEDDING_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print(f"using device: {DEVICE}")

# EMBEDDINGS VE FILEPATHS YÜKLE
data = torch.load(EMBEDDINGS_PATH, map_location=DEVICE)
all_embeddings = data["embeddings"]  # tensor: num_samples x embedding_dim
all_filepaths = data["filepaths"]    # list: num_samples

print(f"loaded {len(all_filepaths)} embeddings")

# TEST RESMİ HAZIRLA
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# Test resmini göster
print(f"query image:")
img.show()

# TEST EMBEDDING ÇIKAR
with torch.no_grad():
    test_emb = model(img_tensor)

# COSINE SIMILARITY HESAPLA
from torch.nn.functional import cosine_similarity

# all_embeddings'i device'a taşı
all_embeddings = all_embeddings.to(DEVICE)

similarities = cosine_similarity(test_emb, all_embeddings)  # shape: (num_samples,)
topk_vals, topk_idx = torch.topk(similarities, TOP_K)

print(f"\ntop {TOP_K} most similar images:\n")

# EN BENZER K IMAGE'İ GÖSTER
for i, idx in enumerate(topk_idx):
    idx = idx.item()
    img_path = all_filepaths[idx]
    similarity = topk_vals[i].item()
    
    # Class name'i path'ten al
    class_name = os.path.basename(os.path.dirname(img_path))
    
    print(f"{i+1}. class: {class_name} | Similarity: {similarity:.4f}")
    print(f"   path: {img_path}")
    
    # Resmi aç ve göster
    try:
        retrieved_img = Image.open(img_path).convert("RGB")
        retrieved_img.show()
    except Exception as e:
        print(f"   could not open image: {e}")
    
    print()
