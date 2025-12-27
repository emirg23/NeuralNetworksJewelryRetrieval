import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# CONFIG
DATASET_ROOT = "jewelry_dataset/train"
BATCH_SIZE = 8
EMBEDDING_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "models/triplet/triplet.pth"
SAVE_PATH = "embeddings.pt"
NUM_WORKERS = 0

# MODEL DEFINITION
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

# SIMPLE IMAGE DATASET
class SimpleImageDataset:
    def __init__(self, root, transform=None):
        self.transform = transform
        self.image_paths = []
        
        for class_name in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_name)
            if os.path.isdir(class_path):
                for img_name in sorted(os.listdir(class_path)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(class_path, img_name)
                        self.image_paths.append(full_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path

# TRANSFORMS & DATASET
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = SimpleImageDataset(DATASET_ROOT, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# LOAD MODEL
model = TripletNet(EMBEDDING_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print(f"using device: {DEVICE}")
print(f"total images: {len(dataset)}")

# EXTRACT EMBEDDINGS
all_embeddings = []
all_filepaths = []

with torch.no_grad():
    for imgs, paths in tqdm(loader, desc="extracting embeddings"):
        imgs = imgs.to(DEVICE)
        embeddings = model(imgs)
        
        all_embeddings.append(embeddings.cpu())
        all_filepaths.extend(paths)  # paths is a tuple/list from the batch

# Concatenate all embeddings
all_embeddings = torch.cat(all_embeddings, dim=0)

# Save
torch.save({
    "embeddings": all_embeddings,
    "filepaths": all_filepaths
}, SAVE_PATH)

print(f"embeddings saved to {SAVE_PATH}")
print(f"shape: {all_embeddings.shape}")
print(f"total files: {len(all_filepaths)}")
print(f"example path: {all_filepaths[0]}")
