import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import TripletFolderDataset
from tqdm import tqdm

# -----------------------
# CONFIG
# -----------------------
DATASET_ROOT = "jewelry_dataset/train"
BATCH_SIZE = 8
EPOCHS = 10          
LR = 1e-4
EMBEDDING_DIM = 128
NUM_WORKERS = 0
SAVE_PATH = "models/triplet/triplet.pth"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# -----------------------
# MODEL
# -----------------------
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

# -----------------------
# MAIN
# -----------------------
def main():
    print("starting training")
    print(f"device: {DEVICE}")

    os.makedirs("models/triplet", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = TripletFolderDataset(DATASET_ROOT, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    model = TripletNet(EMBEDDING_DIM).to(DEVICE)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # -----------------------
    # TRAIN LOOP
    # -----------------------
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        print(f"\nepoch {epoch + 1}/{EPOCHS}")

        progress = tqdm(loader, desc="Training", ncols=100)

        for step, (a, p, n) in enumerate(progress):
            a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)

            optimizer.zero_grad()

            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)

            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (step + 1)

            progress.set_postfix(loss=f"{avg_loss:.4f}")

        print(f"epoch {epoch + 1} finished | avg Loss: {avg_loss:.4f}")

    # -----------------------
    # SAVE MODEL
    # -----------------------
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nmodel saved to {SAVE_PATH}")
    print("training completed successfully")

# -----------------------
if __name__ == "__main__":
    main()
