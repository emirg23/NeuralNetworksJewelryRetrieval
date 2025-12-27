import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

from dataset import ClassFolderDataset
from models.arcface_model import ArcFaceModel
from losses.arcface_loss import ArcFaceLoss


# CONFIG
DATASET_ROOT = "jewelry_dataset/train"
BATCH_SIZE = 32
EPOCHS = 10
EMBED_DIM = 128
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MAIN TRAIN LOOP
def main():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ClassFolderDataset(DATASET_ROOT, transform)
    num_classes = len(dataset.class_to_idx)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    model = ArcFaceModel(embedding_dim=EMBED_DIM).to(DEVICE)

    criterion = ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=EMBED_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    os.makedirs("models_output", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        loop = tqdm(loader, desc=f"epoch [{epoch+1}/{EPOCHS}]")

        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            embeddings = model(images)
            loss = criterion(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"epoch {epoch+1} finished | Avg Loss: {avg_loss:.4f}")

        torch.save(
            model.state_dict(),
            f"models_output/arcface_epoch_{epoch+1}.pth"
        )

    print("training completed successfully.")


# ENTRY POINT
if __name__ == "__main__":
    main()
