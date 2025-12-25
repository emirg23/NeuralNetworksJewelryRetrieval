import os, random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TripletFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.classes = [
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ]

        self.images = {
            c: os.listdir(os.path.join(root, c)) for c in self.classes
        }

        # Use provided transform or default one
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        
        print("Classes found:", self.classes)
        for c in self.classes:
            print(c, "->", len(self.images[c]), "images")

    def __getitem__(self, idx):
        cls = random.choice(self.classes)
        pos_imgs = random.sample(self.images[cls], 2)
        neg_cls = random.choice([c for c in self.classes if c != cls])
        neg_img = random.choice(self.images[neg_cls])

        def load(c, img):
            return self.transform(
                Image.open(os.path.join(self.root, c, img)).convert("RGB")
            )

        return load(cls, pos_imgs[0]), load(cls, pos_imgs[1]), load(neg_cls, neg_img)

    def __len__(self):
        return 10000
