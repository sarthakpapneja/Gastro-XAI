import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import glob

class KvasirDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset='train', split_ratio=0.8, seed=42):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            subset (string): 'train' or 'val'
            split_ratio (float): Ratio of training data
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        # Filter out non-directory files like .DS_Store
        self.classes = [c for c in self.classes if os.path.isdir(os.path.join(root_dir, c))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.targets = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            img_paths = glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.png"))
            img_paths.sort() # Ensure deterministic split
            
            # Split
            split_idx = int(len(img_paths) * split_ratio)
            if subset == 'train':
                selected_imgs = img_paths[:split_idx]
            else:
                selected_imgs = img_paths[split_idx:]
            
            self.images.extend(selected_imgs)
            self.targets.extend([self.class_to_idx[cls_name]] * len(selected_imgs))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target

def get_data_loaders(data_dir, batch_size=32, input_size=224):
    # Standard Kvasir v2 transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = KvasirDataset(data_dir, transform=data_transforms['train'], subset='train')
    val_dataset = KvasirDataset(data_dir, transform=data_transforms['val'], subset='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_dataset.classes
