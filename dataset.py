import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import rasterio
import torch
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from torch.utils.data import random_split

class Inria(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        # Define paths for images and masks
        self.img_dir = os.path.join(os.getcwd(), img_dir)
        self.mask_dir = os.path.join(os.getcwd(), mask_dir)

        # Get sorted list of image and mask files
        self.imgs = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.tif', '.tiff'))])
        self.masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(('.png', '.jpg', '.tif', 'tiff'))])

        # Check that image and mask counts match
        if len(self.imgs) != len(self.masks):
            raise ValueError("Mismatch between number of images and masks!")

        self.transform = transform or Compose([Resize((512, 512)), ToTensor()])
        self.target_transform = target_transform or Compose([Resize((512, 512)), ToTensor()])
        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Load RGB image and grayscale mask
        with rasterio.open(os.path.join(self.img_dir, self.imgs[idx])) as img_file:
            image = img_file.read([1, 2, 3])  # Read RGB bands, shape [3, H, W]
            image = image.transpose(1, 2, 0)  # Convert to [H, W, C] for PIL.Image

        with rasterio.open(os.path.join(self.mask_dir, self.masks[idx])) as mask_file:
            label = mask_file.read(1)  # Read first band, shape [H, W]

        # Convert numpy arrays to PIL.Image
        image = Image.fromarray(image.astype('uint8'))  # Ensure the dtype is uint8 for RGB
        label = Image.fromarray(label.astype('uint8'))  # Grayscale mask

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
    # def __getitem__(self, idx):
    #         # Load RGB image and grayscale mask
    #         with rasterio.open(os.path.join(self.img_dir, self.imgs[idx])) as img_file:
    #             image = img_file.read([1, 2, 3])  # Read RGB bands, shape [3, H, W]
    #         with rasterio.open(os.path.join(self.mask_dir, self.masks[idx])) as mask_file:
    #             label = mask_file.read(1)  # Read first band, shape [H, W]

    #         # Apply transformations (if any)
    #         if self.transform:
    #             image = self.transform(image)
    #         if self.target_transform:
    #             label = self.target_transform(label)

    #         # Convert to tensors
    #         image = torch.tensor(image, dtype=torch.float32)  # Shape: [3, H, W]
    #         label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # Shape: [1, H, W]

    #         return image, label

    # def __getitem__(self, idx):
    #     # Validate index
    #     if idx >= len(self) or idx < 0:
    #         raise IndexError(f"Index {idx} out of range for dataset with {len(self)} samples.")

    #     # Get image and mask paths
    #     img_path = os.path.join(self.img_dir, self.imgs[idx])
    #     mask_path = os.path.join(self.mask_dir, self.masks[idx])

    #     # Load image and mask
    #     image = rasterio.open(img_path)
    #     label = rasterio.open(mask_path)
    #     image = image.read()
    #     label = label.read()

    #     # Apply transformations if specified
    #     if self.transform:
    #         image = self.transform(image)
    #     if self.target_transform:
    #         label = self.target_transform(label)

    #     return image, label


# train_dataset = Inria(img_dir="inria/AerialImageDataset/train/images", mask_dir="inria/AerialImageDataset/train/gt", transform=ToTensor(), target_transform=ToTensor())
# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

dataset = Inria(img_dir="inria/AerialImageDataset/train/images", mask_dir="inria/AerialImageDataset/train/gt")

# Split dataset into train (80%) and test (20%) subsets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# # Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch size: {train_features.size()}")
# print(f"Labels batch size: {train_labels.size()}")
# print(f"Train feature shape: {train_features.shape}")  # Should be [B, 3, H, W]
# print(f"Train label shape: {train_labels.shape}")  # Should be [B, 1, H, W]

# # Prepare the image for display
# img = train_features[0].permute(0, 2, 1).numpy()  # Change shape to (H, W, C)
# label = train_labels[0].squeeze().numpy()  # Remove channel dimension for mask

# # Display the image
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.title("Image")
# plt.axis("off")

# # Display the label
# plt.subplot(1, 2, 2)
# plt.imshow(label, cmap="gray")
# plt.title("Label")
# plt.axis("off")

# plt.show()