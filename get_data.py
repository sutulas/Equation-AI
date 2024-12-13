### Custom euqation loader

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image
import random


equations = {}

def quantisize(image, levels):
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = (image * (levels - 1)).astype(np.int64)  # Quantize to [0, levels - 1]
    return image

def duplicate_image_horizontal(image):
    """Duplicate the image horizontally to make it 6 times as wide."""
    return torch.cat([image] * 6, dim=-1)

def generate_equation_image(digit_images, equation):
    """Generate a single image representing an equation, with images concatenated horizontally."""
    equation_image = Image.new('L', (28 * 6, 28), color=255)
    for i, char in enumerate(equation):
        img = random.choice(digit_images[char])  # Randomly pick an image for the digit/symbol
        equation_image.paste(img, (i * 28, 0))  # Paste at the correct position
    return equation_image

def generate_equation_dataset(digit_images, output_dir, num_samples):
    """Generate images for equations of the form '1 + j = x' and save them to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    num = 0
    # i = 1  # Fix i = 1
    for i in range(1, 10):  # i ranges from 1 to 9
      for j in range(0, 10):  # j ranges from 0 to 9
          result = f"{i + j:02d}"  # Two-digit result format
          equation = f"{i}+{j}={result}"
          equations[equation] = num
          num += 1
          equation_folder = os.path.join(output_dir, equation)
          os.makedirs(equation_folder, exist_ok=True)
          for sample_num in range(num_samples):
              img = generate_equation_image(digit_images, equation)
              img.save(os.path.join(equation_folder, f"{sample_num}.png"))
    print(f"Total equations generated: {len(equations)}")

class EquationDataset(Dataset):
    """Custom dataset for loading generated equation images."""
    def __init__(self, data_dir, color_levels, transform=None):
        self.data_dir = data_dir
        self.color_levels = color_levels
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_index = {}
        self.index_to_label = {}

        # Collect all unique labels
        labels_set = set()
        for label_folder in os.listdir(data_dir):
            labels_set.add(label_folder)
        labels_list = sorted(labels_set)
        print(labels_list)
        self.num_labels = len(labels_list)  # Set num_labels

        # Create label to index mapping
        for idx, label in enumerate(labels_list):
            self.label_to_index[label] = idx
            self.index_to_label[idx] = label

        # Load images and labels
        for label_folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, label_folder)
            if os.path.isdir(folder_path):
                idx = self.label_to_index[label_folder]
                for img_name in os.listdir(folder_path):
                    self.image_paths.append(os.path.join(folder_path, img_name))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



def get_custom_loader(batch_size, color_levels, data_dir="./Data/raw_data", 
                     num_train_samples=6000, num_test_samples=1000, force_regenerate=False):
    """
    Generate and load the equation dataset with transformations.
    
    Args:
        ...
        force_regenerate (bool): If True, regenerate the datasets even if they exist
    """
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255).long()),
        transforms.Lambda(lambda x: x.div(255 / (color_levels - 1)).long()),
    ])

    train_output_dir = './Data/generated_train'
    test_output_dir = './Data/generated_test'

    # Only generate datasets if they don't exist or if force_regenerate is True
    if force_regenerate or not (os.path.exists(train_output_dir) and os.path.exists(test_output_dir)):
        print("Generating new datasets...")
        # Generate datasets
        digit_images = load_images(data_dir, (28, 28))  # Load all digit and symbol images into memory
        
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

        generate_equation_dataset(digit_images, train_output_dir, num_train_samples)
        generate_equation_dataset(digit_images, test_output_dir, num_test_samples)
    else:
        print("Using existing datasets...")

    # Load datasets
    train_dataset = EquationDataset(train_output_dir, color_levels, transform=transform)
    test_dataset = EquationDataset(test_output_dir, color_levels, transform=transform)

    num_labels = train_dataset.num_labels  # Get num_labels from the dataset

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True, 
        num_workers=4, 
        pin_memory=True
    )

    return train_loader, test_loader, 28, 168, num_labels


def load_images(data_dir, target_size):
    """
    Load images and organize by label in a dictionary.
    Each key is a digit or symbol (0-9, x, =) and value is a list of PIL Images.
    """
    from PIL import ImageOps
    image_dict = {}
    transform = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])

    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            image_dict[label] = []
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                try:
                    img = Image.open(img_path).convert("L")
                    width, height = img.size
                    max_side = max(width, height)
                    new_img = Image.new('L', (max_side, max_side), color=255)
                    new_img.paste(img, ((max_side - width) // 2, (max_side - height) // 2))

                    new_img = transform(new_img)
                    new_img = transforms.ToPILImage()(new_img)
                    image_dict[label].append(new_img)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return image_dict
global_train_loader, global_test_loader, global_HEIGHT, global_WIDTH, global_num_labels = get_custom_loader(
        32, 2, num_train_samples=6000, num_test_samples=1000
    )
