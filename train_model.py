from functools import total_ordering
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import torchvision
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
import argparse
import os
from tqdm import tqdm
from PixelCNN import PixelCNN
from get_data import global_train_loader, global_test_loader, global_HEIGHT, global_WIDTH, global_num_labels


TRAIN_DATASET_ROOT = '.data/train/'
TEST_DATASET_ROOT = '.data/test/'

MODEL_PARAMS_OUTPUT_DIR = 'model'
MODEL_PARAMS_OUTPUT_FILENAME = 'params.pth'

TRAIN_SAMPLES_DIR = 'train_samples'

import re

# def tokenize_equation(equation):
#     """Tokenizes an equation string into individual tokens of numbers and operators."""
#     # Split the equation into tokens (numbers, operators, etc.)
#     tokens = re.findall(r'\d+|\+|=|-|\*|\/', equation)
#     return tokens

def train(cfg, model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg["epochs"]}'):
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)

        # Ensure images are quantized correctly
        images = images.long()
        inputs = images.float() / (cfg["color_levels"] - 1)

        # Debugging statements


        optimizer.zero_grad()
        outputs = model(inputs, labels)

        # Reshape outputs and targets
        outputs = outputs.permute(0, 2, 3, 4, 1)  # [batch_size, channels, height, width, color_levels]
        outputs = outputs.contiguous().view(-1, cfg["color_levels"])  # [N, color_levels]
        targets = images.view(-1)  # [N]


        loss = F.cross_entropy(outputs, targets)
        total_loss += loss.item()
        num_batches += 1

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=cfg["max_norm"])
        optimizer.step()

    avg_train_loss = total_loss / num_batches
    # avg_train_loss /= len(test_loader.dataset) * height * width * 3
    print(f"Epoch {epoch + 1}, Average Train loss: {avg_train_loss}")

    scheduler.step()



def test_and_sample(cfg, model, device, test_loader, height, width, losses, params, epoch):
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            images = images.long()
            inputs = images.float() / (cfg["color_levels"] - 1)
            outputs = model(inputs, labels)

            outputs = outputs.permute(0, 2, 3, 4, 1)
            outputs = outputs.contiguous().view(-1, cfg["color_levels"])
            targets = images.view(-1)

            test_loss += F.cross_entropy(outputs, targets, reduction='sum').item()

    test_loss /= len(test_loader.dataset) * height * width * 3  # Normalize by total number of pixels
    
    print(f"Epoch {epoch + 1}, Test loss: {test_loss}")
    losses.append(test_loss)
    params.append(model.state_dict())

    # Generate samples
    samples = model.sample((1, height, width), cfg["epoch_samples"], label=None,device=device)
    save_samples(samples, TRAIN_SAMPLES_DIR, f'epoch{epoch + 1}_samples.png')

def save_samples(samples, directory, filename):
    os.makedirs(directory, exist_ok=True)
    grid = torchvision.utils.make_grid(samples, nrow=1, padding=0)
    torchvision.utils.save_image(grid, os.path.join(directory, filename))
    print(f"Samples saved to {os.path.join(directory, filename)}")


# Define a global variable to store the saved model path
SAVED_MODEL_PATH = None

def main():
    # Configuration dictionary
    cfg = {
        "epochs": 15,
        "batch_size": 64,
        "dataset": 'custom_equations',
        "causal_ksize": 7,
        "hidden_ksize": 7,
        "color_levels": 2,
        "hidden_fmaps": 64,
        "out_hidden_fmaps": 32,
        "hidden_layers": 9,
        "learning_rate": 0.0001,
        "weight_decay": 0.00005,
        "max_norm": 1.0,
        "epoch_samples": 3,
        "cuda": True
    }

    # Load data
    train_loader, test_loader, HEIGHT, WIDTH, num_labels = global_train_loader, global_test_loader, global_HEIGHT, global_WIDTH, global_num_labels
    print(f'num labels: {num_labels}')
    # Initialize model with num_labels
    model = PixelCNN(cfg=cfg, num_labels=num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["cuda"] else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    losses = []
    params = []

    for epoch in range(cfg["epochs"]):
        train(cfg, model, device, train_loader, optimizer, scheduler, epoch)
        test_and_sample(cfg, model, device, test_loader, HEIGHT, WIDTH, losses, params, epoch)
        if not os.path.exists(MODEL_PARAMS_OUTPUT_DIR):
            os.mkdir(MODEL_PARAMS_OUTPUT_DIR)
        MODEL_PARAMS_OUTPUT_FILENAME = f'pixelcnn_equations{epoch}.pth'
        SAVED_MODEL_PATH = os.path.join(MODEL_PARAMS_OUTPUT_DIR, MODEL_PARAMS_OUTPUT_FILENAME)
        torch.save(params[np.argmin(np.array(losses))], SAVED_MODEL_PATH)


        print(f"Model saved to {SAVED_MODEL_PATH}")

    if IN_COLAB:
        files.download(SAVED_MODEL_PATH)

    # Save the best model



if __name__ == '__main__':
    main()
