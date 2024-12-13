import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from PIL import Image
from PixelCNN import PixelCNN
import os

def generate_samples(cfg, checkpoint_path, num_samples, device='cuda', label=None, save_dir=None):
    """
    Generates samples using a pretrained PixelCNN model.

    Args:
        cfg (dict): Configuration dictionary with model parameters.
        checkpoint_path (str): Path to the pretrained model checkpoint.
        num_samples (int): Number of samples to generate.
        device (str): Device to run the model on ('cuda' or 'cpu').
        label (int, optional): Label to condition the generation on. If None, random labels are used.
        save_dir (str, optional): Directory to save the generated images. If None, images are not saved.

    Returns:
        samples (torch.Tensor): Generated samples as a tensor.
    """
    # Load the model architecture
    num_labels = cfg.get('num_labels', 10)  # Default to 10 if not specified
    model = PixelCNN(cfg=cfg, num_labels=num_labels)
    model.to(device)

    # Load the pretrained model weights
    if os.path.isfile(checkpoint_path):

        # model.load_state_dict(torch.load(checkpoint_path),map_location=torch.device('cuda'))
        # model.eval()  # Set model to evaluation mode
        # print("Model loaded successfully.")
        # Load the state dict with map_location
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # Load the state dict into the model
        model.load_state_dict(state_dict)

        model.eval()  # Set model to evaluation mode
        print("Model loaded successfully.")
        print("Generating samples, hang tight this may take a few minutes...")
    else:
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'.")

    # Determine the shape of the images to generate
    DATA_CHANNELS = 1  # Grayscale images
    HEIGHT = cfg.get('image_height', 32)  # Update with your image height
    WIDTH = cfg.get('image_width', 192)   # Update with your image width
    shape = (DATA_CHANNELS, HEIGHT, WIDTH)

    # Generate samples
    with torch.no_grad():
        samples = model.sample(shape=shape, count=num_samples, label=label, device=device)

    # Rescale samples to [0, 1] for visualization
    samples = samples.cpu().numpy()
    samples = np.clip(samples, 0, 1)

    # Optionally save or display the samples
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(num_samples):
            img = samples[i].squeeze() * 255  # Convert to [0, 255] grayscale
            img = Image.fromarray(img.astype(np.uint8), mode='L')
            img.save(os.path.join(save_dir, f'sample_{i+1}.png'))

    return samples


def main(checkpoint_path, label_count):
    # Configuration dictionary (ensure it matches the one used during training)
    cfg = {
        "epochs": 50,
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
        "cuda": True,
        "image_height": 28,
        "image_width": 168,
        "num_labels": label_count,  # Ensure this matches your dataset
    }

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["cuda"] else "cpu")

    # Path to your saved model checkpoint

    # Number of samples to generate
    num_samples = 3

    # (Optional) Label to condition on. Replace with a valid label index or set to None.
    label = None

    # Directory to save generated images
    save_dir = './generated_samples'

    # Generate samples
    samples = generate_samples(cfg, checkpoint_path, num_samples, device=device, label=label, save_dir=save_dir)

    # Display the samples
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(samples[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.show()

if __name__ == '__main__':
    sample = input("Would you like to sample a model from the full dataset or the small dataset? (full/small) ")
    if sample == "full":
        path_to_model = "Trained_models/full_data.pth"
        label_count = 81
    else:
        path_to_model = "Trained_models/small_data.pth"
        label_count = 10
    main(path_to_model, label_count)
    print("Samples successfully saved to ./generated_samples")

