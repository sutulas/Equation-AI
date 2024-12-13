import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CroppedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(CroppedConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = super(CroppedConv2d, self).forward(x)

        kernel_height, _ = self.kernel_size
        res = x[:, :, 1:-kernel_height, :]
        shifted_up_res = x[:, :, :-kernel_height-1, :]

        return res, shifted_up_res


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask_type, data_channels, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        assert mask_type in ['A', 'B'], 'Invalid mask type.'

        out_channels, in_channels, height, width = self.weight.size()
        yc, xc = height // 2, width // 2

        mask = np.ones(self.weight.size(), dtype=np.float32)
        mask[:, :, yc + 1:, :] = 0
        mask[:, :, yc, xc + (1 if mask_type == 'B' else 0):] = 0

        self.register_buffer('mask', torch.from_numpy(mask))

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class CausalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, data_channels):
        super(CausalBlock, self).__init__()
        self.split_size = out_channels

        self.v_conv = CroppedConv2d(in_channels,
                                    2 * out_channels,
                                    (kernel_size // 2 + 1, kernel_size),
                                    padding=(kernel_size // 2 + 1, kernel_size // 2))
        self.v_fc = nn.Conv2d(in_channels,
                              2 * out_channels,
                              (1, 1))
        self.v_to_h = nn.Conv2d(2 * out_channels,
                                2 * out_channels,
                                (1, 1))

        self.h_conv = MaskedConv2d(in_channels,
                                   2 * out_channels,
                                   (1, kernel_size),
                                   mask_type='A',
                                   data_channels=data_channels,
                                   padding=(0, kernel_size // 2))
        self.h_fc = MaskedConv2d(out_channels,
                                 out_channels,
                                 (1, 1),
                                 mask_type='A',
                                 data_channels=data_channels)

    def forward(self, image):
        v_out, v_shifted = self.v_conv(image)
        v_out += self.v_fc(image)
        v_out_tanh, v_out_sigmoid = torch.split(v_out, self.split_size, dim=1)
        v_out = torch.tanh(v_out_tanh) * torch.sigmoid(v_out_sigmoid)

        h_out = self.h_conv(image)
        v_shifted = self.v_to_h(v_shifted)
        h_out += v_shifted
        h_out_tanh, h_out_sigmoid = torch.split(h_out, self.split_size, dim=1)
        h_out = torch.tanh(h_out_tanh) * torch.sigmoid(h_out_sigmoid)
        h_out = self.h_fc(h_out)

        return v_out, h_out


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, data_channels):
        super(GatedBlock, self).__init__()
        self.split_size = out_channels

        # Vertical stack convolution
        self.v_conv = CroppedConv2d(
            in_channels,
            2 * out_channels,
            (kernel_size // 2 + 1, kernel_size),
            padding=(kernel_size // 2 + 1, kernel_size // 2)
        )
        self.v_fc = nn.Conv2d(in_channels, 2 * out_channels, (1, 1))
        self.v_to_h = MaskedConv2d(
            2 * out_channels,
            2 * out_channels,
            (1, 1),
            mask_type='B',
            data_channels=data_channels
        )

        # Horizontal stack convolution
        self.h_conv = MaskedConv2d(
            in_channels,
            2 * out_channels,
            (1, kernel_size),
            mask_type='B',
            data_channels=data_channels,
            padding=(0, kernel_size // 2)
        )
        self.h_fc = MaskedConv2d(
            out_channels,
            out_channels,
            (1, 1),
            mask_type='B',
            data_channels=data_channels
        )

        # Skip connection
        self.h_skip = MaskedConv2d(
            out_channels,
            out_channels,
            (1, 1),
            mask_type='B',
            data_channels=data_channels
        )

    def forward(self, x):
        v_in, h_in, skip, label = x[0], x[1], x[2], x[3]

        # Vertical stack
        v_out, v_shifted = self.v_conv(v_in)
        v_out += self.v_fc(v_in)
        v_out_tanh, v_out_sigmoid = torch.split(v_out, self.split_size, dim=1)
        v_out = torch.tanh(v_out_tanh) * torch.sigmoid(v_out_sigmoid)

        # Horizontal stack
        h_out = self.h_conv(h_in)
        v_shifted = self.v_to_h(v_shifted)
        h_out += v_shifted
        h_out_tanh, h_out_sigmoid = torch.split(h_out, self.split_size, dim=1)
        h_out = torch.tanh(h_out_tanh) * torch.sigmoid(h_out_sigmoid)

        # Skip connection
        skip = skip + self.h_skip(h_out)

        h_out = self.h_fc(h_out)

        # Residual connections
        h_out = h_out + h_in
        v_out = v_out + v_in

        return {0: v_out, 1: h_out, 2: skip, 3: label}



class PixelCNN(nn.Module):
    def __init__(self, cfg, num_labels):
        super(PixelCNN, self).__init__()

        DATA_CHANNELS = 1  # Since we repeat the grayscale image to 3 channels

        self.hidden_fmaps = cfg["hidden_fmaps"]
        self.color_levels = cfg["color_levels"]

        self.causal_conv = CausalBlock(DATA_CHANNELS,
                                       cfg["hidden_fmaps"],
                                       cfg["causal_ksize"],
                                       data_channels=DATA_CHANNELS)

        self.hidden_conv = nn.ModuleList(
            [
                GatedBlock(
                    cfg["hidden_fmaps"],
                    cfg["hidden_fmaps"],
                    cfg["hidden_ksize"],
                    DATA_CHANNELS
                )
                for _ in range(cfg["hidden_layers"])
            ]
        )

        self.label_embedding = nn.Embedding(num_labels, cfg["hidden_fmaps"])

        self.out_hidden_conv = MaskedConv2d(cfg["hidden_fmaps"],
                                            cfg["out_hidden_fmaps"],
                                            (1, 1),
                                            mask_type='B',
                                            data_channels=DATA_CHANNELS)

        self.out_conv = MaskedConv2d(cfg["out_hidden_fmaps"],
                                     DATA_CHANNELS * cfg["color_levels"],
                                     (1, 1),
                                     mask_type='B',
                                     data_channels=DATA_CHANNELS)


    def forward(self, image, label):
        batch_size, data_channels, height, width = image.size()

        # Causal convolution
        v, h = self.causal_conv(image)

        # Initialize skip connections
        skip_connections = 0
        h_dict = {0: v, 1: h, 2: skip_connections, 3: label}

        # Pass through Gated Blocks
        for layer in self.hidden_conv:
            h_dict = layer(h_dict)
            skip_connections = skip_connections + h_dict[2]

        out = skip_connections

        # Add label embedding
        label_embedded = self.label_embedding(label).unsqueeze(2).unsqueeze(3)
        out = out + label_embedded

        # Output layers
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)  # Add dropout
        out = F.relu(self.out_hidden_conv(out))
        out = self.out_conv(out)

        out = out.view(batch_size, self.color_levels, data_channels, height, width)
        return out


    def sample(self, shape, count, label=None, device='cuda'):
        channels, height, width = shape

        samples = torch.zeros(count, *shape).to(device)
        if label is None:
            labels = torch.randint(high=10, size=(count,)).to(device)
        else:
            labels = (label*torch.ones(count)).to(device).long()

        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    for c in range(channels):
                        unnormalized_probs = self.forward(samples, labels)
                        pixel_probs = torch.softmax(unnormalized_probs[:, :, c, i, j], dim=1)
                        sampled_levels = torch.multinomial(pixel_probs, 1).squeeze().float() / (self.color_levels - 1)
                        samples[:, c, i, j] = sampled_levels

        return samples
