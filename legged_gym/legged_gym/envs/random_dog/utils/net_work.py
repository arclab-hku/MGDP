import torch
import torch.nn as nn
import torch.nn.functional as F


def create_norm2d(num_channels):
    """Use GroupNorm to preserve absolute depth statistics better than InstanceNorm."""
    num_groups = min(8, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            create_norm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            create_norm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpFuseBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.fuse = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip, target_size):
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        if skip.shape[-2:] != target_size:
            skip = F.interpolate(skip, size=target_size, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class ImageEncoder(nn.Module):
    def __init__(self, input_channels=2, hidden_channels=None, output_channels=128, pool=4):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 96]

        self.pool = pool
        self.hidden_channels = hidden_channels

        self.conv1 = DoubleConv(input_channels, hidden_channels[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = DoubleConv(hidden_channels[0], hidden_channels[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = DoubleConv(hidden_channels[1], hidden_channels[2])
        self.bottleneck = DoubleConv(hidden_channels[2], hidden_channels[2])

        self.global_pool = nn.AdaptiveAvgPool2d((pool, pool))
        proj_hidden = hidden_channels[2] * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[2] * pool * pool, proj_hidden),
            nn.LayerNorm(proj_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(proj_hidden, output_channels),
        )

        self.skip_connections = []

    def forward(self, x):
        self.skip_connections.clear()
        batch_size = x.size(0)

        x1 = self.conv1(x)
        self.skip_connections.append(x1)

        x2 = self.conv2(self.pool1(x1))
        self.skip_connections.append(x2)

        x3 = self.conv3(self.pool2(x2))
        x3 = self.bottleneck(x3)
        self.skip_connections.append(x3)

        pooled = self.global_pool(x3).view(batch_size, -1)
        return self.fc(pooled)


class ImageDecoder(nn.Module):
    def __init__(self, input_dims=128, hidden_channels=None, pool=4):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [96, 64, 32]

        self.pool = pool
        self.hidden_channels = hidden_channels
        self.output_channels = 2

        proj_hidden = hidden_channels[0] * pool * pool
        self.fc = nn.Sequential(
            nn.Linear(input_dims, proj_hidden),
            nn.LayerNorm(proj_hidden),
            nn.SiLU(inplace=True),
        )

        self.input_refine = DoubleConv(hidden_channels[0], hidden_channels[0])
        self.up1 = UpFuseBlock(hidden_channels[0], hidden_channels[0], hidden_channels[0])
        self.up2 = UpFuseBlock(hidden_channels[0], hidden_channels[1], hidden_channels[1])
        self.up3 = UpFuseBlock(hidden_channels[1], hidden_channels[2], hidden_channels[2])
        self.refine = DoubleConv(hidden_channels[2], hidden_channels[2])
        self.out_conv = nn.Conv2d(hidden_channels[2], self.output_channels, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x, encoder_skips, target_size):
        batch_size = x.size(0)
        x = self.fc(x).view(batch_size, self.hidden_channels[0], self.pool, self.pool)
        x = self.input_refine(x)

        skip3 = encoder_skips[2]
        x = self.up1(x, skip3, skip3.shape[-2:])

        skip2 = encoder_skips[1]
        x = self.up2(x, skip2, skip2.shape[-2:])

        skip1 = encoder_skips[0]
        x = self.up3(x, skip1, skip1.shape[-2:])

        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        x = self.refine(x)
        x = self.out_conv(x)
        return self.final_activation(x)


class MapEncoder(ImageEncoder):
    def __init__(self, input_channels=1, hidden_channels=None, output_channels=128, pool=2):
        if hidden_channels is None:
            hidden_channels = [16, 32, 64]
        super().__init__(input_channels, hidden_channels, output_channels, pool)


class MapDecoder(ImageDecoder):
    def __init__(self, input_dims=128, hidden_channels=None, pool=2):
        if hidden_channels is None:
            hidden_channels = [64, 32, 16]
        super().__init__(input_dims, hidden_channels, pool)
        self.output_channels = 1
        self.out_conv = nn.Conv2d(hidden_channels[2], self.output_channels, kernel_size=1)
        self.final_activation = nn.Tanh()


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='gru', num_layers=1, hidden_size=128):
        super().__init__()
        self.type = type
        self.rnn_cls = nn.GRU if self.type.lower() == 'gru' else nn.LSTM
        self.rnn = self.rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        input = input.unsqueeze(1)
        f1, self.hidden_states = self.rnn(input, self.hidden_states)
        out = f1.squeeze(1)
        return out, self.hidden_states

    def reset(self, dones=None):
        if self.hidden_states is not None and dones is not None:
            index = torch.where(dones)[0]
            if self.type == "lstm":
                self.hidden_states[0][..., index, :] = 0.0
                self.hidden_states[1][..., index, :] = 0.0
            else:
                self.hidden_states[0][..., index, :] = 0.0

    def forward_onnx(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None

        if batch_mode:
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")

            input = input.unsqueeze(1)
            if self.type == "lstm":
                f1, _ = self.rnn(input, hidden_states)
            else:
                f1, _ = self.rnn(input, hidden_states[0])
            out = f1.squeeze(1)
            return out, None

        input = input.unsqueeze(1)
        f1, hidden_states = self.rnn(input, hidden_states)
        out = f1.squeeze(1)
        self.hidden_states = hidden_states
        return out, self.hidden_states
