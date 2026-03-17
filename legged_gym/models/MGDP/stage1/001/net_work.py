import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------- Conv / Deconv blocks (InstanceNorm2d) --------------------------

def create_conv_block_in(in_channels, out_channels):
    """Double Conv + InstanceNorm2d + ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def create_deconv_block_in(in_channels, out_channels, is_last=False):
    """Deconv + InstanceNorm2d + ReLU + Upsample"""
    layers = [
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
    ]

    if not is_last:
        layers.extend([
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        ])
    else:
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    return nn.Sequential(*layers)


# -------------------------- 1. ImageEncoder --------------------------
class ImageEncoder(nn.Module):
    def __init__(self, input_channels=2, hidden_channels=[16, 32, 64], output_channels=128, pool=2):
        super().__init__()
        self.pool = pool
        self.hidden_channels = hidden_channels

        self.conv1 = create_conv_block_in(input_channels, hidden_channels[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = create_conv_block_in(hidden_channels[0], hidden_channels[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = create_conv_block_in(hidden_channels[1], hidden_channels[2])

        self.global_pool = nn.AdaptiveAvgPool2d((pool, pool))
        self.fc = nn.Linear(hidden_channels[2] * pool * pool, output_channels)
        self.output_activation = nn.Sigmoid()

        self.skip_connections = []

    def forward(self, x):
        self.skip_connections.clear();
        batch_size = x.size(0)

        x1 = self.conv1(x);
        self.skip_connections.append(x1);
        x = self.pool1(x1)
        x2 = self.conv2(x);
        self.skip_connections.append(x2);
        x = self.pool2(x2)
        x3 = self.conv3(x);
        self.skip_connections.append(x3)

        x = self.global_pool(x3)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


# -------------------------- 2. ImageDecoder --------------------------
class ImageDecoder(nn.Module):
    def __init__(self, input_dims=128, hidden_channels=[64, 32, 16], pool=2):
        super().__init__()
        self.pool = pool
        self.hidden_channels = hidden_channels
        self.output_channels = 2

        self.fc = nn.Linear(input_dims, hidden_channels[0] * pool * pool)

        self.deconv1 = create_deconv_block_in(
            in_channels=hidden_channels[0] + hidden_channels[0], out_channels=hidden_channels[1]
        )
        self.deconv2 = create_deconv_block_in(
            in_channels=hidden_channels[1] + hidden_channels[1], out_channels=hidden_channels[2]
        )
        self.deconv3 = create_deconv_block_in(
            in_channels=hidden_channels[2] + hidden_channels[2], out_channels=self.output_channels, is_last=True
        )
        self.final_activation = nn.Sigmoid()
        self._init_decoder_weights()

    def _init_decoder_weights(self):
        nn.init.normal_(self.fc.weight, std=0.01)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        for m in self.deconv3.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, encoder_skips, target_size):
        batch_size = x.size(0)
        x = self.fc(x).view(batch_size, self.hidden_channels[0], self.pool, self.pool)

        skip3 = encoder_skips[2];
        skip3_downsampled = nn.AdaptiveAvgPool2d(x.shape[2:])(skip3)
        x = torch.cat([x, skip3_downsampled], dim=1);
        x = self.deconv1(x)

        skip2 = encoder_skips[1];
        skip2_downsampled = nn.AdaptiveAvgPool2d(x.shape[2:])(skip2)
        x = torch.cat([x, skip2_downsampled], dim=1);
        x = self.deconv2(x)

        skip1 = encoder_skips[0];
        skip1_downsampled = nn.AdaptiveAvgPool2d(x.shape[2:])(skip1)
        x = torch.cat([x, skip1_downsampled], dim=1);
        x = self.deconv3(x)

        x = nn.AdaptiveAvgPool2d(target_size)(x)
        x = self.final_activation(x)
        return x


# -------------------------- 3. MapEncoder --------------------------
class MapEncoder(ImageEncoder):
    def __init__(self, input_channels=1, hidden_channels=[16, 32, 64], output_channels=128, pool=2):
        super().__init__(input_channels, hidden_channels, output_channels, pool)
        self.output_activation = nn.Tanh()


# -------------------------- 4. MapDecoder --------------------------
class MapDecoder(ImageDecoder):
    def __init__(self, input_dims=128, hidden_channels=[64, 32, 16], pool=2):
        super().__init__(input_dims, hidden_channels, pool)
        self.output_channels = 1
        self.final_activation = nn.Tanh()

        self.deconv3 = create_deconv_block_in(
            in_channels=hidden_channels[2] + hidden_channels[2],
            out_channels=self.output_channels,
            is_last=True
        )
        for m in self.deconv3.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, encoder_skips, target_size):
        batch_size = x.size(0)
        x = self.fc(x).view(batch_size, self.hidden_channels[0], self.pool, self.pool)

        skip3 = encoder_skips[2]
        skip3_downsampled = nn.AdaptiveAvgPool2d(x.shape[2:])(skip3)
        x = torch.cat([x, skip3_downsampled], dim=1)
        x = self.deconv1(x)

        skip2 = encoder_skips[1]
        skip2_downsampled = nn.AdaptiveAvgPool2d(x.shape[2:])(skip2)
        x = torch.cat([x, skip2_downsampled], dim=1)
        x = self.deconv2(x)

        skip1 = encoder_skips[0]
        skip1_downsampled = nn.AdaptiveAvgPool2d(x.shape[2:])(skip1)
        x = torch.cat([x, skip1_downsampled], dim=1)
        x = self.deconv3(x)

        x = nn.AdaptiveAvgPool2d(target_size)(x)
        return self.final_activation(x)


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='gru', num_layers=1, hidden_size=128):
        super().__init__()
        # RNN
        self.type = type
        self.rnn_cls = nn.GRU if self.type.lower() == 'gru' else nn.LSTM
        self.rnn = self.rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        input = input.unsqueeze(1)  # Shape: (batch_size,  128) to (batch_size, 1, 128)
        f1, self.hidden_states = self.rnn(input, self.hidden_states)
        out = f1.squeeze(1)  # Shape: (batch_size, 1, 128) to  (batch_size, 128)
        return out, self.hidden_states

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        if self.hidden_states is not None and dones is not None:
            # Find indices where dones is True
            index = torch.where(dones)[0]
            if self.type == "lstm":
                # Reset the specified slices
                self.hidden_states[0][..., index, :] = 0.0
                self.hidden_states[1][..., index, :] = 0.0
            else:
                # Reset the specified slices
                self.hidden_states[0][..., index, :] = 0.0

    def forward_onnx(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        # print('batch_mode',masks, batch_mode, hidden_states)

        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")

            input = input.unsqueeze(1)  # Shape: (batch_size, 1, 64)
            if self.type == "lstm":
                f1, _ = self.rnn(input, hidden_states)
            else:
                f1, _ = self.rnn(input, hidden_states[0])
            out = f1.squeeze(1)
            return out, None

        else:
            input = input.unsqueeze(1)  # Shape: (batch_size,  128) to (batch_size, 1, 128)
            f1, hidden_states = self.rnn(input, hidden_states)
            out = f1.squeeze(1)  # Shape: (batch_size, 1, 128) to  (batch_size, 128)
            self.hidden_states = hidden_states
            return out, self.hidden_states


