import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [B, L, D] -> [L, B, D]
        x = x.transpose(0, 1)

        # Self Attention
        attended = self.attention(x, x, x)[0]
        x = self.norm1(x + self.dropout(attended))
        # x = self.norm1(attended)

        # Feed Forward
        fed_forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(fed_forward))
        # x = self.norm2(fed_forward)

        #[L, B, D] -> [B, L, D]
        return x.transpose(0, 1)

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=2, dim=32, heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class MLPModule(nn.Module):
    def __init__(self, num_encoder_obs, encoder_hidden_dims):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_encoder_obs, encoder_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[0], encoder_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[1], encoder_hidden_dims[2]),
        )

    def forward(self, dm):
        return self.encoder(dm)

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, output_dim=128, pool=1):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if x.dim() == 3:  
            # x = x[:, 0, :]  
            x = torch.mean(x, dim=1)  

        return self.projection(x)  

class VAEModule(nn.Module):
    def __init__(self, input_dim=128, latent_dim=16):
        super().__init__()
        self.encoder_mean = nn.Linear(input_dim, latent_dim)
        self.encoder_logvar = nn.Linear(input_dim, latent_dim)

    def encode(self, x):
        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar

class NEWCNNHieghtEncoder(nn.Module):
    def __init__(self, num_classes):  
        super(NEWCNNHieghtEncoder, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            # nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  
        )

        self.fc = nn.Linear(32, num_classes)  

    def forward(self, x):
        f1 = self.encoder1(x)  
        f2 = self.encoder2(f1)  
        f3 = self.encoder3(f2) 
        f4 = f3.view(f3.size(0), -1)  
        x = self.fc(f4) 
        return x

class HeightEncoder(nn.Module):
    def __init__(self, num_encoder_obs, encoder_hidden_dims):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_encoder_obs, encoder_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[0], encoder_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[1], encoder_hidden_dims[2]),
        )

    def forward(self, dm):
        """
        Encodes depth map
        Input:
            dm: a depth map usually shape (187)
        """
        return self.encoder(dm)
