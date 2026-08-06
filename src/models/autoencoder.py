import torch
import torch.nn as nn

class WildfireAutoencoder(nn.Module):
    """
    Autoencoder aligned with the Badhan et al. paper design,
    adapted to this project: a single pre-fused input tensor
    (B, goes_channels + cov_channels, H, W). Resolution alignment and channel
    concatenation are already done upstream by WildfireDataset.
    """

    def __init__(self, in_channels: int, prior_fire_fraction: float = 0.0085):
        super().__init__()

        self.enc_conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU())
        self.enc_conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU())
        self.enc_conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU())

        self.dec_tconv1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.dec_block1 = nn.Sequential(
            nn.Conv2d(256 + 128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.dec_tconv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.dec_block2 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.dec_relu = nn.ReLU()

        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        #self.activation = nn.Sigmoid()
        
        with torch.no_grad():
            bias_val = torch.logit(torch.tensor(prior_fire_fraction))
            self.final_conv.bias.fill_(bias_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc_conv1(x)
        e2 = self.enc_conv2(e1)             # skip 1: 64ch @ full res
        p1 = self.pool1(e2)
        e3 = self.enc_conv3(p1)
        e4 = self.enc_conv4(e3)             # skip 2: 128ch @ /2
        p2 = self.pool2(e4)
        latent = self.enc_conv5(p2)

        u1 = self.dec_relu(self.dec_tconv1(latent))
        d1 = self.dec_block1(torch.cat([u1, e4], dim=1))
        u2 = self.dec_relu(self.dec_tconv2(d1))
        d2 = self.dec_block2(torch.cat([u2, e2], dim=1))

        out = self.final_conv(d2)
        return out #self.activation(out)


def make_model(goes_channels: int, cov_channels: int, stage1_channels: int = 0,
                prior_fire_fraction: float = 0.0085, **kwargs) -> WildfireAutoencoder:
    return WildfireAutoencoder(
        in_channels=goes_channels + cov_channels + stage1_channels,
        prior_fire_fraction=prior_fire_fraction,
    )