import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Downsampling path
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Upsampling path
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(128, 64)

        # Final output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t=None):
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))

        # Bottleneck
        b = self.bottleneck(self.pool(d2))

        # Upsampling
        u2 = self.up_conv2(torch.cat((self.up2(b), d2), dim=1))
        u1 = self.up_conv1(torch.cat((self.up1(u2), d1), dim=1))

        return self.out(u1)


if __name__ == "__main__":
    # Test the model
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn(8, 1, 28, 28)  # Batch size of 8, MNIST image size
    out = model(x)
    print(out.shape)  # Should be (8, 1, 28, 28)

