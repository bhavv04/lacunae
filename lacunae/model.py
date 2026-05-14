import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder
        ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(ch, f))
            ch = f

        # bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # decoder
        for f in reversed(features):
            self.decoders.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # encode
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # decode
        for i in range(0, len(self.decoders), 2):
            x = self.decoders[i](x)
            skip = skip_connections[i // 2]

            # handle odd dimensions
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.decoders[i + 1](x)

        return torch.sigmoid(self.final(x))


if __name__ == "__main__":
    model = UNet()
    x = torch.randn(1, 1, 320, 320)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")