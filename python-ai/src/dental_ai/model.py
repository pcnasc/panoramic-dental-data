import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Bloco de construção base: duas convoluções seguidas de normalização e ativação."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    1. A Descida (Encoder): esmaga a radiografia sucessivas vezes, diminuindo o
       tamanho da imagem e aumentando a profundidade dos filtros, para entender
       *o que* compõe a imagem (texturas de osso, contrastes de lesão, raízes).
    2. O Gargalo (Bottleneck): ponto de maior compressão, a representação mais
       densa e abstrata do raio-x.
    3. A Subida (Decoder): expande de volta ao tamanho original usando
       conexões de atalho (skip connections) da descida para lembrar a
       geografia dos dentes e pintar pixel por pixel onde está a anomalia.
    """

    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1. A Descida (Encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 2. O Gargalo (Bottleneck)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 3. A Subida (Decoder)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Camada final para converter os filtros de volta em 1 canal (máscara)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Garante que os tamanhos batam caso o arredondamento do pooling
            # tenha cortado 1px (ex.: entradas com dimensão ímpar).
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)
