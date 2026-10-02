import sys
from pathlib import Path
import torch

# Conecta ao nosso módulo
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from dental_ai.model import UNet

# 1. Instancia o modelo
modelo = UNet(in_channels=1, out_channels=1)

# 2. Simula o lote que saiu do seu DataLoader (16 imagens, 1 canal, 256x256)
lote_falso = torch.randn((16, 1, 256, 256))

print(f"Formato da imagem de entrada: {lote_falso.shape}")

# 3. Passa pela rede neural
resultado = modelo(lote_falso)

print(f"Formato da máscara prevista: {resultado.shape}")