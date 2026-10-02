"""Script de inferência: Carrega o modelo salvo e tenta prever lesões em uma imagem."""
import sys
from pathlib import Path

# 1. Forçar a injeção da pasta 'src' no caminho do Python antes das importações
raiz_projeto = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(raiz_projeto / "src"))

import torch
import cv2
import matplotlib.pyplot as plt
from dental_ai.model import UNet

def main():
    caminho_modelo = raiz_projeto / "unet_pesos.pth"
    caminho_dados = raiz_projeto.parent / "data" / "raw" / "dental-panoramic-xrays"

    print("A carregar o cérebro da IA guardado no disco...")

    # Instanciar a rede e injetar a "memória" (os pesos matemáticos)
    modelo = UNet(in_channels=1, out_channels=1)
    modelo.load_state_dict(torch.load(str(caminho_modelo), weights_only=True))
    modelo.eval() # Tranca a rede no modo de inferência (não aprende, só prevê)

    img_path = caminho_dados / "images" / "360.png"
    mask_path = caminho_dados / "labels" / "360.png"

    if not img_path.exists():
        print(f"Erro: Não encontrei a imagem em {img_path}")
        return

    raio_x = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    gabarito = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Simular a linha de montagem (redimensionamento e normalização)
    rx_resized = cv2.resize(raio_x, (256, 256))
    gab_resized = cv2.resize(gabarito, (256, 256))

    rx_tensor = torch.from_numpy(rx_resized).float() / 255.0
    rx_tensor = rx_tensor.unsqueeze(0).unsqueeze(0) # Formato final: [1, 1, 256, 256]

    print("A processar a radiografia através da U-Net...")

    with torch.no_grad():
        palpite_cru = modelo(rx_tensor)
        # Transformar os números brutos numa percentagem de certeza usando Sigmoid
        probabilidade = torch.sigmoid(palpite_cru)
        palpite_numpy = probabilidade.squeeze().numpy()

    # Mostrar o resultado final no ecrã
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.imshow(rx_resized, cmap="gray")
    ax1.set_title("Raio-X Original (Input)")
    ax1.axis("off")

    ax2.imshow(gab_resized, cmap="gray")
    ax2.set_title("Gabarito Clínico (A Resposta Correta)")
    ax2.axis("off")

    ax3.imshow(palpite_numpy, cmap="magma") # Mapa de calor para destacar o diagnóstico
    ax3.set_title("Palpite da IA (Após 2 Épocas)")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()