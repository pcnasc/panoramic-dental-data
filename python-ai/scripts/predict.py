"""Script de inferência: Carrega o modelo salvo e tenta prever lesões em uma imagem."""
import argparse
import sys
from pathlib import Path

# 1. Forçar a injeção da pasta 'src' no caminho do Python antes das importações
raiz_projeto = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(raiz_projeto / "src"))

import torch
import cv2
import matplotlib.pyplot as plt
from dental_ai.model import UNet


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(sample_id: str) -> None:
    device = get_device()
    caminho_modelo = raiz_projeto / "checkpoints" / "unet_pesos_best.pth"
    caminho_dados = raiz_projeto.parent / "data" / "raw" / "dental-panoramic-xrays"

    if not caminho_modelo.exists():
        print(f"Erro: não encontrei pesos treinados em {caminho_modelo}. Rode train_ai.py primeiro.")
        return

    print("A carregar o cérebro da IA guardado no disco...")

    # Instanciar a rede e injetar a "memória" (os pesos matemáticos)
    modelo = UNet(in_channels=1, out_channels=1).to(device)
    modelo.load_state_dict(torch.load(str(caminho_modelo), map_location=device, weights_only=True))
    modelo.eval()  # Tranca a rede no modo de inferência (não aprende, só prevê)

    img_path = caminho_dados / "images" / f"{sample_id}.png"
    mask_path = caminho_dados / "labels" / f"{sample_id}.png"

    if not img_path.exists():
        print(f"Erro: Não encontrei a imagem em {img_path}")
        return

    raio_x = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    gabarito = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Simular a linha de montagem (redimensionamento e normalização)
    rx_resized = cv2.resize(raio_x, (256, 256))
    gab_resized = cv2.resize(gabarito, (256, 256), interpolation=cv2.INTER_NEAREST)

    rx_tensor = torch.from_numpy(rx_resized).float() / 255.0
    rx_tensor = rx_tensor.unsqueeze(0).unsqueeze(0).to(device)  # Formato final: [1, 1, 256, 256]

    print("A processar a radiografia através da U-Net...")

    with torch.no_grad():
        palpite_cru = modelo(rx_tensor)
        # Transformar os números brutos numa percentagem de certeza usando Sigmoid
        probabilidade = torch.sigmoid(palpite_cru)
        palpite_numpy = probabilidade.squeeze().cpu().numpy()

    # Mostrar o resultado final no ecrã
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.imshow(rx_resized, cmap="gray")
    ax1.set_title("Raio-X Original (Input)")
    ax1.axis("off")

    ax2.imshow(gab_resized, cmap="gray")
    ax2.set_title("Gabarito Clínico (A Resposta Correta)")
    ax2.axis("off")

    ax3.imshow(palpite_numpy, cmap="magma")  # Mapa de calor para destacar o diagnóstico
    ax3.set_title("Palpite da IA")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_id", nargs="?", default="360", help="Image/label filename stem, e.g. 360")
    args = parser.parse_args()
    main(args.sample_id)