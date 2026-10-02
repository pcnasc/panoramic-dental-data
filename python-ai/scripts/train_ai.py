"""Script de wiring: conecta o Dataset à U-Net, inicia o treino e guarda os pesos."""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Conecta aos nossos módulos
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from dental_ai.dataset import DentalXRayDataset
from dental_ai.model import UNet

# 1. Configuração de Caminhos
DATASET_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "dental-panoramic-xrays"
images_dir = DATASET_ROOT / "images"
masks_dir = DATASET_ROOT / "labels"

def main():
    print("1. Preparando a esteira de dados...")
    dataset = DentalXRayDataset(images_dir, masks_dir)

    if len(dataset) == 0:
        print(f"ERRO CRÍTICO: Nenhuma imagem encontrada em {images_dir}")
        return

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("2. Instanciando o modelo U-Net...")
    modelo = UNet(in_channels=1, out_channels=1)

    print("3. Configurando a matemática de aprendizado...")
    criterio_perda = nn.BCEWithLogitsLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=0.001)

    epocas = 2

    print("\n--- INICIANDO TREINAMENTO ---")
    modelo.train()

    for epoca in range(epocas):
        perda_acumulada = 0.0

        for lote_idx, (imagens, mascaras) in enumerate(dataloader):
            otimizador.zero_grad()
            palpites = modelo(imagens)
            erro = criterio_perda(palpites, mascaras)
            erro.backward()
            otimizador.step()
            perda_acumulada += erro.item()

            print(f"Época {epoca+1}/{epocas} | Lote {lote_idx+1}/{len(dataloader)} | Erro (Loss): {erro.item():.4f}")

        erro_medio = perda_acumulada / len(dataloader)
        print(f">>> Fim da Época {epoca+1} | Erro Médio: {erro_medio:.4f}\n")

    # Guardar a "memória" da IA no disco
    caminho_modelo = Path(__file__).resolve().parents[1] / "unet_pesos.pth"
    torch.save(modelo.state_dict(), str(caminho_modelo))
    print(f"Cérebro da IA guardado com sucesso em: {caminho_modelo}")

if __name__ == "__main__":
    main()