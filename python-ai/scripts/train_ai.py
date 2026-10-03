"""Treino com validação cruzada em k folds (estimativa confiável de
desempenho, já que 100 imagens são poucas para um único split fixo),
seguido do treino do modelo final em 100% dos dados para uso em predict.py."""
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dental_ai.dataset import DentalXRayDataset
from dental_ai.model import UNet
from dental_ai.losses import BCEFocalTverskyLoss
from dental_ai.engine import set_seed, get_device, run_training

SEED = 42
K_FOLDS = 5
EPOCHS = 150
PATIENCE = 25
BATCH_SIZE = 4


def build_folds(images_dir: Path, k=K_FOLDS, seed=SEED):
    filenames = sorted(f.name for f in images_dir.iterdir() if f.suffix == ".png")
    rng = random.Random(seed)
    rng.shuffle(filenames)
    return [filenames[i::k] for i in range(k)]


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Usando dispositivo: {device}")

    raiz_projeto = Path(__file__).resolve().parents[1]
    caminho_dados = raiz_projeto.parent / "data" / "raw" / "dental-panoramic-xrays"
    images_dir = caminho_dados / "images"
    masks_dir = caminho_dados / "labels"

    checkpoints_dir = raiz_projeto / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    folds = build_folds(images_dir)
    total_imagens = sum(len(f) for f in folds)
    print(f"1. Validação cruzada com {K_FOLDS} folds ({total_imagens} imagens no total)")

    resultados = []
    for fold_idx in range(K_FOLDS):
        val_files = folds[fold_idx]
        train_files = [f for i, fold in enumerate(folds) if i != fold_idx for f in fold]

        print(f"\n=== FOLD {fold_idx + 1}/{K_FOLDS} | treino={len(train_files)} val={len(val_files)} ===")

        # Augmentação só no treino de cada fold; validação sempre com dados reais.
        train_dataset = DentalXRayDataset(images_dir, masks_dir, filenames=train_files, augment=True)
        val_dataset = DentalXRayDataset(images_dir, masks_dir, filenames=val_files, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        modelo = UNet(in_channels=1, out_channels=1).to(device)
        criterio_perda = BCEFocalTverskyLoss()

        resultado = run_training(modelo, train_loader, val_loader, device, EPOCHS, PATIENCE, criterio_perda)
        print(
            f"--- Fold {fold_idx + 1} melhor: Dice={resultado['dice']:.4f} "
            f"Recall={resultado['recall']:.4f} (época {resultado['epoch']})"
        )
        resultados.append(resultado)

    dices = np.array([r["dice"] for r in resultados])
    recalls = np.array([r["recall"] for r in resultados])
    epocas_melhores = [r["epoch"] for r in resultados]

    print("\n=== RESULTADO DA VALIDAÇÃO CRUZADA (estimativa confiável de desempenho) ===")
    print(f"Dice:   {dices.mean():.4f} +/- {dices.std():.4f}  (folds: {[round(d, 4) for d in dices]})")
    print(f"Recall: {recalls.mean():.4f} +/- {recalls.std():.4f}  (folds: {[round(r, 4) for r in recalls]})")

    # --- Modelo final: usa 100% dos dados (sem held-out) para a versão real ---
    # Sem validação aqui, então usamos a média das "melhores épocas" dos folds
    # como referência de por quanto tempo treinar antes de começar a overfitar.
    epocas_finais = max(1, round(sum(epocas_melhores) / len(epocas_melhores)))
    print(f"\n2. Treinando modelo final em 100% dos dados por {epocas_finais} épocas...")

    todas_imagens = sorted(f.name for f in images_dir.iterdir() if f.suffix == ".png")
    dataset_final = DentalXRayDataset(images_dir, masks_dir, filenames=todas_imagens, augment=True)
    loader_final = DataLoader(dataset_final, batch_size=BATCH_SIZE, shuffle=True)

    modelo_final = UNet(in_channels=1, out_channels=1).to(device)
    criterio_perda = BCEFocalTverskyLoss()
    otimizador = torch.optim.Adam(modelo_final.parameters(), lr=1e-3)

    modelo_final.train()
    for epoca in range(epocas_finais):
        perda_acumulada = 0.0
        for imagens, mascaras in loader_final:
            imagens, mascaras = imagens.to(device), mascaras.to(device)
            otimizador.zero_grad()
            palpites = modelo_final(imagens)
            erro = criterio_perda(palpites, mascaras)
            erro.backward()
            torch.nn.utils.clip_grad_norm_(modelo_final.parameters(), max_norm=1.0)
            otimizador.step()
            perda_acumulada += erro.item()
        print(f">>> [Final] Época {epoca + 1:03d}/{epocas_finais} | Treino: {perda_acumulada / len(loader_final):.4f}")

    caminho_modelo_final = checkpoints_dir / "unet_pesos_best.pth"
    torch.save(modelo_final.state_dict(), str(caminho_modelo_final))

    print(f"\nModelo final salvo em: {caminho_modelo_final}")
    print(
        "Desempenho esperado em dados novos (via validação cruzada): "
        f"Dice {dices.mean():.4f} +/- {dices.std():.4f}, "
        f"Recall {recalls.mean():.4f} +/- {recalls.std():.4f}"
    )


if __name__ == "__main__":
    main()
