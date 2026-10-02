"""Script de treino: Dice+BCE Loss, split treino/validação, augmentação e
checkpoint do melhor modelo (por Dice de validação)."""
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dental_ai.dataset import DentalXRayDataset
from dental_ai.model import UNet

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --- NOVA MATEMÁTICA: BCE + Dice Loss ---
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        # Calcula o erro tradicional
        bce_loss = self.bce(inputs, targets)

        # Converte os palpites para probabilidades
        inputs = torch.sigmoid(inputs)

        # Achata as matrizes para comparar a sobreposição
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calcula o Coeficiente Dice (a intersecção)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth))

        # Combina as duas funções (peso maior para a sobreposição)
        return bce_loss + dice_loss


@torch.no_grad()
def dice_score(preds_logits, targets, smooth=1):
    """Dice coeficiente 'de verdade': usa a máscara binarizada (threshold 0.5),
    não a versão suavizada usada na loss. É o número que importa clinicamente:
    quanto da lesão real o modelo efetivamente acertou."""
    preds = (torch.sigmoid(preds_logits) > 0.5).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return ((2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)).item()


def split_filenames(images_dir: Path, val_fraction=0.2, seed=SEED):
    filenames = sorted(f.name for f in images_dir.iterdir() if f.suffix == ".png")
    rng = random.Random(seed)
    rng.shuffle(filenames)
    n_val = max(1, int(len(filenames) * val_fraction))
    return filenames[n_val:], filenames[:n_val]


def main():
    set_seed()
    device = get_device()
    print(f"Usando dispositivo: {device}")

    raiz_projeto = Path(__file__).resolve().parents[1]
    caminho_dados = raiz_projeto.parent / "data" / "raw" / "dental-panoramic-xrays"
    images_dir = caminho_dados / "images"
    masks_dir = caminho_dados / "labels"

    print("1. Preparando a esteira de dados (split treino/validação)...")
    train_files, val_files = split_filenames(images_dir)
    print(f"   -> {len(train_files)} imagens de treino, {len(val_files)} de validação")

    # Augmentação só no treino: a validação precisa refletir dados reais, sem distorção.
    train_dataset = DentalXRayDataset(images_dir, masks_dir, filenames=train_files, augment=True)
    val_dataset = DentalXRayDataset(images_dir, masks_dir, filenames=val_files, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print("2. Instanciando a U-Net...")
    modelo = UNet(in_channels=1, out_channels=1).to(device)

    print("3. Configurando a matemática avançada (Dice Loss)...")
    criterio_perda = DiceBCELoss()
    otimizador = optim.Adam(modelo.parameters(), lr=0.001)
    # Reduz o LR quando o Dice de validação estagna: ajuda a estabilizar o fim
    # do treino (evita os picos de val loss causados por passos grandes demais).
    agendador_lr = optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="max", factor=0.5, patience=10
    )

    epocas = 150
    paciencia_early_stop = 25
    epocas_sem_melhora = 0
    melhor_dice = 0.0
    checkpoints_dir = raiz_projeto / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    caminho_melhor_modelo = checkpoints_dir / "unet_pesos_best.pth"

    print("\n--- INICIANDO TREINAMENTO PROFUNDO ---")

    for epoca in range(epocas):
        modelo.train()
        perda_acumulada = 0.0

        for imagens, mascaras in train_loader:
            imagens, mascaras = imagens.to(device), mascaras.to(device)
            otimizador.zero_grad()
            palpites = modelo(imagens)
            erro = criterio_perda(palpites, mascaras)
            erro.backward()
            # Evita passos explosivos quando um lote raro tem gradiente grande
            # (comum com máscaras tão desbalanceadas quanto estas).
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()
            perda_acumulada += erro.item()

        erro_medio = perda_acumulada / len(train_loader)

        # --- Validação: mede generalização em imagens nunca vistas no treino ---
        modelo.eval()
        val_perda_acumulada = 0.0
        val_dice_acumulado = 0.0
        with torch.no_grad():
            for imagens, mascaras in val_loader:
                imagens, mascaras = imagens.to(device), mascaras.to(device)
                palpites = modelo(imagens)
                val_perda_acumulada += criterio_perda(palpites, mascaras).item()
                val_dice_acumulado += dice_score(palpites, mascaras)

        val_erro_medio = val_perda_acumulada / len(val_loader)
        val_dice_medio = val_dice_acumulado / len(val_loader)
        agendador_lr.step(val_dice_medio)

        lr_atual = otimizador.param_groups[0]["lr"]
        print(
            f">>> Época {epoca + 1:03d}/{epocas} | Treino: {erro_medio:.4f} | "
            f"Val: {val_erro_medio:.4f} | Val Dice: {val_dice_medio:.4f} | LR: {lr_atual:.2e}"
        )

        if val_dice_medio > melhor_dice:
            melhor_dice = val_dice_medio
            epocas_sem_melhora = 0
            torch.save(modelo.state_dict(), str(caminho_melhor_modelo))
            print(f"    -> novo melhor modelo salvo (Dice={melhor_dice:.4f})")
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= paciencia_early_stop:
                print(f"\nSem melhora no Dice por {paciencia_early_stop} épocas. Parando mais cedo.")
                break

    print(f"\nTreino concluído. Melhor Dice de validação: {melhor_dice:.4f}")
    print(f"Pesos salvos em: {caminho_melhor_modelo}")


if __name__ == "__main__":
    main()
