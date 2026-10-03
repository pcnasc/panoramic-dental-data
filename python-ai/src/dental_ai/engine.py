import copy
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def dice_score(preds_logits, targets, smooth=1):
    """Dice usando a máscara binarizada (threshold 0.5): quanto da área prevista
    e da área real realmente se sobrepõem."""
    preds = (torch.sigmoid(preds_logits) > 0.5).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return ((2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)).item()


@torch.no_grad()
def recall_score(preds_logits, targets, smooth=1):
    """Sensibilidade: de toda a lesão real, quanto o modelo conseguiu marcar.
    Clinicamente mais importante que Dice sozinho — um falso negativo aqui é
    uma cárie que passou despercebida."""
    preds = (torch.sigmoid(preds_logits) > 0.5).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    tp = (preds * targets).sum()
    fn = ((1 - preds) * targets).sum()
    return ((tp + smooth) / (tp + fn + smooth)).item()


def run_training(modelo, train_loader, val_loader, device, epochs, patience, criterio_perda, lr=1e-3, log_fn=print):
    """Loop de treino com scheduler de LR, early stopping e clipping de gradiente.

    Retorna o melhor state_dict visto (guardado em memória, nunca em disco,
    para não poluir o diretório durante a validação cruzada), junto com o
    Dice/Recall/época em que ele ocorreu.
    """
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    agendador_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="max", factor=0.5, patience=10)

    melhor_dice = 0.0
    melhor_recall = 0.0
    melhor_epoca = 0
    melhor_state = None
    epocas_sem_melhora = 0

    for epoca in range(epochs):
        modelo.train()
        perda_acumulada = 0.0
        for imagens, mascaras in train_loader:
            imagens, mascaras = imagens.to(device), mascaras.to(device)
            otimizador.zero_grad()
            palpites = modelo(imagens)
            erro = criterio_perda(palpites, mascaras)
            erro.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()
            perda_acumulada += erro.item()
        erro_medio = perda_acumulada / len(train_loader)

        modelo.eval()
        val_perda_acumulada = 0.0
        val_dice_acumulado = 0.0
        val_recall_acumulado = 0.0
        with torch.no_grad():
            for imagens, mascaras in val_loader:
                imagens, mascaras = imagens.to(device), mascaras.to(device)
                palpites = modelo(imagens)
                val_perda_acumulada += criterio_perda(palpites, mascaras).item()
                val_dice_acumulado += dice_score(palpites, mascaras)
                val_recall_acumulado += recall_score(palpites, mascaras)

        val_erro_medio = val_perda_acumulada / len(val_loader)
        val_dice_medio = val_dice_acumulado / len(val_loader)
        val_recall_medio = val_recall_acumulado / len(val_loader)
        agendador_lr.step(val_dice_medio)

        lr_atual = otimizador.param_groups[0]["lr"]
        log_fn(
            f">>> Época {epoca + 1:03d}/{epochs} | Treino: {erro_medio:.4f} | "
            f"Val: {val_erro_medio:.4f} | Dice: {val_dice_medio:.4f} | "
            f"Recall: {val_recall_medio:.4f} | LR: {lr_atual:.2e}"
        )

        if val_dice_medio > melhor_dice:
            melhor_dice = val_dice_medio
            melhor_recall = val_recall_medio
            melhor_epoca = epoca + 1
            melhor_state = copy.deepcopy(modelo.state_dict())
            epocas_sem_melhora = 0
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= patience:
                log_fn(f"Sem melhora no Dice por {patience} épocas. Parando mais cedo.")
                break

    return {
        "state_dict": melhor_state,
        "dice": melhor_dice,
        "recall": melhor_recall,
        "epoch": melhor_epoca,
    }
