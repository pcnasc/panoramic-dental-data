import torch
import torch.nn as nn


class BCEFocalTverskyLoss(nn.Module):
    """
    Combina BCE (estabiliza o início do treino) com Focal Tversky Loss.

    A Dice Loss trata falso positivo e falso negativo como igualmente ruins.
    Clinicamente não são: deixar passar uma cárie (falso negativo) é pior do
    que marcar uma área saudável por engano (falso positivo). A Tversky Loss
    pesa os dois separadamente via alpha (peso do FP) e beta (peso do FN) —
    aqui beta > alpha, então o modelo é penalizado mais por lesões que ele
    não marcou do que por marcações demais.

    O termo focal (expoente gamma) faz o gradiente se concentrar nos casos
    difíceis em vez dos já fáceis/óbvios — importante aqui porque lesões
    cobrem ~0.1% da imagem, então a grande maioria dos pixels é "fácil"
    (fundo/osso saudável) e não deveria dominar o aprendizado.
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        bce_loss = self.bce(inputs, targets)

        probs = torch.sigmoid(inputs).view(-1)
        targets_flat = targets.view(-1)

        tp = (probs * targets_flat).sum()
        fp = ((1 - targets_flat) * probs).sum()
        fn = (targets_flat * (1 - probs)).sum()

        tversky_index = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        focal_tversky_loss = (1 - tversky_index) ** self.gamma

        return bce_loss + focal_tversky_loss
