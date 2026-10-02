import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DentalXRayDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, img_size=(256, 256), filenames=None, augment=False):
        """
        Inicializa a linha de montagem das imagens.
        img_size: Redimensiona todas as imagens para 256x256 pixels.
        filenames: lista explícita de arquivos a usar (para separar treino/validação
                   sem duplicar a lógica de split). Se None, usa todos os .png da pasta.
        augment: aplica flips/rotação/brilho aleatórios (somente para treino).
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.augment = augment

        if filenames is not None:
            self.images = list(filenames)
        else:
            self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    def __len__(self):
        # Diz ao PyTorch quantas imagens temos no total
        return len(self.images)

    def _augment(self, image, mask):
        """Aumentação leve: flip horizontal, pequena rotação e jitter de brilho/contraste.
        A máscara sofre exatamente a mesma transformação geométrica que a imagem
        (senão o rótulo deixa de corresponder à anatomia), mas o jitter de
        brilho/contraste é aplicado só na imagem.
        """
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            h, w = image.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if random.random() < 0.5:
            brightness = random.uniform(-20, 20)
            contrast = random.uniform(0.85, 1.15)
            image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

        return image, mask

    def __getitem__(self, idx):
        # 1. Localiza a imagem e a máscara correspondente
        img_name = self.images[idx]
        img_path = self.images_dir / img_name
        mask_path = self.masks_dir / img_name

        # 2. Carrega as imagens em tons de cinza
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # 3. Redimensiona para o tamanho padrão (256x256)
        # A máscara usa INTER_NEAREST: as lesões já ocupam ~0.1% dos pixels,
        # então a interpolação bilinear (padrão) borraria/apagaria os rótulos
        # em vez de manter 0/255 exatos.
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        # 4. Converte para Tensor e Normaliza (divide por 255 para ficar entre 0 e 1)
        image_tensor = torch.from_numpy(image).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        # Garante rótulo binário exato mesmo após augmentations geométricas
        mask_tensor = (mask_tensor > 0.5).float()

        # 5. Adiciona a dimensão do canal (PyTorch exige o formato: Canal x Altura x Largura)
        image_tensor = image_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor
