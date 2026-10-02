import os
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset


class DentalXRayDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, img_size=(256, 256)):
        """
        Inicializa a linha de montagem das imagens.
        img_size: Redimensiona todas as imagens para 256x256 pixels.
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size

        # Cria uma lista com o nome de todos os arquivos .png
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    def __len__(self):
        # Diz ao PyTorch quantas imagens temos no total
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Localiza a imagem e a máscara correspondente
        img_name = self.images[idx]
        img_path = self.images_dir / img_name
        mask_path = self.masks_dir / img_name

        # 2. Carrega as imagens em tons de cinza
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # 3. Redimensiona para o tamanho padrão (256x256)
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size)

        # 4. Converte para Tensor e Normaliza (divide por 255 para ficar entre 0 e 1)
        image_tensor = torch.from_numpy(image).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float() / 255.0

        # 5. Adiciona a dimensão do canal (PyTorch exige o formato: Canal x Altura x Largura)
        image_tensor = image_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor