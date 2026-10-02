from pathlib import Path
import sys
from torch.utils.data import DataLoader


# Adiciona a pasta src ao caminho do Python para podermos importar o nosso módulo
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from dental_ai.dataset import DentalXRayDataset

# Caminhos para os dados brutos
DATASET_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "dental-panoramic-xrays"
images_dir = DATASET_ROOT / "images"
masks_dir = DATASET_ROOT / "labels"

# Inicializa a nossa classe
meu_dataset = DentalXRayDataset(images_dir, masks_dir)

print(f"Total de pares (raio-x e máscara) encontrados: {len(meu_dataset)}")

# Puxa o primeiro item processado (índice 0)
imagem, mascara = meu_dataset[0]

print(f"Formato final do Raio-X (Tensor PyTorch): {imagem.shape}")
print(f"Valores do Raio-X normalizados: Max={imagem.max():.2f}, Min={imagem.min():.2f}")

# Cria a esteira industrial, agrupando 16 imagens por lote e embaralhando a ordem
esteira_treino = DataLoader(meu_dataset, batch_size=16, shuffle=True)

# Puxa o primeiro lote da esteira
lote_imagens, lote_mascaras = next(iter(esteira_treino))
print(f"Formato do Lote (Batch): {lote_imagens.shape}")