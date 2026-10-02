"""Quick visual sanity check: original radiograph vs. its segmentation label with overlay."""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

DATASET_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "dental-panoramic-xrays"


def main(sample_id: str) -> None:
    image_path = DATASET_ROOT / "images" / f"{sample_id}.png"
    label_path = DATASET_ROOT / "labels" / f"{sample_id}.png"

    raio_x = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mascara = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

    if raio_x is None or mascara is None:
        raise FileNotFoundError(f"Sample '{sample_id}' not found under {DATASET_ROOT}")

    # 1. Cria a sobreposição: converte o raio-x (1 canal) para RGB (3 canais) para suportar cores
    sobreposicao = cv2.cvtColor(raio_x, cv2.COLOR_GRAY2RGB)

    # 2. Pinta de vermelho ([255, 0, 0]) todos os pixels onde a máscara tem marcação (> 0)
    sobreposicao[mascara > 0] = [255, 0, 0]

    # 3. Altera a grade para 3 colunas e ajusta o tamanho (18, 6)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.imshow(raio_x, cmap="gray")
    ax1.axis("off")

    ax2.imshow(mascara, cmap="gray")
    ax2.axis("off")

    ax3.imshow(sobreposicao)
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_id", nargs="?", default="360", help="Image/label filename stem, e.g. 360")
    args = parser.parse_args()
    main(args.sample_id)