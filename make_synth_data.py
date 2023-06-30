import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--source",
        type=Path,
        required=True
    )
    parser.add_argument(
        "-t", "--target",
        type=Path,
        required=True
    )
    args = parser.parse_args()
    args.target.mkdir(parents=True)
    image_number = 0
    for batch_path in tqdm(args.source.iterdir()):
        arr = np.load(batch_path)["arr_0"]
        for image in arr:
            save_path = args.target / f"synthetic_image_{image_number}.jpg"
            Image.fromarray(image).save(save_path)
            image_number += 1
            
if __name__ == "__main__":
    main()