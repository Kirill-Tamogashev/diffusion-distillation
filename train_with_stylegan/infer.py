from argparse import ArgumentParser
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from train_with_stylegan.utils import (
    configure_unet_model_from_pretrained,
    tensor_min,
    tensor_max
)


BASE_CHECKPOINT_PATH = Path("./wandb-checkpoints")
BASE_MODELS = ["google/ddpm-cifar10-32"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-n", "--n-samples", type=int, required=True)
    parser.add_argument("-d", "--device", type=int, default=0 if torch.cuda.is_available() else None)
    parser.add_argument("-r", "--resolution", type=int, default=32)
    parser.add_argument("-t", "--base-model", type=str, choices=BASE_MODELS, required=True)
    parser.add_argument("-c", "--checkpoint", type=Path, required=True)
    parser.add_argument("-o", "--image-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint-path", type=Path, default=BASE_CHECKPOINT_PATH)
    return parser.parse_args()


def load_checkpoint(args, device, name: str = "student"):
    ckpt_path = args.base_checkpoint_path / args.checkpoint
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    return ckpt_dict[name]


@torch.no_grad()
def infer(args):
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device is not None else "cpu"
    )
    model = configure_unet_model_from_pretrained(args.base_model)
    checkpoint = load_checkpoint(args, device)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    noise = torch.randn(args.n_samples, 3, args.resolution, args.resolution).to(device)
    t = torch.zeros(args.n_samples).to(device)

    images = model(noise, t).sample.cpu().permute(0, 2, 3, 1)

    min_ = tensor_min(images, dims=(1, 2, 3))
    max_ = tensor_max(images, dims=(1, 2, 3))

    normed_images = (images - min_) / (max_ - min_)
    numpy_images = normed_images.mul(255).numpy().astype(np.uint8)

    for idx, image in enumerate(numpy_images):
        Image.fromarray(image).save(args.image_dir / f"image-{idx}.jpg")


def main():
    arg_dict = parse_args()
    infer(arg_dict)


if __name__ == "__main__":
    main()
