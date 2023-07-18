from argparse import ArgumentParser
from pathlib import Path

import torch

from train_with_stylegan.trainer import DIffGANTrainer
from train_with_stylegan.utils import (
    configure_unet_model_from_pretrained,
    load_params,
    configure_checkpoint_path,
)


TEACHER_MODEL_OPTIONS = ["google/ddpm-cifar10-32"]
DEFAULT_CKPT_DIR = Path("wandb-checkpoints")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-t", "--teacher", type=str, choices=TEACHER_MODEL_OPTIONS)
    parser.add_argument("-p", "--params", type=Path, default="./train_with_stylegan/params/boot.yaml")
    parser.add_argument("-d", "--device", type=str, default=0)
    parser.add_argument("--project", type=str, default="diff-to-gan")
    parser.add_argument("--dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--save-code", action="store_true")
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def train(args):
    params = load_params(args)

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available else torch.device("cpu")
    teacher = configure_unet_model_from_pretrained(args.teacher, device, train=True)
    student = configure_unet_model_from_pretrained(args.teacher, device, train=False)
    log_dir, last_ckpt = configure_checkpoint_path(args)

    if last_ckpt is not None:
        raise AssertionError(
            "You may start training from existing checkpoint.\n"
            f"Existing checkpoint: {last_ckpt} \n"
            "But there is not code for that !!!"
        )
    DIffGANTrainer(
        teacher=teacher,
        student=student,
        params=params,
        device=device,
        log_dir=log_dir,
    ).run_training(args)


if __name__ == "__main__":
    args_dict = parse_args()
    train(args_dict)
