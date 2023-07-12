import typing as tp
from argparse import ArgumentParser
from pathlib import Path

import yaml
from ml_collections import ConfigDict

import torch

from train_with_stylegan.trainer import DIffGANTrainer
from train_with_stylegan.utils import configure_unet_model_from_pretrained

TEACHER_MODEL_OPTIONS = ["google/ddpm-cifar10-32"]
DEFAULT_CKPT_DIR = Path("wandb-checkpoints")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-t", "--teacher", type=str, choices=TEACHER_MODEL_OPTIONS)
    parser.add_argument("-p", "--params", type=Path, default="./train_with_stylegan/params.yaml")
    parser.add_argument("-d", "--device", type=str, default=0)
    parser.add_argument("--project", type=str, default="diff-to-gan")
    parser.add_argument("--dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--save-code", action="store_true")
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def load_params(args):
    with args.params.open("r") as f:
        config = yaml.safe_load(f)
    return ConfigDict(config)


def  configure_checkpoint_path(args) -> tp.Tuple[Path, Path]:
    log_dir = args.dir / args.name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    existing_ckpts = log_dir.glob("*.pt")
    if next(existing_ckpts, None) is None:
        return log_dir, None
    last_ckpt = max(existing_ckpts, key=lambda x: x.stat().st_ctime)
    return log_dir, last_ckpt


def train(args):

    params = load_params(args)
    
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available else torch.device("cpu")
    teacher = configure_unet_model_from_pretrained(args.teacher)
    student = configure_unet_model_from_pretrained(args.teacher)
    
    teacher.to(device)
    student.to(device)
    
    teacher.eval()
    student.train()

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


if __name__=="__main__":
    args = parse_args()
    train(args)
