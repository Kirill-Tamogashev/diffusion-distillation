import typing as tp
from argparse import ArgumentParser
from pathlib import Path

import yaml
from ml_collections import ConfigDict

import torch

from train_with_stylegan.trainer import DIffGANTrainer
from train_with_stylegan.discriminator import Discriminator
from train_with_stylegan.utils import configure_unet_model_from_pretrained

from improved_diffusion.unet import UNetModel


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-t", "--teacher", type=Path, required=True)
    parser.add_argument("-p", "--params-path", type=Path, required=True)
    parser.add_argument("-d", "--device", type=str, required=True)
    parser.add_argument("--project", type=str, default="diff-to-gan")
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--save-code", action="store_true")
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def load_params(args):
    with args.params_path.open("r") as f:
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

    # discriminator = Discriminator(**params.discriminator, 
    #                               image_resolution=params.training.resolution,
    #                               time_countinious=params.training.sampling_countinious,
    #                               n_timesteps=params.training.n_timesteps)
    # discriminator.to(device)
    # discriminator.train()
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
        disc=None,
        params=params,
        device=device,
        log_dir=log_dir,
    ).run_training(args)


if __name__=="__main__":
    args = parse_args()
    train(args)
