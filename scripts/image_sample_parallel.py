"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

BASE_IMAGES_PATH = Path("./synthesized_images")


def create_argparser():
    defaults = dict(
        world_size=1,
        progress=True,
        clip_denoised=True,
        num_samples=10000,
        batch_size=32,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main_parallel():
    args = create_argparser().parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    run_name = BASE_IMAGES_PATH / args.run_name
    if run_name.exists():
        raise AssertionError("Such run already exists, choose another name")
    else:
        run_name.mkdir(parents=True)
    mp.spawn(
        run_parallel_inference,
        args=(args.world_size, args, run_name, ),
        nprocs=args.world_size,
        join=True
    )


def run_parallel_inference(rank, world_size, args, run_name):
    logger.configure()
    logger.log("creating model and diffusion...")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )    
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.eval()
    model.to(rank)
    
    n_steps = 0
    while n_steps * args.batch_size < args.num_samples / world_size:
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            progress=args.progress
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).cpu().numpy()
    
        save_path = os.path.join(
            run_name, f"samples_rank-{rank}-sample-{n_steps}.npz"
        )
        logger.log(f"saving to {save_path}")
        np.savez(save_path, sample)
        n_steps += 1



if __name__ == "__main__":
    main_parallel()