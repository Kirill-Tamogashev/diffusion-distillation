"""
Train a diffusion model on images.
"""

import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    model_and_diff_args = model_and_diffusion_defaults()
    model_and_diff_args["diffusion_steps"] = args.diffusion_steps

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    teacher, teacher_diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diff_args.keys())
    )
    teacher.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    teacher.to(dist_util.dev())
    
    student, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diff_args.keys())
    )
    student.to(dist_util.dev())

    logger.log(f"run name is {args.run_name}")
    logger.log("training...")
    TrainLoop(
        teacher=teacher,
        student=student,
        teacher_diffusion=teacher_diffusion,
        device=dist_util.dev(),
        params=args
    ).run_per_batch_distillation()


def create_argparser():
    defaults = dict(
        n_epochs=100,
        lr=1e-4,
        model_path='',
        img_size=256,
        hidden_coeff=0.005,
        diffusion_steps=1_000,
        n_iterations=1500, 
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10_000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
