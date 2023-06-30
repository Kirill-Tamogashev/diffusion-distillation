"""
Train a diffusion model on images.
"""

import argparse
import os

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
    log_dir = os.path.join("./wandb", args.run_name)
    os.environ["OPENAI_LOGDIR"] = log_dir

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    # logger.log(
    #     f"""
    #     LR: {args.lr}
    #     EPOCHS: {args.n_epochs}
    #     N DIFFUSION STEPS {args.diffusion_steps}
    #     N ITERATIONS PER EPOCH {args.n_iterations}
    #     BATCH TRAIN: {args.batch_size_train}
    #     """
    # )
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        params=args,
        log_dir=log_dir,
        schedule_sampler=schedule_sampler
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="synthetic_dataset",
        run_name="test-run",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_pic_interval=5_000,
        log_interval=5_000,
        save_interval=5_000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        use_wandb=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
