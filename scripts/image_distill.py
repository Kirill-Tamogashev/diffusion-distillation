"""
Train a diffusion model on images.
"""
import os
import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


def main():
    os.environ["OPENAI_LOGDIR"] = "/home/iddpm/distill_logs"
    args = create_argparser().parse_args()
    model_and_diff_args = model_and_diffusion_defaults()
    model_and_diff_args["diffusion_steps"] = args.diffusion_steps

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating teacher model and diffusion...")
    teacher, teacher_diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diff_args.keys())
    )
    teacher.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    teacher.to(dist_util.dev())
    logger.log("creating student model ...")
    student = create_model(
        **args_to_dict(args, model_and_diff_args.keys())
    )
    if args.reinit:
        student.load_state_dict(
            dist_util.load_state_dict(args.reinit_path, map_location="cpu")
        )
    # student, _ = create_model_and_diffusion(
        # **args_to_dict(args, model_and_diff_args.keys())
    # )
    student.to(dist_util.dev())

    logger.log(f"run name is {args.run_name}")
    logger.log("training...")
    logger.log(
        f"""
        LR: {args.lr}
        EPOCHS: {args.n_epochs}
        N DIFFUSION STEPS {args.diffusion_steps}
        N ITERATIONS PER EPOCH {args.n_iterations}
        BATCH TRAIN: {args.batch_size_train}
        BATCH SAMPLE: {args.batch_size_sample}
        HIDDEN LOSS COEFF: {args.hidden_coeff}
        """
    )
    TrainLoop(
        teacher=teacher,
        student=student,
        teacher_diffusion=teacher_diffusion,
        device=dist_util.dev(),
        params=args
    ).run_per_batch_distillation()


def create_argparser():
    defaults = dict(
        reinit=False,
        reinit_path="",
        n_epochs=100,
        lr=1e-5,
        model_path='',
        img_size=256,
        hidden_coeff=0.005,
        diffusion_steps=1000,
        n_iterations=5000, 
        weight_decay=0.05,
        lr_anneal_steps=0,
        batch_size_sample=16,
        batch_size_train=4,
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
