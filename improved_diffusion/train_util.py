import copy
import functools
import os

from IPython.display import clear_output

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
import wandb
from tqdm import tqdm


class TrainLoop:
    def __init__(
        self,
        *,
        teacher,
        student,
        teacher_diffusion,
        device,
        params,
    ):
        self.device=device
        self.teacher_model = teacher
        self.student_model = student
        self.teacher_diffusion = teacher_diffusion
        
        self.run_name = params.run_name
        self.img_size = params.img_size
        self.n_epochs = params.n_epochs
        self.n_per_epoch_iterations = params.n_iterations
        self.hidden_coeff = params.hidden_coeff
        self.batch_size_sample = params.batch_size_sample
        self.batch_size_train = params.batch_size_train
        self.lr = params.lr
        self.ema_rate = (
            [params.ema_rate] if isinstance(params.ema_rate, float)
            else [float(x) for x in params.ema_rate.split(",")]
        )
        self.log_interval = params.log_interval
        self.save_interval = params.save_interval
        self.resume_checkpoint = params.resume_checkpoint
        # self.n_sample_steps = sample_steps
        # self.data = data
        # self.use_fp16 = use_fp16
        # self.fp16_scale_growth = fp16_scale_growth
        # self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        # self.microbatch = microbatch if microbatch > 0 else batch_size
        self.weight_decay = params.weight_decay
        self.lr_anneal_steps = params.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size_train * dist.get_world_size()

        self.model_params = list(self.student_model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        
        # if self.use_fp16:
        #     self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.student_model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        
        if self.run_name != "test-run":
            logger.log("Init wandb logger")
            self.wandb_logger = wandb.init(
                project="distillation", 
                name=self.run_name
            )
        else:
            self.wandb_logger = None

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.student_model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    # def _setup_fp16(self):
    #     self.master_params = make_master_params(self.model_params)
    #     self.model.convert_to_fp16()

    # def run_loop(self):
    #     while (
    #         not self.lr_anneal_steps
    #         or self.step + self.resume_step < self.lr_anneal_steps
    #     ):
    #         batch, cond = next(self.data)
    #         self.run_step(batch, cond)
    #         if self.step % self.log_interval == 0:
    #             logger.dumpkvs()
    #         if self.step % self.save_interval == 0:
    #             self.save()
    #             # Run for a finite amount of time in integration tests.
    #             if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
    #                 return
    #         self.step += 1
    #     # Save the last checkpoint if it wasn't already saved.
    #     if (self.step - 1) % self.save_interval != 0:
    #         self.save()

    # def run_step(self, batch, cond):
    #     self.forward_backward(batch, cond)
    #     if self.use_fp16:
    #         self.optimize_fp16()
    #     else:
    #         self.optimize_normal()
    #     self.log_step()
        
    def run_per_batch_distillation(self):
        for epoch in range(1, self.n_epochs + 1):
            clear_output(wait=True)
            logger.log(f"START EPOCH: {epoch}")
            noise = th.randn((self.batch_size_sample, 3, self.img_size, self.img_size), device=self.device)
            with th.no_grad():
                logger.log("START SAMPLING TRAGECTORY")
                _, full_trajectory = self.teacher_diffusion.p_sample_loop(
                    model=self.teacher_model,
                    shape=noise.size(),
                    clip_denoised=True,
                    noise=noise
                )
            logger.log("TRAJECTORY SAMPLED, RUNNING EPOCH ...")
            for _ in tqdm(range(self.n_per_epoch_iterations)):
                tau = th.randint(len(full_trajectory), (1,)).item()
                batch = full_trajectory[tau]
                batch_idx = th.randint(self.batch_size_sample, (self.batch_size_train,))
                losses = self.run_distill_iteration(
                    x_t=batch["input"][batch_idx].to(self.device), 
                    t=batch["time"][batch_idx].to(self.device),
                    hidden=batch["hidden"][batch_idx].to(self.device),
                    eps=batch["eps"][batch_idx].to(self.device),
                )
                self.optimize_normal()
                
                if self.wandb_logger is not None:
                    self.wandb_logger.log(losses)
                
                if self.step and self.step % self.save_interval == 0:
                    self.save()
                self.step  += 1
    
    def run_distill_iteration(self, x_t, t, hidden, eps):
        student_out, student_hidden = self.student_model(x_t, t)
        # print
        mse_out = (student_out.flatten(1) - eps.flatten(1)).pow(2)
        mse_hidden = (student_hidden.flatten(1) - hidden.flatten(1)).pow(2)
        
        loss_model = th.mean(mse_out.sum(dim=-1))
        loss_hidden = th.mean(mse_hidden.sum(dim=-1))
        
        loss = loss_model + loss_hidden * self.hidden_coeff
        if th.isnan(loss).any():
            raise AssertionError("Loss is NaN. Terminating training...")
        loss.backward()
        return {
            "loss": loss.item(), 
            "loss_hidden": loss_hidden.item(), 
            "loss_model": loss_model.item(),
            "loss_hidden_pixelwise": mse_hidden.mean(),
            "loss_out_pixelwise": mse_out.mean(),
        }         

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    # def optimize_fp16(self):
    #     if any(not th.isfinite(p.grad).all() for p in self.model_params):
    #         self.lg_loss_scale -= 1
    #         logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
    #         return

    #     model_grads_to_master_grads(self.model_params, self.master_params)
    #     self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
    #     self._log_grad_norm()
    #     self._anneal_lr()
    #     self.opt.step()
    #     for rate, params in zip(self.ema_rate, self.ema_params):
    #         update_ema(params, self.master_params, rate=rate)
    #     master_params_to_model_params(self.model_params, self.master_params)
    #     self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        # if self.use_fp16:
        #     master_params = unflatten_master_params(
        #         self.model.parameters(), master_params
        #     )
        state_dict = self.student_model.state_dict()
        for i, (name, _value) in enumerate(self.student_model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.student_model.named_parameters()]
        # if self.use_fp16:
        #     return make_master_params(params)
        # else:
        return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
