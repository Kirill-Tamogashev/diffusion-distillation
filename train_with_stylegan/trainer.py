from typing import Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision.utils import make_grid
import numpy as np
import wandb
from tqdm import tqdm

from train_with_stylegan.utils import (
    DiffusionScheduler
)


class DIffGANTrainer(nn.Module):
    def __init__(
        self,
        teacher:    nn.Module,
        student:    nn.Module,
        params,
        device:     torch.device,
        log_dir:    Path,
        t_delta:    float = 0.001,
        rank:       int = 0
    ) -> None:
        super().__init__()

        self._rank = rank
        self._log_dir = log_dir
        self._run = None  # Placeholder for wandb run
        self._use_wandb = False
        self._t_delta = t_delta
        self._device = device

        self._teacher = teacher
        self._student = student
        self._params = params.training
        assert self._params.solver in {"heun", "euler"}, "Solver is not known"

        self.scheduler = DiffusionScheduler(
            b_min=self._params.beta_min,
            b_max=self._params.beta_max,
            n_timesteps=self._params.n_timesteps,
            continuous=self._params.continuous,
            device=self._device
        )

        self._opt_stu = optim.AdamW(student.parameters(), lr=self._params.lr)

    def _sample_t_and_s(self):
        t = torch.randint(0, self._params.n_timesteps, (self._params.batch_size, ))
        s = t - self._params.step_size * self._params.n_timesteps
        s = torch.maximum(s, torch.zeros(self._params.batch_size))
        return t.to(self._device), s.to(self._device)

    def _compute_target_euler(self, z, t, s):
        """
        y(t) = y_student_theta(eps, t)
        x_t = alpha_t * y_t + sigma_t * z_t
        y(s)_target = SG[y(t) + lambda * delta * (f_teacher(x_t, t) - y(t))]
        """
        alpha_t, sigma_t = self.scheduler.get_schedule(t)
        alpha_s, sigma_s = self.scheduler.get_schedule(s)
        lambda_prime = alpha_s / alpha_t - sigma_s / sigma_t

        with torch.no_grad():

            y_t = self._student(z, t).sample

            x_t = alpha_t * y_t + sigma_t * z
            eps_t = self._teacher(x_t.float(), t).sample
            f_t = (x_t - sigma_t * eps_t) / alpha_t

            y_target = y_t + lambda_prime * (f_t - y_t)
        return y_target.detach().float()

    def _compute_target_heun(self, z, t, s, eps=1e-10) -> Tuple[torch.Tensor, torch.Tensor]:

        alpha_t, sigma_t = self.scheduler.get_schedule(t)
        alpha_s, sigma_s = self.scheduler.get_schedule(s)
        lambda_prime = alpha_s / (alpha_t + eps) - sigma_s / (sigma_t + eps)

        with torch.no_grad():

            y_t = self._student(z, t).sample

            x_t = alpha_t * y_t + sigma_t * z
            eps_t = self._teacher(x_t.float(), t).sample
            f_t = (x_t - sigma_t * eps_t) / alpha_t

            y_s = y_t + lambda_prime * (f_t - y_t)

            x_s = alpha_s * y_s + sigma_s * z
            eps_s = self._teacher(x_s.float(), s).sample
            f_s = (x_s - sigma_s * eps_s) / alpha_s

            y_target = y_t + 0.5 * ((f_t - y_t) + (f_s - y_s))

        return y_target.detach().float()

    def train(self, *args):

        if self._use_wandb and self._rank == 0:
            wandb.watch(
                self._student, log="gradients",
                log_freq=self._params.grads_log_freq
            )

        with tqdm(total=self._params.n_iterations) as pbar:
            for step in range(1, self._params.n_iterations):

                dims = (self._params.batch_size, 3, self._params.resolution, self._params.resolution)
                z = torch.randn(dims, device=self._device)
                t, s = self._sample_t_and_s()
                t_max = torch.ones(self._params.batch_size, device=self._device) * (self._params.n_timesteps - 1)

                # compute target for s and t_max
                if self._params.solver == "euler":
                    y_s_hat = self._compute_target_euler(z, t, s)
                elif self._params.solver == "heun":
                    y_s_hat = self._compute_target_heun(z, t, s)
                y_t_max_hat = self._teacher(z, t_max).sample

                # compute predictions for s and t_max
                y_s = self._student(z, s).sample
                y_t_max = self._student(z, t_max).sample

                mse_loss = F.mse_loss(y_s, y_s_hat, reduction="mean")
                boundary_loss = F.mse_loss(y_t_max, y_t_max_hat, reduction="mean")

                self._opt_stu.zero_grad()
                loss = mse_loss / (self._params.step_size ** 2) + self._params.boundary_coeff * boundary_loss
                assert not loss.isnan(), "Loss is NaN"

                loss.backward()
                self._opt_stu.step()

                loss_dict = {
                    "loss":     loss.item(),
                    "mse":      mse_loss.item(),
                    "boundary": boundary_loss.item(),
                    "lr":       self._opt_stu.param_groups[-1]["lr"]  # noqa
                }
                if self._rank == 0:
                    self._log_data(loss_dict)

                if step % self._params.image_log_freq == 0 and self._rank == 0:
                    self._generate_student_images_to_log(step)
                    self._generate_teacher_images_to_log(step)

                if step % self._params.save_ckpt_freq == 0 and self._rank == 0:
                    self._save_checkpoint(step)

                pbar.update()

    def _get_teacher_step(
            self,
            x_0:    torch.Tensor,
            x_t:    torch.Tensor,
            t:      torch.Tensor,
    ):
        alpha_t, sigma_t = self.scheduler.get_schedule(t)
        alpha_prev, sigma_prev = self.scheduler.get_schedule(t - 1)
        current_alpha_t = alpha_t.pow(2) / alpha_prev.pow(2)

        pred_original_sample = (x_t - sigma_t * x_0) / alpha_t
        pred_original_sample.clamp_(-1.0, 1.0)

        pred_original_sample_coeff = alpha_prev * (1 - current_alpha_t) / sigma_t.pow(2)
        current_sample_coeff = current_alpha_t.sqrt() * (1 - alpha_prev.pow(2)) / sigma_t.pow(2)

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * x_t
        variance = 1 - current_alpha_t
        noise = (variance ** 0.5) * torch.randn_like(pred_prev_sample,
                                                     dtype=pred_prev_sample.dtype,
                                                     device=pred_prev_sample.device)
        return (pred_prev_sample + noise).float()

    @torch.no_grad()
    def sample_with_teacher(self, n_images: int = 20):
        images = torch.randn(n_images, 3, self._params.resolution, self._params.resolution, device=self._device)
        for t in tqdm(self.scheduler.timesteps.flip((0,)), desc="Sampling images with teacher...", leave=False):
            t = torch.ones(n_images, dtype=torch.int, device=self._device) * t
            model_output = self._teacher(images, t).sample
            images = self._get_teacher_step(x_0=model_output, x_t=images, t=t)

        images = images.cpu() / 2.0 + 0.5
        return images

    def _save_checkpoint(self, step):
        ckpt_dict = {
            "student":      self._student.state_dict(),
            "student_opt":  self._opt_stu.state_dict(),
            "step":         step
        }
        torch.save(ckpt_dict, self._log_dir / f"ckpt-step-{step}.pt")

    def run_training(self, args):
        """Wraps wandb usage for training."""
        self._use_wandb = args.use_wandb
        if self._use_wandb and self._rank == 0:
            wandb_config = dict(
                dir=args.dir,
                config=self._params,
                project=args.project,
                name=args.name,
                resume=args.resume,
                save_code=args.save_code
            )
            with wandb.init(**wandb_config) as run:
                self._run = run
                self.train()
        else:
            self.train()

    def _log_data(self, data: dict, stage: str = "train"):
        if self._use_wandb:
            log_dict = {f"{stage}/{key}": value for key, value in data.items()}
            self._run.log(log_dict)

    @staticmethod
    def _log_images(images, prefix: str, message: str = ""):
        grid = make_grid(images, nrow=10).permute(1, 2, 0)
        grid = grid.data.mul(255).numpy().astype(np.uint8)
        wandb.log({f"{prefix}/Images": wandb.Image(grid, caption=message)})

    @torch.no_grad()
    def _generate_student_images_to_log(self, step: int, prefix: str = "student", n_images=20):
        z = torch.randn(n_images, 3, self._params.resolution, self._params.resolution)
        time = torch.zeros(n_images, device=self._device)
        images = self._student(z.to(self._device), time).sample

        images = images.cpu() / 2.0 + 0.5
        self._log_images(images=images, message=f"Student images on step {step}", prefix=prefix)

    def _generate_teacher_images_to_log(self, step: int, prefix: str = "teacher", n_images=20):
        images = self.sample_with_teacher(n_images=n_images)
        self._log_images(images=images, message=f"Teacher images on step {step}", prefix=prefix)
