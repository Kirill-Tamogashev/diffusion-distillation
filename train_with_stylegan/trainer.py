from functools import partial
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
from train_with_stylegan.loss import LPIPS


def update_ema(current_params, ema_params_state_dict, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for param_name, param_value in current_params:
        ema_param = ema_params_state_dict[param_name]
        param_value.detach().mul_(rate).add_(ema_param, alpha=1 - rate)


class DIffGANTrainer(nn.Module):
    def __init__(
        self, 
        teacher:    nn.Module, 
        student:    nn.Module,
        disc:       nn.Module,
        params:     dict,
        device:     torch.device,
        log_dir:    Path,
        t_delta:    float = 0.001,
    ):
        super().__init__()
        
        self._log_dir = log_dir
        self._run = None # Placeholder for wandb run
        self._use_wandb = False
        self._t_delta = t_delta
        self._device = device

        self._teacher = teacher
        self._student = student
        # self._disc = disc
        self._params = params.training
        self._diffuison_scheduler = DiffusionScheduler(
            b_min=self._params.beta_min,
            b_max=self._params.beta_max,
            n_steps=self._params.n_steps,
            continious=self._params.sampling_countinious,
            device=self._device
        )
        
        self._lpips = LPIPS()
        self._lpips.to(self._device)
        
        self._opt_stu = optim.AdamW(student.parameters(), lr=self._params.lr_gen)
        # self._opt_d = optim.AdamW(disc.parameters(), lr=self._params.lr_disc)
        
        self._scheduler_g = None
        # self._scheduler_d = None
        
        # self._ema_stundent = student
        # self._ema_student_state_dict = student.state_dict()
        # self._ema_reload_required = True
   
    def _sample_t_and_s(self):
        if self._params.sampling_countinious:
            U1 = torch.rand(self._params.batch_size)
            t = 1 + (self._t_delta - 1) * U1
            s = torch.maximum(t - self._params.step_size, torch.ones(self._params.batch_size) * self._t_delta)
        else:
            assert isinstance(self._params.step_size, int), \
                f"For discrete time step_size should be int, got {self._params.step_size}"
            
            t = np.random.randint(1, self._params.n_timesteps, self._params.batch_size)
            t = torch.tensor(t)

        s = torch.maximum(t - self._params.step_size, torch.zeros(self._params.batch_size ))
            
        return t.to(self._device), s.to(self._device)
    
    def _compute_target(self, z, t, s):
        """
        y(t) = y_student_theta(eps, t)
        x_t = alpha_t * y_t + sigma_t * z_t
        y(s)_target = SG[y(t) + lmbda * delta * (f_teacher(x_t, t) - y(t))]
        """
        alpha_t, sigma_t = self._diffuison_scheduler.get_schedule(t)
        alpha_s, sigma_s = self._diffuison_scheduler.get_schedule(s)
        lambda_prime = 1 - (alpha_t * sigma_s) / (alpha_s * sigma_t)
        
        with torch.no_grad():
            # if self._ema_reload_required:
            #     self._ema_stundent.load_state_dict(self._ema_student_state_dict)
            #     self._ema_reload_required = False

            y_t_ema = self._student(z, t).sample
            x_t_ema = alpha_t * y_t_ema + sigma_t * z
            
            eps_t_ema = self._teacher(x_t_ema.float(), t).sample
            x_0_ema = (x_t_ema - sigma_t * eps_t_ema) / alpha_t
            y_target = y_t_ema + lambda_prime * (x_0_ema - y_t_ema)
        return y_target.detach().float()
    
    def train(self):

        if self._use_wandb:
            wandb.watch(
                self._student, log="gradients", 
                log_freq=self._params.gen_log_freq
            )
            # wandb.watch(
            #     self._disc, log="gradients", 
            #     log_freq=self._params.disc_log_freq
            # )
        with tqdm(total=self._params.n_steps) as pbar:
            for step in range(1, self._params.n_steps):
                
                dims = (self._params.batch_size, 3, 
                        self._params.resolution, self._params.resolution)
                z = torch.randn(dims, device=self._device)
                
                t, s = self._sample_t_and_s()
                
                # compute target
                y_target = self._compute_target(z, t, s)

                y_s = self._student(z, s).sample
                t_max = torch.ones(self._params.batch_size, device=self._device)
                y_max_teacher = self._teacher(z, t_max).sample
                y_max_stu = self._student(z, t_max).sample
                
                
                mse_loss = F.mse_loss(y_s.float(), y_target.float(), reduction="mean")
                boundary_loss = F.mse_loss(y_max_stu, y_max_teacher, reduction="mean")

                self._opt_stu.zero_grad()
                loss = mse_loss + self._params.boundary_coeff * boundary_loss
                loss.backward()
                self._opt_stu.step()

                loss_dict = {
                    "mse": mse_loss,
                    "boundary": boundary_loss
                }
                self._log_losses(loss_dict)

                # # Calculate Losses
                # gen_losses = self._optimize_generator(y_s, y_target, y_max_teacher, y_max_stu, s)
                
                # if step % self._params.discr_update_freq == 0:
                #     discr_losses = self._optimize_discriminator(y_s, y_target, s)
                #     self._log_losses(discr_losses)
                
                # EMA update
                # if step % self._params.ema_update_freq == 0:
                #     update_ema(
                #         self._student.named_parameters(), 
                #         self._ema_student_state_dict, 
                #         rate=self._params.ema_rate
                #     )
                #     self._ema_student_state_dict = self._student.state_dict()
                #     self._ema_reload_required = True 

                if step % self._params.image_log_freq == 0:
                    self._generate_images_to_log(step)
                    
                if step % self._params.save_ckpt_freq == 0:
                    self._save_checkpoint(step)
                
                pbar.update()

    def _save_checkpoint(self, step):
        ckpt_dict = {
            "student": self._student.state_dict(),
            "student_opt": self._opt_stu.state_dict(),
            "step": step
        }
        torch.save(ckpt_dict, self._log_dir / f"ckpt-step-{step}.pt")

    def _optimize_generator(
        self, 
        pred:           torch.Tensor, 
        target:         torch.Tensor,
        y_stu_max:      torch.Tensor,
        y_target_max:   torch.Tensor,
        s:              torch.Tensor,
    ) -> dict:

        self._opt_stu.zero_grad()
        gan_loss = F.softplus(- self._disc(pred, s)).mean()
        
        mse_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        
        lpips_loss = self._lpips.calc_loss(pred.float(), target.float()).squeeze().mean()
        
        boundary_loss = F.mse_loss(y_stu_max, y_target_max)
        loss = (
            gan_loss
            + self._params.mse_coeff * mse_loss
            + self._params.lpips_coeff * lpips_loss
            + self._params.boundary_coeff * boundary_loss
        )
        loss.backward()
        self._opt_stu.step()
        if self._scheduler_g is not None:
            self._scheduler_g.step()
        
        loss_dict = {
            "loss":             loss,
            "gan_loss":         gan_loss,
            "mse_loss":         mse_loss,
            "lpips":            lpips_loss,
            "boundary_loss":    boundary_loss,
        }
        return loss_dict
    
    
    def _optimize_discriminator(self, pred, target, s):
        self._opt_d.zero_grad()
        
        real_logits = self._disc(target, s)
        loss_d_real = F.softplus(- real_logits).mean()
        fake_logits = self._disc(pred.detach(), s)
        loss_d_fake = F.softplus(fake_logits).mean()

        d_loss = (loss_d_real + loss_d_fake).float()
        d_loss.backward()
        self._opt_d.step()
        if self._scheduler_d is not None:
            self._scheduler_d.step()
        
        d_loss_dict = {
            "d_loss":       d_loss,
            "loss_d_real":  loss_d_real,
            "loss_d_fake":  loss_d_fake
        }
        return d_loss_dict
        
            
                
    def run_training(self, args):
        """Wraps wandb usage for training."""
        self._use_wandb = args.use_wandb
        if self._use_wandb:
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
            
            
    def _log_losses(self, losses: dict, stage: str = "train"):
        if self._use_wandb:
            
            losses = {f"{stage}/{key}" : value for key, value in losses.items()}
            self._run.log(losses)
    
    def _log_images(self, images, message: str = ""):
        images = wandb.Image(make_grid(images, nrow=10), caption=message)
        wandb.log({"sampled images": images})
        
    def _generate_images_to_log(self, step: int, n_images=20):
        z = torch.randn(n_images, 3, self._params.resolution, self._params.resolution)
        time = torch.zeros(n_images, device=self._device)
        images = self._student(z.to(self._device), time).sample
        self._log_images(images=images, message=f"Images on step {step}")