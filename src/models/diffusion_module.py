import pytorch_lightning as pl
from models.TwoResUnet import TwoResUNet
import torch 
import torch.nn as nn
from models.utils import cosine_beta_schedule, linear_beta_schedule
import torch.nn.functional as F
from utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, identity, extract
from tqdm import tqdm 
from ema_pytorch import EMA
from torchvision.utils import save_image
import wandb
from torch.optim import AdamW


class DiffusionModel(nn.Module):


  
    def __init__(self,
                model : nn.Module, 
                timesteps : int = 1000,
                auto_normalize : bool = True,
                img_size : int = 64,
                device : str = 'cuda'
                ):
        super().__init__()
        #ALSO DEAL WITH YAML CONFIGURATION FILES 
        #   
        self.device = device
        self.image_size = img_size
        self.beta_scheduler_fn = linear_beta_schedule
        self.diffusion_model = model

        betas = self.beta_scheduler_fn(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

        register_buffer = lambda name, val: self.register_buffer(
        name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)


        timesteps, *_ = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = timesteps
        
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity



    @torch.inference_mode()
    def p_sample(self, x : torch.Tensor, timestamp :int) -> torch.Tensor:
            b, *_, device = *x.shape, x.device
            batched_timestamps = torch.full(
                (b,), timestamp, device=device, dtype=torch.long
            )
            preds = self.diffusion_model(x, batched_timestamps)
            betas_t = extract(self.betas, batched_timestamps, x.shape)
            sqrt_recip_alphas_t = extract(
                self.sqrt_recip_alphas, batched_timestamps, x.shape
            )
            sqrt_one_minus_alphas_cumprod_t = extract(
                self.sqrt_one_minus_alphas_cumprod, batched_timestamps, x.shape
            )

            predicted_mean = sqrt_recip_alphas_t * (
                x - betas_t * preds / sqrt_one_minus_alphas_cumprod_t
            )

            if timestamp == 0:
                return predicted_mean
            else:
                posterior_variance = extract(
                    self.posterior_variance, batched_timestamps, x.shape
                )
                noise = torch.randn_like(x)
                return predicted_mean + torch.sqrt(posterior_variance) * noise
            

    @torch.inference_mode()
    def p_sample_loop(
            self, shape: tuple, return_all_timesteps: bool = False
        ) -> torch.Tensor:
            batch, device = shape[0], self.device

            img = torch.randn(shape, device=device)
            # This cause me a RunTimeError on MPS device due to MPS back out of memory
            # No ideas how to resolve it at this point

            # imgs = [img]

            for t in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps):
                img = self.p_sample(img, t)
                # imgs.append(img)

            ret = img  # if not return_all_timesteps else torch.stack(imgs, dim=1)

            ret = self.unnormalize(ret)
            return ret
    


    def q_sample(
            self, x_start: torch.Tensor, t: int, noise: torch.Tensor = None
        ) -> torch.Tensor:
            if noise is None:
                noise = torch.randn_like(x_start)

            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )

            return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_loss(
            self,
            x_start: torch.Tensor,
            t: int,
            noise: torch.Tensor = None,
            loss_type: str = "l2",
        ) -> torch.Tensor:
            
            if noise is None:
                noise = torch.randn_like(x_start)
            x_noised = self.q_sample(x_start, t, noise=noise)
            predicted_noise = self.diffusion_model(x_noised, t)

            if loss_type == "l2":
                loss = F.mse_loss(noise, predicted_noise)
            elif loss_type == "l1":
                loss = F.l1_loss(noise, predicted_noise)
            else:
                raise ValueError(f"unknown loss type {loss_type}")
            return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, device, img_size = *x.shape, x.device, self.image_size
        assert h == w == img_size, f"image size must be {img_size}"

        timestamp = torch.randint(0, self.num_timesteps, (1,)).long().to(device)
        x = self.normalize(x)
        return self.p_loss(x, timestamp)
    


class LightningDiffusionModule(pl.LightningModule):
    def __init__(self, 
                model : DiffusionModel,
                ema_update_every: int = 10,
                ema_decay: float = 0.995,
                train_lr: float = 1e-4,
                adam_betas: tuple[float, float] = (0.9, 0.99),
                wandb_flag : bool = True):
        super().__init__()
        self.diff_model = model 

        self.train_lr = train_lr
        self.adam_betas = adam_betas
        self.ema = EMA(self.diff_model, beta=ema_decay, update_every=ema_update_every)

        self.wandb_flag = wandb_flag
        if self.wandb_flag : 
            wandb.init(project="diffusion_model_project")

        self.step_counter = 0 

    def training_step(self, batch, batch_idx):
        self.step_counter += 1 
        images = batch
        loss = self.diff_model(images)
        self.log('train_loss', loss)
        if self.wandb_flag : 
            wandb.log({'train_loss': loss.item()})

        return loss 

    def validation_step(self, batch, batch_idx):
        self.ema.ema_model.eval()
 
        images = batch
        loss = self.diff_model(images)

        generated_images = self.diff_model.p_sample_loop((1, 3, self.diff_model.image_size, self.diff_model.image_size))
        save_image(generated_images, f'generated_images/step_{self.step_counter}.png')
        self.log('val_loss', loss)
        if self.wandb_flag : 
            wandb.log({"generated_image": [wandb.Image(generated_images, caption=f"Step {self.step_counter}")]})

  
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()

    def configure_optimizers(self):
        optimizer = AdamW(self.diff_model.parameters(), lr=self.train_lr, betas=self.adam_betas)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        # Save EMA state
        checkpoint['ema_state_dict'] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # Load EMA state
        self.ema.load_state_dict(checkpoint['ema_state_dict'])
