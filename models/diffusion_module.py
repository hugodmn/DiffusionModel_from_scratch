import pytorch_lightning as pl
from models.TwoResUnet import TwoResUNet
import torch 
import torch.nn as nn
from models.utils import cosine_beta_schedule
import torch.nn.functional as F
from utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, identity, extract
import tqdm 

class LightningDiffusionModel(pl.LightningModule):


  
  def __init__(self, model : nn.Module, 
               timesteps : int = 1000,
               auto_normalize : bool = True
               ):
    super().__init__()


    self.beta_scheduler = cosine_beta_schedule()
    self.model = model

    betas = self.beta_scheduler_fn(timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
    print(alphas_cumprod, alphas_cumprod_prev, posterior_variance)

    register_buffer = lambda name, val: self.register_buffer(
    name, val.to(torch.float32)
)

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
        preds = self.model(x, batched_timestamps)
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
        batch, device = shape[0], "mps"

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