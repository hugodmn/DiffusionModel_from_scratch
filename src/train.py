import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from models.diffusion_module import DiffusionModel, LightningDiffusionModule
from models.TwoResUnet import TwoResUNet
from datasets.dataset import PokemonDataset  # Assuming your dataset class is in a file named dataset.py
import wandb
import torch
from configs.dotdict import load_yaml_into_dotdict
from pytorch_lightning.callbacks import ModelCheckpoint

def main(args):
    # Data transformations
  
    torch.set_float32_matmul_precision('high')
    if wandb.run:
        wandb.finish()

    params = load_yaml_into_dotdict(args.params_path)
    # Dataset and DataLoader
    dataset = PokemonDataset(csv_file='datasets/pokesprites/pokemon.csv', img_dir='datasets/pokemon/images')
    
    train_size = int(0.99 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=params.model.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=params.model.batch_size, shuffle=False, num_workers=4)


    # Initialize the model
    model = TwoResUNet(dim = params.model.input_dim)  # Initialize with appropriate parameters

    # Initialize Lightning Diffusion Model
    diffusion_model = DiffusionModel(model, 
                                    img_size = params.diffusion.image_size,
                                    timesteps=params.diffusion.timesteps)

    # Initialize Diffusion Module with EMA
    lightning_module = LightningDiffusionModule(diffusion_model, 
                                                train_lr=float(params.training.train_lr))

    # Initialize WandB logger
    wandb_logger = WandbLogger(project="diffusion_model_project")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='results',
        filename='diffusion_model-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        mode='min',
        monitor='train_loss'
    )
    # Initialize Trainer
    trainer = pl.Trainer(logger=wandb_logger, 
                        val_check_interval=0.25,
                        accelerator = "gpu",
                        devices = "auto",
                        enable_checkpointing=True,
                        default_root_dir="results",
                        callbacks=[checkpoint_callback]
                        )

    # Train the model
    trainer.fit(lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a diffusion model on Pokémon images")
    
    parser.add_argument('--params_path', type=str, required = True, help='Path of params file (should be in configs file)')

   

    args = parser.parse_args()

    print(f'Params configuration path: {args.params_path}')
    main(args)
