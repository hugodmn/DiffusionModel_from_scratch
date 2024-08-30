from src.models.diffusion_module import DiffusionModel, LightningDiffusionModule
from src.models.TwoResUnet import TwoResUNet
import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToPILImage
import os 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
save_directory = 'generated_images'

def load_model(path : str, size : int = 64):
    model= TwoResUNet(dim = size)
    diff_model = DiffusionModel(model = model, device = device)
    light_module = LightningDiffusionModule(model=diff_model,  wandb_flag = False)
    checkpoint = torch.load(path, map_location=device)
    light_module.on_load_checkpoint(checkpoint=checkpoint)
    light_module = light_module
    return light_module


# Title
st.title("Pokésprite Generator")
MODEL_CHECKPOINT_PATH = "src/results/diffusion_model-epoch=139-train_loss=0.00.ckpt"
light_module = load_model(MODEL_CHECKPOINT_PATH)



# Button to generate a new Pokémon
if st.button('Generate Pokémon'):
    # Generate a new Pokémon image
    light_module.diff_model.eval()
    to_pil = ToPILImage()
    with st.spinner('Generating...'):
        
       
        generated_image = light_module.diff_model.p_sample_loop((1,3,64,64))
        image = generated_image.squeeze().detach().cpu().numpy()

        # If the image is in [-1, 1], unnormalize it to [0, 1]
        #image = (image + 1) / 2
        image = np.transpose(image, (1, 2, 0))
        # image = (image + 1) / 2
        image = np.clip(image, 0, 1)
        #generated_image = to_pil(generated_image)
        st.image(image, caption='Generated Pokémon Image', use_column_width=True)

       
        # Provide a button to save the generated image
        if st.button('Save Image'):
            # Save the generated image
            image_path = os.path.join(save_directory, "generated_pokemon.png")
            generated_image.save(image_path)
            st.success(f"Image saved to {image_path}")
