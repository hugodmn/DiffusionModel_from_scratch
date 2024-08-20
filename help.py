import os
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm 


# Path to the folder containing Pokémon images
image_folder = 'datasets/pokesprites'  # Replace with the correct path
output_csv = 'pokemon.csv'


# Step 1: Load all images and their paths
images = []
image_paths = []
for dirname, _, filenames in os.walk(image_folder):
    for filename in filenames:
        img_path = os.path.join(dirname, filename)
        images.append(Image.open(img_path))
        image_paths.append(img_path)

# Step 2: Remove duplicates
hashes = {}
duplicates = []

# Use tqdm for progress bar
for i in tqdm(range(len(images)), desc="Processing images"):
    img_hash = hash(images[i].tobytes())  # Hashing the image data for faster comparison
    if img_hash in hashes:
        duplicates.append(i)
    else:
        hashes[img_hash] = i

# Filter out the duplicates
unique_images = [img for idx, img in enumerate(images) if idx not in duplicates]
unique_image_paths = [path for idx, path in enumerate(image_paths) if idx not in duplicates]

# Step 3: Rename images and create a CSV file
data = []
for img_num, img_path in enumerate(tqdm(unique_image_paths, desc="Renaming images")):
    # Generate new image name and path
    new_image_name = f"{img_num}.png"
    new_image_path = os.path.join(os.path.dirname(img_path), new_image_name)
    
    # Rename the image file
    os.rename(img_path, new_image_path)
    
    # Append data to list (name, number, path)
    pokemon_name = os.path.basename(img_path).split('.')[0]
    data.append([pokemon_name, img_num, new_image_path])

# Step 4: Save the data to a CSV file
df = pd.DataFrame(data, columns=['Name', 'Image Number', 'Path'])
df.to_csv(output_csv, index=False)

print(f"CSV file created at: {output_csv}")

