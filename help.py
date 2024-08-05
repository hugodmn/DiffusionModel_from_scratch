import os
import csv

def create_csv_from_images(image_dir, output_csv):
    fieldnames = ['Name', 'Filepath', 'Type1', 'Type2', 'Evolution']
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    name = os.path.splitext(file)[0]
                    filepath = os.path.join(root, file)
                    type1, type2, evolution = '', '', ''  # You can customize this part
                    writer.writerow({'Name': name, 'Filepath': filepath, 'Type1': type1, 'Type2': type2, 'Evolution': evolution})

if __name__ == "__main__":
    image_directory = 'path_to_image_directory'
    output_csv_file = 'pokemon_images.csv'
    create_csv_from_images(image_directory, output_csv_file)