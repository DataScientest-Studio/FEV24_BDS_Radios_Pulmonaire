'''
extract_metadata_from_img.py

Script permettant d'extraire les métadonnées directement depuis les images d'un dossier spécifié.
La sortie est un dataframe, enregistré en *.csv.
'''

import os
from PIL import Image
import pandas as pd

# Chemin vers le dossier contenant les catégories
root_folder = ''

# Initialiser une liste pour stocker les métadonnées
data = []

# Parcourir chaque catégorie de dossier
for category in ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']:
    images_folder = os.path.join(root_folder, category)
    if os.path.isdir(images_folder):
        for image_name in os.listdir(images_folder):
            image_path = os.path.join(images_folder, image_name)
            if os.path.isfile(image_path):
                # Ouvrir l'image pour obtenir des informations supplémentaires
                with Image.open(image_path) as img:
                    width, height = img.size
                    resolution = f"{width}x{height}"
                    channels = img.mode
                    channels_count = len(channels)
                    data.append({
                        'FILE NAME': image_name,
                        'FORMAT' : "",
                        'SIZE': resolution,
                        'LABEL': category,
                        'CHANNELS': channels_count,
                        'PATH': image_path})

# Créer un DataFrame à partir des métadonnées collectées
df = pd.DataFrame(data)
df['FORMAT'] = df['FILE NAME'].str.split('.').str[1]
df['FILE NAME'] = df['FILE NAME'].str.split('.').str[0]
df['FORMAT'] = df['FORMAT'].str.upper()
df.to_csv('df_metadata.csv', index = False)