'''
violinplot_light_intensity.py

Ce script permet d'afficher violinplot présentant la répartition de l'intensité lumineuse des images de chaque LABEL.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.image as mpimg

# Fonction pour calculer l'intensité lumineuse moyenne d'une image
def calc_mean_intensity(image_path):
    img = mpimg.imread(image_path)
    # Convertir en nuances de gris si l'image est en couleur
    if img.ndim == 3:
        img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        img_gray = img
    return np.mean(img_gray)


# Appliquer la fonction à chaque chemin d'image pour obtenir l'intensité moyenne
df_images['MEAN_INTENSITY'] = df_images['PATH'].apply(calc_mean_intensity)
df_images['Masque'] = 'Sans masque'
df_images_masked['Masque'] = 'Avec masque'
df_combined = pd.concat([df_images, df_images_masked])
df_combined['COMBINED_INTENSITY'] = df_combined['MEAN_INTENSITY'].combine_first(df_combined['MEAN_INTENSITY_MASK'])
df_combined['Label_Masque'] = df_combined['LABEL'] + ' (' + df_combined['Masque'] + ')'

palette = {
    'Normal (Sans masque)': '#A1C9F4',
    'Normal (Avec masque)': '#517EA4',
    'Lung_Opacity (Sans masque)': '#8DE5A1',
    'Lung_Opacity (Avec masque)': '#4E7D51',
    'COVID (Sans masque)': '#FFB482',
    'COVID (Avec masque)': '#BF6D41',
    'Viral Pneumonia (Sans masque)': '#D0BBFF',
    'Viral Pneumonia (Avec masque)': '#7E6CBF'
}

new_labels = ['COVID\nAvec masque', 'COVID\nSans masque',
              'Lung_Opacity\nAvec masque', 'Lung_Opacity\nSans masque',
              'Normal\nAvec masque', 'Normal\nSans masque',
              'Viral Pneumonia\nAvec masque', 'Viral Pneumonia\nSans masque']

# Tri par ordre alphabétique de Label_Masque
df_combined = df_combined.sort_values('Label_Masque')

plt.figure(figsize = (15, 8))
sns.violinplot(x = 'Label_Masque', y = 'COMBINED_INTENSITY', data = df_combined, palette = palette, split = True, inner = "quartile")
plt.ylabel('Intensité lumineuse moyenne')
plt.xlabel('')
plt.title("Distribution de l'intensité lumineuse moyenne normalisée par LABEL avec et sans application des masques")
plt.xticks(ticks = range(len(new_labels)), labels = new_labels)
plt.show()