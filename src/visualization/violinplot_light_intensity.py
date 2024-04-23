'''
violinplot_light_intensity.py

Ce script permet d'afficher violinplot présentant la répartition de l'intensité lumineuse des images de chaque LABEL.
'''

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

plt.figure(figsize = (10, 6))
sns.violinplot(x = 'LABEL', y = 'MEAN_INTENSITY', data = df_images,
               palette = {'Normal': '#A1C9F4', 'Lung_Opacity': '#8DE5A1', 'COVID': '#FFB482', 'Viral Pneumonia': '#D0BBFF'})
plt.ylabel('Intensité lumineuse moyenne')
plt.title("Distribution de l'intensité lumineuse moyenne normalisée par LABEL")
plt.show()