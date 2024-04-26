'''
usefull_surface_ratio.py

Ce script permet de calculer le ratio de surface utile à partir de son masque. Ensuite, d'afficher la distribution des ratios par classe.
'''
# Packages nécessaires
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2

# Calcul et affiche une distribution des ratios de la surface utile pour chaque catégorie

# Calcul des ratios de surface utile

RATIO = []

for index, mask_path in enumerate(df_masks['PATH']):
    img_msk = Image.open(mask_path)
    arr_msk = np.array(img_msk)
    msk_size = arr_msk.shape

    # calcul du ratio surface utile à partir du mask
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_arr = np.array(mask_img)
    mask_ratio = np.round(np.count_nonzero(mask_arr)/(299*299),4)*100
    RATIO.append(mask_ratio)

df_masks['RATIO'] = pd.Series(RATIO)

# Affichage des distributions
j = 1

plt.figure(figsize=(15, 10))

for mask_class in df_masks['LABEL'].unique():

    plt.subplot(2, 2, j)
    j = j +1
    sns.histplot(x=mask_metadata[mask_metadata['mask_class']==mask_class]['mask_ratio'], bins=70, hue=mask_metadata['mask_class'],
                 label=mask_class, kde=True, palette=['#A1C9F4', '#8DE5A1', '#FFB482', '#D0BBFF'], edgecolor='gray', data=mask_metadata)

    plt.legend()
    plt.suptitle("Ratio de la surface utile en appliquant les masques", fontsize=18, y=0.95)
    #plt.savefig('ratio_surface_utile.svg')  # pour sauvegarder la figure en .svg
    #plt.savefig('ratio_surface_utile.png')  # pour sauvegarder la figure en .png