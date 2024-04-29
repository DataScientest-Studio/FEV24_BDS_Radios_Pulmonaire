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

# Affichage des distributions
j = 1

plt.figure(figsize=(15, 10))

for mask_class in df_masks['LABEL'].unique():

    plt.subplot(2, 2, j)
    j = j +1
    sns.histplot(x = df_masks[df_masks['LABEL'] == mask_class]['RATIO'], bins=70, hue = df_masks['LABEL'],
                 label = mask_class, kde = True, palette = ['#A1C9F4', '#8DE5A1', '#FFB482', '#D0BBFF'], edgecolor = 'gray', data = df_masks)

    plt.legend()
    plt.suptitle("Ratio de la surface utile en appliquant les masques", fontsize = 18, y = 0.95)
    #plt.savefig('ratio_surface_utile.svg')  # pour sauvegarder la figure en .svg
    #plt.savefig('ratio_surface_utile.png')  # pour sauvegarder la figure en .png