'''
show_img_and_mask.py

Ce script permet d'afficher une image et son masque.
'''

import cv2
import matplotlib.pyplot as plt

# Saisir les chemins de l'image et du masque souhaitées
img = cv2.imread("")
mask = cv2.imread("")

fig, axes = plt.subplots(1, 2, figsize = (6, 4))

axes[0].imshow(img, cmap = "gray")
axes[0].set_title('Image')
axes[0].axis('off')

axes[1].imshow(mask, cmap = "gray")
axes[1].set_title('Masque')
axes[1].axis('off')

plt.show()