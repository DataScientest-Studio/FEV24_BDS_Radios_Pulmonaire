'''
preprocessing.py

Ce script définit une fonction pour le preprocessing des images.
La fonction renvoie un ensemble d'entrainement et un ensemble de test.
'''

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, to_categorical

def preproc_img(df_images, df_masks, n_img, normalize, files_path, resolution, with_masks):
    np.random.seed(42)

    # Gestion des erreurs
    if resolution[2] != 1 and resolution[2] != 3:
        return print("Le nombre de canaux doit être de 1 (en nuances de gris) ou de 3 (en couleur)")

    if resolution[0] != resolution[1]:
        return print("La largeur de l'image doit être la même que sa hauteur.")
    
    if normalize != 'imagenet' and normalize != 'simple':
        print("Attention : aucune normalisation n'a été appliquée. Utilisez 'imagenet' pour une normalisation standardisée selon le mode opératoire du set ImageNet ou 'simple' pour simplement normaliser la valeur des canaux entre 0 et 1.")

    df_images_selected_list = []
    for label, group in df_images.groupby('LABEL'):
        n_samples = min(len(group), n_img)
        df_images_selected_list.append(group.sample(n=n_samples, replace=False))
    df_images_selected = pd.concat(df_images_selected_list)

    # Initialiser une liste pour stocker les images prétraitées
    images = []

    # Sélectionner le nombre d'image à utiliser par classe
    df_masks_selected = df_masks[df_masks['FILE_NAME'].isin(df_images_selected['FILE_NAME'])] if with_masks else None

    for i in range(len(df_images_selected)):
        img_path = df_images_selected[files_path].iloc[i]
        mask_path = df_masks_selected[files_path].iloc[i] if with_masks else None

        # Charger l'image avec PIL
        img = Image.open(img_path)
        img = img.convert("L")  # Convertir en niveaux de gris

        if resolution[2] == 3:
            img = img.convert("RGB")  # Convertir en mode RGB

        img_resized = img.resize((resolution[0], resolution[1]))

        # Normalisation des valeurs des pixels
        if normalize == 'imagenet'
            if resolution[2] == 1:  # Image en nuances de gris
                mean_gray = np.mean([0.485, 0.456, 0.406])
                std_gray = np.mean([0.229, 0.224, 0.225])
                img_normalized = (img_resized / 255.0 - mean_gray) / std_gray
            elif resolution[2] == 3:  # Image en couleur
                img_normalized = np.array(img_resized) / 255.0
                img_normalized -= np.array([0.485, 0.456, 0.406])
                img_normalized /= np.array([0.229, 0.224, 0.225])
        else:
            if normalize == 'simple':
                img_normalized = img_resized / 255
            else:
                img_normalized = img_resized

        # Ajouter l'image à la liste
        images.append(img_normalized)

    # Reshaper pour ajouter la dimension du canal
    data = np.array(images).reshape(-1, resolution[0], resolution[1], resolution[2])
    target = df_images_selected['LABEL']

    return data, target

# Utilisation de la fonction
data, target = preproc_img(df_images, df_masks, 
                           n_img = 1345, 
                           normalize = 'imagenet', 
                           files_path = 'PATH', 
                           resolution = [224, 224, 3], 
                           with_masks = False)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 42)

label_encoder = LabelEncoder()

# Encoder les labels textuels en entiers
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Appliquer l'encodage one-hot
y_train = to_categorical(y_train_encoded)
y_test = to_categorical(y_test_encoded)