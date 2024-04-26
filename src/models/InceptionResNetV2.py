import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import InceptionResNetV2
from sklearn.metrics import classification_report

# Fixer les random seeds
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# Chemin vers les données sur Kaggle
data_directory = "/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset/"

# Chemins vers les sous-dossiers contenant les images pour chaque classe
data_directories = [
    os.path.join(data_directory, "COVID/images"),
    os.path.join(data_directory, "Lung_Opacity/images"),
    os.path.join(data_directory, "Normal/images"),
    os.path.join(data_directory, "Viral Pneumonia/images")
]

# Fonction pour charger et prétraiter les images
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(299, 299))  # InceptionResNetV2 prend en entrée des images de taille 299x299
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalisation des valeurs de pixel
    return img_array

# Liste pour stocker les images et les étiquettes
images = []
labels = []

# Parcourir les sous-dossiers et récupérer les images et les étiquettes
for label, directory in enumerate(data_directories):
    images_in_directory = os.listdir(directory)
    images_in_directory = random.sample(images_in_directory, min(900, len(images_in_directory)))  # Limite à 900 images par classe
    images.extend([load_and_preprocess_image(os.path.join(directory, img)) for img in images_in_directory])
    labels.extend([label] * len(images_in_directory))

# Convertir en tableaux numpy
X = np.array(images)
y = np.array(labels)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle InceptionResNetV2 pré-entraîné
model = Sequential([
    InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3)),  # Utiliser les poids pré-entraînés sur ImageNet
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(4, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=30, batch_size=36, validation_data=(X_test, y_test))

# Afficher les courbes d'accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Afficher les courbes de loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Faire des prédictions sur les données de test
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Générer le rapport de classification
report = classification_report(y_test, y_pred_classes)
print(report)

# Afficher la matrice de confusion
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia'], yticklabels=['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
