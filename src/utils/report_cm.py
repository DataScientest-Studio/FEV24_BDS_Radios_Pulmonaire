'''
report_cm.py

Affichage du classification report et de la matrice de confusion d'un modèle.
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Calcul des prédictions et récupération des probabilités les plus élevées
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis = 1)
y_test_classes = np.argmax(y_test, axis = 1)

# Liste ordonnée des noms de classe
class_labels = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia']

# Affichage du report
report = classification_report(y_test_classes, y_pred_classes, target_names = class_labels)
print("Rapport de Classification :\n", report)

# Affichage de la matrice de confusion
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize = (5, 4))
sns.heatmap(conf_matrix, annot = True, fmt = "d", cmap = "Blues", xticklabels = class_labels, yticklabels = class_labels, annot_kws = {"fontsize":8})
plt.title('Matrice de Confusion : DenseNet201 fine-tuned', fontsize = 10)
plt.xlabel('Labels prédits', fontsize = 9)
plt.ylabel('Vrais labels', fontsize = 9)
plt.tick_params(axis = 'both', which = 'major', labelsize = 8)
plt.yticks(rotation = 0.5)
plt.xticks(rotation = 1)

plt.show()