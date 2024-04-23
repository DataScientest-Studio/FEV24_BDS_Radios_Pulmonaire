'''
plot_metrics.py

Définition de deux fonctions pour afficher le suivi des quatre métriques intéressantes pendant l'entrainement des modèles.
Précision, Perte, AUC et F1 Score.
'''

import matplotlib.pyplot as plt
import numpy as np

def plot_lc(history):

    plt.figure(figsize=(14, 5))
    
    # Tracer la courbe de perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label = 'Perte d\'entraînement', color = 'lightblue', marker = 'o', markersize = 5, linestyle = '-')
    plt.plot(history.history['val_loss'], label = 'Perte de validation', color = 'salmon', marker = 'o', markersize = 5, linestyle = '-')
    plt.title('Courbe de Perte')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.xticks(range(0, 21, 5))
    plt.legend()
    plt.grid(False)
    
    # Tracer la courbe d'accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['precision'], label = 'Précision d\'entraînement', color = 'lightblue', marker = 'o', markersize = 5, linestyle = '-')
    plt.plot(history.history['val_precision'], label = 'Précision de validation', color = 'salmon', marker = 'o', markersize = 5, linestyle = '-')
    plt.title('Courbe de Précision')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.xticks(range(0, 21, 5))
    plt.legend()
    plt.grid(False)
   
    plt.show()

def plot_auc_f1(history):
    
    plt.figure(figsize=(14, 5))
    
    # Tracer la courbe AUC
    plt.subplot(1, 2, 1)
    plt.plot(history.history['auc'], label = "AUC moyen sur les données d'entraînement", color = 'lightblue', marker = 'o', markersize = 5, linestyle = '-')
    plt.plot(history.history['val_auc'], label = "AUC moyen sur les données de validation", color = 'salmon', marker = 'o', markersize = 5, linestyle = '-')
    plt.title('Courbe de AUC-ROC')
    plt.xlabel('Époque')
    plt.ylabel('Area Under Curve')
    plt.xticks(range(0, 21, 5))
    plt.legend()
    plt.grid(False)
    
    # Tracer la courbe F1 Score
    plt.subplot(1, 2, 2)
    plt.plot(np.mean(history.history['f1_score'], axis = 1), label = "F1 Score moyen sur les données d'entrainement", color = 'lightblue', marker = 'o', markersize = 5, linestyle = '-')
    plt.plot(np.mean(history.history['val_f1_score'], axis = 1), label = "F1 Score moyen sur les données de validation", color = 'salmon', marker = 'o', markersize = 5, linestyle = '-')
    plt.title('Courbe de F1 Score')
    plt.xlabel('Époque')
    plt.ylabel('F1 Score')
    plt.xticks(range(0, 21, 5))
    plt.legend()
    plt.grid(False)
   
    plt.show()

plot_auc_f1(history)
plot_lc(history)