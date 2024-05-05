# Deep learning : Classification de radiographies pulmonaires COVID-19

## Présentation

Ce répertoire contient le code de notre projet de classification de radiographies pulmonaires développé pendant notre [formation de Data Scientist](https://datascientest.com/en/data-scientist-course) avec [DataScientest](https://datascientest.com/) sous `Python 3.11.9`.

L’objectif principal de ce projet est de développer un système d’analyse automatisée de radiographies pulmonaires pour la détection efficace du COVID-19 sur moins de deux mois en parallèle de notre formation. 
En utilisant les techniques de Deep Learning, nous cherchons à créer un modèle de classification capable de distinguer avec précision les radiographies de patients atteints de COVID-19 de celles présentant d’autres affections pulmonaires ou étant saines. Les objectifs spécifiques incluent l’acquisition et la préparation d’un ensemble de données diversifié et annoté, la conception et l'entraînement d’un modèle de Deep Learning robuste, ainsi que l’évaluation rigoureuse de ses performances en termes de sensibilité, de spécificité et de précision.

Ce projet a été développé par l'équipe suivante :
- Alexandre LANGLAIS ([GitHub](https://github.com/a-langlais/) / [LinkedIn](http://www.linkedin.com/in/alexlanglais/))
- Chaouki BENZEKRI ([GitHub](https://github.com/ChaoukiBenzekri/) / [LinkedIn](https://www.linkedin.com/in/chaouki-benzekri-3b0b57136/))
- Camille RUBI ([GitHub](https://github.com/Rubicamille ) / [LinkedIn](https://www.linkedin.com/in/camille-rubi/))
- Pierre-Jean CORNEJO ([GitHub](https://github.com/PJCornejo) / [LinkedIn](https://www.linkedin.com/in/pierre-jean-cornejo-74a3b0b6/))

Sous le mentorat de Gaël PENESSOT ([GitHub](https://github.com/gpenessot/) / [LinkedIn](http://www.linkedin.com/in/gael-penessot/)).

Le rapport complet de notre projet est disponible dans le dossier ([report](./report)).

Voici le résumé :

> La COVID-19, causée par le coronavirus SARS-CoV-2, peut provoquer des anomalies sévères chez certains patients caractérisée par des lésions pulmonaires spécifiques visibles sur les radiographies sous un effet de “verre dépoli”. Ce projet porte sur la conception d’un système basé sur l’intelligence artificielle, plus spécifiquement grâce à un modèle de réseau de neurones, permettant de détecter la COVID-19 à partir d’une radiographie pulmonaire. Pour ce faire, nous avons à notre disposition un jeu de 21165 radiographies et de masques provenant de huit sources différentes. 
Nos analyses nous ont permis de mettre en évidence des différences entre les métadonnées fournies et celles des radiographies que nous avons pu prendre en compte lors de notre étape de preprocessing, visant à redimensionner, homogénéiser le nombre de canaux et normaliser les images pour qu’elles soient exploitables dans nos modèles. Ensuite, nous avons pu exploiter l’architecture LeNet-5 afin de tester nos données, l’influence des masques et réaliser plusieurs benchmarks afin de nous donner des pistes sur la manière d’appréhender notre problématique. Nous avons utilisé le transfer learning pour tester 15 modèles aux stratégies d’architectures différentes dans le but d’avoir des performances de base élevées selon un protocole de test standardisé. Suite à ces expérimentations itératives et progressives, nous avons sélectionné les quatre modèles avec les meilleurs résultats dans le but de réaliser des ajustements fins : l’EfficientNetB4, le VGG16, le DenseNet201 et le ResNet50. 
L’étape de finetuning a été caractérisée par le dégèlement de couches en profondeur des modèles pré entraînés, l’optimisation des hyperparamètres par keras-tuner et l’ajout de couches supplémentaires pour ajuster les modèles au plus proche de notre problématique initiale. Grâce à ces étapes d’optimisation, nous avons pu atteindre une précision de validation et un F1 Score de 0.96 pour la classe ‘COVID’ et 0.91 pour la classe ‘Normal’ avec le modèle VGG16 finement ajusté et optimisé. Ce modèle à l’avantage d’être à la fois performant sur de nouvelles images et suffisamment léger pour être un parfait compromis entre performance et souplesse d'entraînement.

## Organisation du répertoire

Ce répertoire suit une hiérarchie classique et facile à explorer.
Les modèles étant trop volumineux pour ce repo GitHub, ils sont disponibles sur [🤗 HuggingFace](https://huggingface.co/spaces/a-langlais/streamlit_chest_radiography_DL/tree/main/models).

```
FEV24_BDS_Radios_Pulmonaire/
│
├── datasets/               # Dossiers pour les tableaux de données créés
│
├── docs/                   # Documentation du projet (bibliographie)
│
├── models/                 # Historiques d'entrainement des deux modèles finaux
│
├── notebooks/              # Jupyter notebooks
│
├── outputs/                # Toutes les sorties graphiques du projet
│
├── reports/                # Rapport du projet
│
├── src/                    # Code source du projet
│   ├── data/               # Scripts pour télécharger ou générer des données
│   ├── models/             # Modèles de deep learning (architecture, entraînement, évaluation)
│   ├── utils/              # Fonctions utilitaires (preprocessing, calculs métriques)
│   └── visualization/      # Scripts pour visualiser les données
│
├── streamlit_app/          # Dossier de l'application streamlit
│
├── environment.yml         # Fichier pour recréer l'environnement (conda)
├── requirements.txt        # Dépendances Python (pip)
│
├── .gitignore              # Spécifie les fichiers intentionnellement non suivis
├── LICENSE                 # Licence du projet
└── README.md               # Lisez-moi
```

## Installation

Dans un premier temps, clonez le dépôt sur votre machine locale via votre méthode préférée ou en utilisant la commande suivante :

```shell
git clone https://github.com/DataScientest-Studio/FEV24_BDS_Radios_Pulmonaire.git
```

Ensuite, deux solutions s'offrent à vous si vous souhaitez relancer ce projet dans les mêmes conditions que lorsqu'il a été conçu.

Vous pouvez créer un environnement virtuel en téléchargeant spécifiquement les dépéndances Python nécessaires via le fichier `requirements.txt`.

```shell
pip install -r requirements.txt
```

Si vous utilisez Conda, vous pouvez tout simplement recréer un environnement en utiliser le fichier `environment.yml`.

```shell
conda env create -f environment.yml
conda activate environment
```

## Streamlit App

Le Streamlit est constitué d'un menu horizontal où les différentes étapes du projet sont résumées.
Dnas la partie "Utilisation du modèle", il est possible d'utiliser l'application web pour intégrer une image et recevoir la prédiction en temps réel, accompagnée de la GRADCAM associée.

Pour lancer l'application streamlit :

```shell
conda create --name chest_radiography_streamlit python=3.11.9
conda activate chest_radiography_streamlit
pip install -r streamlit_requirements.txt
streamlit run app.py
```

L'application devrait être disponible à [localhost:8501](http://localhost:8501).
