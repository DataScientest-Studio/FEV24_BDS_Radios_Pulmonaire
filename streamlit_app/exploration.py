import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image
from custom_functions import calc_mean_intensity, source_extract

df_images = pd.read_csv('data\df_images.csv')
df_masks = pd.read_csv('data\df_masks.csv')
df_combined = pd.read_csv('data\df_combined.csv')

def show_exploration():
    # Style des onglets
    st.markdown("""
        <style>
            .stTabs [data-baseweb="tab-list"] {
                display: flex;
                gap: 10px;
            }

            .stTabs [data-baseweb="tab"] {
                padding: 10px 15px;
                border: 1px solid transparent;
                border-radius: 5px 5px 0 0;
                background-color: transparent;
                cursor: pointer;
                transition: all 0.3s ease;
            }

            .stTabs [data-baseweb="tab"]:hover {
                background-color: #8f8d9b;
            }

            .stTabs [aria-selected="true"] {
                background-color:  #57546a;
                border-color: #ccc;
                border-bottom-color: transparent;
            }
        </style>""", unsafe_allow_html = True)

    tab1, tab2 = st.tabs(["🗂️ Métadonnées", "🖼️ Images & masques"])

    ### Premier onglet
    with tab1:
        st.header("Exploration des métadonnées")
        st.markdown("Nous avons à notre disposition un important jeu de données provenant de [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). Il s'agit de 21 165 images de radiographies pulmonaires labellisées dans leur nom, ainsi qu'autant de masques basiques associés aux images. De plus, quatre fichiers de métadonnées au format *.xlxs sont disponibles en accompagnement des quatre catégories.")
        
        st.dataframe(df_images.head())

        ## Valeurs uniques
        unique_classes = df_images['LABEL'].unique()
        unique_classes = ", ".join(map(str, unique_classes))
        unique_sources = df_images['SOURCE'].dropna().unique()
        unique_sources = ", ".join(map(str, unique_sources))
        unique_format = df_images["FORMAT"].unique()
        unique_format = ", ".join(map(str, unique_format))    
        unique_resolution = df_images["SIZE"].unique()
        unique_resolution = ", ".join(map(str, unique_resolution))
        unique_channel = df_images["CHANNELS"].unique()
        unique_channel = ", ".join(map(str, unique_channel))

        data = [
            ("Liste des résolutions", unique_resolution),
            ("Liste des formats", unique_format),
            ("Liste des classes", unique_classes),
            ("Liste des sources", unique_sources),
            ("Nombre de canaux", unique_channel),
            ("Nombre d'images", len(df_images))
        ]

        # Créer le DataFrame à partir de la liste de tuples
        df = pd.DataFrame(data, columns = ["Titre", "Variable"])

        # Convertir le dataframe en HTML avec les styles CSS
        html_table = df.to_html(index = False, header = False, justify = 'center', classes = 'styled-table', border = 0)

        # Afficher le HTML dans Streamlit avec la largeur calculée
        st.markdown(f"<div style='border: 1px solid white; border-radius: 5px; padding: 10px; background-color: #343434; line-height: 1; width: 616px; margin: 0 auto;'>{html_table}</div>", unsafe_allow_html=True)
        st.header("", divider = 'gray') 

        # ==================================================================================================
        # ==================================================================================================

        ## Countplot
        # Preprocess des données pour affichage
        total_images = df_images.groupby('LABEL').size().reset_index(name='Total Images')
        max_values = df_images.groupby('LABEL')['LABEL'].count().reset_index(name='Max Images')
        df_merged = pd.merge(total_images, max_values, on='LABEL')
        df_sorted = df_merged.sort_values('Total Images', ascending = False)

        palette = {'Normal': '#A1C9F4', 'Lung_Opacity': '#8DE5A1', 'COVID': '#FFB482', 'Viral Pneumonia': '#D0BBFF'}

        # Création du graphique Plotly avec affichage des valeurs maximales uniquement au survol
        fig = px.bar(df_sorted, x = 'LABEL', y = 'Total Images', color = 'LABEL',
                    color_discrete_map = palette,
                    title = "Distribution du nombre d'images par LABEL",
                    labels = {'LABEL': 'Label', 'Total Images': 'Nombre d\'images', 'Max Images': 'Nombre maximal d\'images'},
                    hover_data = {'LABEL': False, 'Max Images': True, 'Total Images': False}
                    )

        fig.update_layout(
            xaxis_title = 'Label',
            yaxis_title = "Nombre d'images",
            showlegend = True,
            bargap = 0.2,  # Espace entre les barres
            width = 600,
            height = 500
        )

        st.plotly_chart(fig)
        st.markdown("Les classes sont particulièrement déséquilibrées, il est cependant intéressant de constater que la classe minoritaire contient tout de même 1345 images.")
        st.header("", divider = 'gray')
        
        # ==================================================================================================
        # ==================================================================================================

        ## Pieplot
        # Supposons que df_images est votre DataFrame et 'SOURCE' est la colonne d'intérêt
        source_counts = df_images['SOURCE'].value_counts()

        # Création d'un graphique pie avec Plotly
        fig = px.pie(
            values = source_counts.values, 
            names = source_counts.index, 
            title = 'Répartition des sources des images',
            hole = 0.5, 
            color_discrete_sequence = px.colors.qualitative.Pastel
        )

        fig.update_traces(
            textinfo = 'label+percent',
            marker = dict(line = dict(color = 'black', width = 1)),
        )

        fig.update_layout(
            showlegend = True,
            width = 700,
            height = 700
        )

        # Affichage du graphique avec Streamlit
        st.plotly_chart(fig)

    ### Deuxième onglet
    with tab2:
        st.header("Exploration des images")

        ## Affichage d'une image aléatoire pour chaque catégorie
        dossier_radio = "radios/"
        sous_dossiers = [d for d in os.listdir(dossier_radio) if os.path.isdir(os.path.join(dossier_radio, d))]

        st.info("Cliquez sur le boutton ci-dessous pour afficher un échantillon d'images", icon="ℹ️")
        if st.button("Afficher une image aléatoire de chaque LABEL"):
            cols = st.columns(len(sous_dossiers))
            for i, sous_dossier in enumerate(sous_dossiers):
                    sous_dossier_path = os.path.join(dossier_radio, sous_dossier)
                    
                    fichiers_images = [f for f in os.listdir(sous_dossier_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    
                    if fichiers_images:
                        image_selectionnee = random.choice(fichiers_images)
                        image_path = os.path.join(sous_dossier_path, image_selectionnee)
                        
                        with open(image_path, "rb") as f:
                            image = Image.open(f)
                            cols[i].image(image, caption=f"{sous_dossier}", use_column_width = "auto", width = 299)
            st.success('Images affichées avec succès !', icon = "✅")
            st.markdown("Nous pouvons remarquer que les images sont toutes en nuances de gris, malgré leur nombre de canaux quelques fois différent. De plus, toutes les radiographies semblent avoir été prises selon une méthode standard, mettant bien les poumons au centre de l'image. Quelques variations peuvent cependant apparaître (bras vers le haut, artefacts visuels, annotations, etc.)")
        st.header("", divider = 'gray')
        
        # ==================================================================================================
        # ==================================================================================================

        col1, col2 = st.columns([0.6, 0.4])

        with col1:
            ## Violintplot
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

            # Création du graphique Plotly
            fig = px.violin(df_combined, x = 'Label_Masque', y = 'COMBINED_INTENSITY',
                            violinmode = 'overlay',
                            color = 'Label_Masque',
                            color_discrete_map = palette,
                            category_orders = {'Label_Masque': new_labels},
                            title = "Distribution de l'intensité lumineuse moyenne normalisée",
                            labels = {'Label_Masque': ''},
                            )

            # Mise en forme du graphique
            fig.update_layout(
                yaxis_title = 'Intensité lumineuse moyenne',
                xaxis_title = '',
                width = 1000,
                height = 600
            )

            # Affichage du graphique
            st.plotly_chart(fig)

        with col2:
            st.markdown("L'application des masques réduit considérablement l'intensité lumineuse moyenne des images. Ce comportement est tout à fait normal car les masques noircient les parties non pertinentes et font ainsi tendre la moyenne des pixels vers 0.")
        
        st.header("", divider = 'gray')
        
        # ==================================================================================================
        # ==================================================================================================

        ## Histogramme de la fréquence de l'intensité des pixels
        palette_list = ['#A1C9F4','#8DE5A1','#FFB482', '#D0BBFF']

        st.info("Cliquez sur le bouton ci-dessous pour afficher un échantillon d'images et leur histogramme", icon="ℹ️")
        if st.button("Afficher la fréquence de l'intensité des pixels"):
            # Créer une ligne avec quatre colonnes
            cols = st.columns(4)
            for idx, sous_dossier in enumerate(sous_dossiers):
                sous_dossier_path = os.path.join(dossier_radio, sous_dossier)
                fichiers_images = [f for f in os.listdir(sous_dossier_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                
                if fichiers_images:
                    image_selectionnee = random.choice(fichiers_images)
                    image_path = os.path.join(sous_dossier_path, image_selectionnee)
                    
                    with open(image_path, "rb") as f:
                        image = Image.open(f)
                        # Afficher l'image avec sa véritable résolution sans agrandissement
                        cols[idx].image(image, caption=f"{sous_dossier}", width = image.width, use_column_width = "auto")
                        
                        image_array = np.array(image)
                        if len(image_array.shape) == 3:
                            image_array = image_array.mean(axis=2)

                        filtered_array = image_array[image_array > 0]
                        hist_values, bin_edges = np.histogram(filtered_array, bins=255, range=(0, 256))
                        
                        # Création de l'histogramme avec Plotly
                        fig = px.bar(
                            x=bin_edges[:-1],
                            y=hist_values,
                            labels={'x': 'Intensité des pixels', 'y': 'Nombre de pixels'},
                            color_discrete_sequence=[palette_list[idx]]
                        )
                        cols[idx].plotly_chart(fig, use_container_width=True)
            st.success('Histogrammes générés avec succès !', icon="✅")
        st.header("", divider='gray')
        
        # ==================================================================================================
        # ==================================================================================================

        col1, col2 = st.columns([0.3, 0.6])

        with col1:
            st.markdown("test de texte")

        with col2:
            ## Surface utile
            # Création de l'histogramme avec Plotly Express
            fig2 = px.histogram(df_masks, x = 'RATIO', color = 'LABEL',
                            nbins = 70,
                            barmode = 'overlay',  # Superpose les distributions
                            color_discrete_sequence = ['#A1C9F4', '#8DE5A1', '#FFB482', '#D0BBFF'],  # Palette de couleurs
                            opacity = 0.75)

            # Personnalisation supplémentaire
            fig2.update_traces(marker_line_color = 'gray', marker_line_width = 1.5)  # Ajouter la bordure de barre
            fig2.update_layout(
                title = 'Ratio de la surface utile en appliquant les masques',
                xaxis_title = 'Ratio',
                yaxis_title = 'Nombre de cas',
                legend_title = 'Label',
                height = 500,
                width = 800
            )

            # Affichage du graphique
            st.plotly_chart(fig2)

