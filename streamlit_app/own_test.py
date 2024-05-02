import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import time

model_densenet = tf.keras.models.load_model("models\DenseNet201_finetuned.h5")
model_vgg = tf.keras.models.load_model("models\VGG16_finetuned.h5")
model = None

def show_test():
    st.header("Réaliser des prédictions")

    # Configuration initiale de l'état de session
    if 'model_selected' not in st.session_state:
        st.session_state.model_selected = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = None

    col1, col2, col3 = st.columns([0.3, 0.4, 0.3])

    with col1:
        st.session_state.model_selected = st.selectbox(
            "Quel modèle utiliser ?",
            ("DenseNet201", "VGG16"),
            index=None,
            placeholder="Choisissez un modèle..."
        )
        if st.session_state.model_selected == "DenseNet201":
            st.write("Modèle le plus performant, mais aussi le plus lourd.")
            st.session_state.model_loaded = 'DenseNet201'
        elif st.session_state.model_selected == "VGG16":
            st.write("Modèle le plus équilibré.")
            st.session_state.model_loaded = 'VGG16'

    with col2:
        if st.session_state.model_selected == "DenseNet201" and st.session_state.model_loaded == 'DenseNet201':
            with st.spinner('Chargement du modèle DenseNet201...'):
                model = model_densenet
                st.success('👏 Modèle DenseNet201 chargé et prêt à prédire !')
        elif st.session_state.model_selected == "VGG16" and st.session_state.model_loaded == 'VGG16':
            with st.spinner('Chargement du modèle VGG16...'):
                model = model_vgg
                st.success('👏 Modèle VGG16 chargé et prêt à prédire !')

    with col3:
        if st.button('Réinitialiser le modèle', type = 'primary'):
            keys_to_delete = ['model_selected', 'model_loaded', 'file_uploaded']
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
                    st.experimental_rerun()

    st.header("", divider = 'gray')

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        file_container = st.empty()
        uploaded_file = file_container.file_uploader("", type=['png', 'jpg', 'jpeg'])

    if uploaded_file and st.session_state.model_selected is not None:
        with col2:
            st.write('')
            st.write('')
            original = Image.open(uploaded_file)
            my_bar = st.progress(0, text = "Calculs en cours...")
            for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1, text = "Calculs en cours...")
            time.sleep(1)
            my_bar.empty()
            st.success("👏 Prédictions réalisées avec succès !")       
    
    st.header("", divider = 'gray')

    if (uploaded_file is not None) and (st.session_state.model_selected is not None):
        # Ouvrir l'image téléchargée
        original = Image.open(uploaded_file)        

        # Image traitée
        gray_image = original.convert('L')
        channelized = gray_image.convert("RGB")
        resized = channelized.resize((224, 224))
        img_normalized = np.array(resized) / 255.0  # Convertir l'image en array et normaliser entre 0 et 1
        img_normalized -= np.array([0.485, 0.456, 0.406])  # Soustraction de la moyenne par canal
        img_normalized /= np.array([0.229, 0.224, 0.225])  # Division par l'écart-type par canal
        img_normalized = img_normalized.reshape(-1, 224, 224, 3)  # Remodeler pour correspondre aux attentes du modèle (batch_size, height, width, channels)

        col1, col2 = st.columns([0.3, 0.7])

        if original.width > 500:
            width = 500
        else:
            width = original.width

        with col1:
            st.header("Image originale")
            st.image(original, use_column_width = False, width = width, clamp = True)
            st.warning("Image redimensionnée pour des raisons d'affichage.", icon = "⚠️") if original.width > 500 else None
        with col2:
            st.header("Image traitée")
            st.image(img_normalized, use_column_width = False, clamp = True)
        
        st.header("", divider = 'gray')

        predictions = model.predict(img_normalized)

        class_names = {0 : 'COVID',
                       1 : 'Lung_Opacity',
                       2 : 'Normal',
                       3 : 'Viral Pneumonia'}
        df_predictions = pd.DataFrame(predictions)
        df_predictions = df_predictions.rename(columns = class_names)
        df_predictions_sorted = df_predictions.sort_values(by = 0, axis = 1, ascending = False)
        df_transposed = df_predictions_sorted.T
        table_html = df_transposed.to_html(header = False, index = True)

        classe_predite_indice = np.argmax(predictions)
        nom_classe_predite = class_names[classe_predite_indice]
        probabilite_predite = np.max(predictions)
        probabilite_predite = "{:.4f}".format(probabilite_predite)

        col1, col2 = st.columns([1, 1])

        with col1:
            # Afficher la classe prédite et sa probabilité
            st.markdown("Tableau des probabilités estimées :")
            st.write(table_html, unsafe_allow_html = True)
        
        with col2:
            st.markdown(f"**Classe prédite** : {nom_classe_predite}")
            st.markdown(f"**Probabilité** : {probabilite_predite}")
            if float(probabilite_predite) > 0.90:
                st.markdown(f"Avec une probabilité de {probabilite_predite}, il est **certain** que cette radiographie illustre un cas {nom_classe_predite}. Attention cependant, cette prédiction est à prendre en compte seulement si l'image s'agit bien d'une radiographie conforme.")
            elif float(probabilite_predite) > 0.75:
                st.markdown(f"Avec une probabilité de {probabilite_predite}, il est **probable** que cette radiographie illustre un cas {nom_classe_predite}. Attention cependant, cette prédiction est à prendre en compte seulement si l'image s'agit bien d'une radiographie conforme.")
            elif float(probabilite_predite) > 0.5:
                st.markdown(f"Avec une probabilité de {probabilite_predite}, il est **possible** que cette radiographie illustre un cas {nom_classe_predite}. Attention cependant, cette prédiction est à prendre en compte seulement si l'image s'agit bien d'une radiographie conforme.")
            elif float(probabilite_predite) <= 0.5:
                st.markdown(f"Avec une probabilité de {probabilite_predite}, il **n'est pas prudent de dire** que cette radiographie illustre un cas {nom_classe_predite}. Il est possible que l'image ne soit pas adaptée au modèle.")

    else:
        # Message affiché tant qu'aucune image n'est téléchargée
        st.warning("Veuillez télécharger une image pour commencer l'analyse.", icon = "⚠️")
