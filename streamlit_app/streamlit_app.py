# Importation des modules
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from PIL import Image
from custom_functions import create_styled_box, show_profile
import tensorflow as tf
#from tensorflow.keras.models import load_model

# Importation des pages
import presentation
import exploration
import fine_tuning
import own_test

st.set_page_config(page_title = "Classification de radiographies pulmonaires", layout = "wide", page_icon = "⚕️")

# HEADER
image = Image.open('images/banniere.jpg')
st.image(image, use_column_width = True, width = image.width)

st.header("", divider = 'rainbow')
st.markdown("<h1 style='text-align: center; color: white;'>Classification de radiographies pulmonaires par Deep Learning</h1>", unsafe_allow_html = True)
st.header("", divider = 'rainbow')

# Menu de navigation
selected = option_menu(
    menu_title = None,
    options = ["Présentation", "Exploration", "Modélisation", "Utiliser le modèle"],
    icons = ["easel3-fill", "eye-fill", "wrench-adjustable", "lightbulb-fill"],
    menu_icon = "cast",
    default_index = 0,
    orientation = "horizontal",
)

# Lancement du contenu des pages
if selected == "Présentation":
    presentation.show_presentation()
elif selected == "Exploration":
    exploration.show_exploration()
elif selected == "Modélisation":
    fine_tuning.show_fine_tuning()
elif selected == "Utiliser le modèle":
    own_test.show_test()

st.header("", divider = 'gray')
# create_styled_box("texte", text_color = '#C9BBCF', background_color = '#957DAD')
# create_styled_box("texte2", text_color = '#AC7D88', background_color = '#85586F')

st.markdown("""
    <style>
        .reduce-line-spacing {
            margin-bottom: 10px; /* Réduire l'espace entre les lignes */
        }
        .vertical-divider {
            border-left: 1px solid #ccc; /* Ajouter un diviseur vertical gris */
            height: 100%;
        }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([0.7, 0.15, 0.15])

with col1:
    show_profile(name = "Alexandre LANGLAIS", linkedin_url = "https://www.linkedin.com/in/alexlanglais/", github_url = "https://github.com/a-langlais")
    show_profile(name = "Chaouki BENZEKRI", github_url = "https://github.com/ChaoukiBenzekri/", linkedin_url = "https://www.linkedin.com/in/chaouki-benzekri-3b0b57136/")
    show_profile(name = "Camille RUBI", github_url = "https://github.com/Rubicamille", linkedin_url = "https://www.linkedin.com/in/camille-rubi/")
    show_profile(name = "Pierre-Jean CORNEJO", github_url = "https://github.com/PJCornejo", linkedin_url = "http://www.linkedin.com/in/pjCornejo/")

with col2:
    logo_dst = Image.open('images/logo_dst.jpg')
    st.image(logo_dst, width = 200, use_column_width = False)

with col3:
    logo_plm = Image.open('images/logo_esl_plm.jpg')
    st.image(logo_plm, width = 200, use_column_width = False)

