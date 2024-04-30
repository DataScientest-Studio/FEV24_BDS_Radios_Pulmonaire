import streamlit as st
import uuid
import re
import numpy as np
import matplotlib.image as mpimg

# Afficher le nom et l'icone des réseaux d'un membre
def show_profile(name, linkedin_url, github_url):
    st.markdown("""
    <style>
    .icon-img {
        height: 20px;  # Taille des icônes
        width: 20px;
        margin-top: 0px;
        margin-left: 5px;
        margin-right:5px;
    }
    .name-with-icons {
        display: flex-shrink;
        align-items: center; 
        margin-top: -5px;
        margin-bottom: 0px;
    }
    .name-with-icons:first-child {
        margin-top: -5px;  /* Marge négative pour le premier élément */
    } 
    </style>
    """, unsafe_allow_html = True)

    st.markdown(f"""
    <div class='name-with-icons'>
        <a href='{github_url}' target='_blank'><img class='icon-img' src='https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg'></a>
        <a href='{linkedin_url}' target='_blank'><img class='icon-img' src='https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png'></a>
        {name}
    </div>
    """, unsafe_allow_html = True)

# Création de bloc de texte esthétiques
def create_styled_box(text, text_color, background_color, alignment = 'left', display = 'block'):
    unique_id = uuid.uuid4().hex
    # Ajout de styles CSS pour le cadre avec une couleur de fond et une couleur de texte personnalisée
    st.markdown(f"""
    <style>
    .styled-box-{unique_id} {{
        background-color: {background_color};   /* couleur de fond */
        color: {text_color};                    /* couleur de texte */
        padding: 10px 20px;                     /* espace intérieur vertical et horizontal */
        border: 1px solid {text_color};         /* bordure de la même couleur que le texte */
        border-radius: 8px;                     /* bord arrondi */
        font-size: 16px;                        /* taille de la police */
        font-weight: regular;                   /* format de la police */
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);  /* ombre pour donner de la profondeur */
        text-align: {alignment};                /* alignement du texte */
        display: {display};
    }}
    </style>
    """, unsafe_allow_html = True)

    # Afficher le texte dans le cadre stylisé
    st.markdown(f"""
    <div class="styled-box-{unique_id}">{text}</div>
    """, unsafe_allow_html = True)

# Fonction pour calculer l'intensité lumineuse moyenne d'une image
def calc_mean_intensity(image_path):
    img = mpimg.imread(image_path)
    # Convertir en nuances de gris si l'image est en couleur
    if img.ndim == 3:
        img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        img_gray = img
    return np.mean(img_gray)

# Fonction pour extraire les sources depuis les urls
def source_extract(url):
    pattern = re.compile(r'https?://(?:www\.)?([^/]+)')
    match = pattern.search(url)
    if match:
        return match.group(1)
    else:
        return None