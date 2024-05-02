import streamlit as st
import uuid
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objects as go
import plotly.figure_factory as ff
import tensorflow as tf
import keras

from keras.layers import Conv2D
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from keras.models import load_model

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

# Fonctions de plot des métriques
def plot_loss_curve(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = list(range(len(history['loss']))),
                             y = history['loss'],
                             mode = 'lines+markers',
                             name = 'Perte d\'entraînement',
                             marker = dict(color = 'lightblue')))

    fig.add_trace(go.Scatter(x = list(range(len(history['val_loss']))),
                             y = history['val_loss'],
                             mode = 'lines+markers',
                             name = 'Perte de validation',
                             marker = dict(color = 'salmon')))

    fig.update_layout(title = dict(text = "Courbe de Perte", font = dict(color = 'white')),
                      xaxis_title = dict(text = "Époque", font = dict(color = 'white')),
                      yaxis_title = dict(text = "Perte", font = dict(color = 'white')),
                      template = 'plotly_white',
                      paper_bgcolor = 'rgba(0,0,0,0)',
                      plot_bgcolor = 'rgba(0,0,0,0)',
                      legend = dict(font = dict(color = 'white')))
    st.plotly_chart(fig)

def plot_precision_curve(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = list(range(len(history['precision']))),
                             y = history['precision'],
                             mode = 'lines+markers',
                             name = "Précision d'entraînement",
                             marker = dict(color = 'lightblue')))

    fig.add_trace(go.Scatter(x = list(range(len(history['val_precision']))),
                             y = history['val_precision'],
                             mode = 'lines+markers',
                             name = 'Précision de validation',
                             marker = dict(color = 'salmon')))

    fig.update_layout(title = dict(text = "Courbe de Précision", font = dict(color = 'white')),
                      xaxis_title=dict(text = "Époque", font = dict(color = 'white')),
                      yaxis_title=dict(text = "Précision", font = dict(color = 'white')),
                      template = 'plotly_white',
                      paper_bgcolor = 'rgba(0,0,0,0)',
                      plot_bgcolor = 'rgba(0,0,0,0)',
                      legend = dict(font = dict(color='white')))
    st.plotly_chart(fig)

def plot_auc(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = list(range(len(history['auc']))),
                              y = history['auc'],
                              mode = 'lines+markers',
                              name = "AUC moyen d'entraînement",
                              marker = dict(color='lightblue')))

    fig.add_trace(go.Scatter(x = list(range(len(history['val_auc']))),
                              y = history['val_auc'],
                              mode = 'lines+markers',
                              name = 'AUC moyen de validation',
                              marker=dict(color = 'salmon')))

    fig.update_layout(title = "Courbe de AUC-ROC",
                      xaxis_title = "Époque",
                      yaxis_title = "Area Under Curve",
                      template = 'plotly_white',
                      paper_bgcolor = 'rgba(0,0,0,0)',
                      plot_bgcolor = 'rgba(0,0,0,0)',
                      legend = dict(font = dict(color = 'white')),
                      xaxis = dict(tickfont = dict(color = 'white')),
                      yaxis = dict(tickfont = dict(color = 'white')),
                      title_font = dict(color = 'white'))
    st.plotly_chart(fig)

def plot_f1_score(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = list(range(len(history['f1_score']))),
                              y=np.mean(history['f1_score'], axis = 1),
                              mode = 'lines+markers',
                              name = "F1 Score d'entraînement",
                              marker = dict(color = 'lightblue')))

    fig.add_trace(go.Scatter(x = list(range(len(history['val_f1_score']))),
                              y = np.mean(history['val_f1_score'], axis = 1),
                              mode = 'lines+markers',
                              name = 'F1 Score moyen de validation',
                              marker = dict(color = 'salmon')))

    fig.update_layout(title = "Courbe de F1 Score",
                      xaxis_title = "Époque",
                      yaxis_title = "F1 Score",
                      template = 'plotly_white',
                      paper_bgcolor = 'rgba(0,0,0,0)',
                      plot_bgcolor = 'rgba(0,0,0,0)',
                      legend = dict(font = dict(color = 'white')),
                      xaxis = dict(tickfont = dict(color = 'white')),
                      yaxis = dict(tickfont = dict(color = 'white')),
                      title_font = dict(color = 'white'))
    st.plotly_chart(fig)

# Plot des matrices de confusion
def plot_CM(matrix):
    confusion_matrix = np.array(matrix)
    class_names = ['Covid', 'Lung opacity', 'Normal', 'Viral Pneumonia']
    st.title('Matrice de Confusion')
    fig = ff.create_annotated_heatmap(z = confusion_matrix, x = class_names, y = class_names, colorscale = 'RdBu')
    fig.update_layout(
        title = 'Matrice de Confusion',
        xaxis = dict(title = 'Classe Prédite'),
        yaxis = dict(title = 'Classe Réelle')
    )
    st.plotly_chart(fig)


def plot_CM_ResNetV2():
    confusion_lines = [
        [192, 3, 7, 2],
        [9, 148, 24, 0],
        [5, 17, 139, 4],
        [1, 0, 4, 165]
    ]

    confusion_matrix = np.array(confusion_lines)
    class_names = ['Covid', 'Lung opacity', 'Normal', 'Viral Pneumonia']
    st.title('Matrice de Confusion')
    fig = ff.create_annotated_heatmap(z=confusion_matrix, x=class_names, y=class_names, colorscale='Cividis')
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis=dict(title='Classe Prédite'),
        yaxis=dict(title='Classe Réelle')
    )
    st.plotly_chart(fig)

def plot_CM_ResNet121():
    confusion_lines = [
        [181, 11, 11, 1],
        [8, 148, 25, 0],
        [10, 17, 135, 3],
        [2, 0, 1, 167]
    ]

    confusion_matrix = np.array(confusion_lines)
    class_names = ['Covid', 'Lung opacity', 'Normal', 'Viral Pneumonia']
    st.title('Matrice de Confusion')
    fig = ff.create_annotated_heatmap(z=confusion_matrix, x=class_names, y=class_names, colorscale='Cividis')
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis=dict(title='Classe Prédite'),
        yaxis=dict(title='Classe Réelle')
    )
    st.plotly_chart(fig)

def plot_CM_DenseNet201():
    confusion_lines = [
        [190, 5, 7, 2],
        [6, 156, 19, 0],
        [4, 21, 139, 1],
        [1, 0, 0, 169]
    ]

    confusion_matrix = np.array(confusion_lines)
    class_names = ['Covid', 'Lung opacity', 'Normal', 'Viral Pneumonia']
    st.title('Matrice de Confusion')
    fig = ff.create_annotated_heatmap(z=confusion_matrix, x=class_names, y=class_names, colorscale='Cividis')
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis=dict(title='Classe Prédite'),
        yaxis=dict(title='Classe Réelle')
    )
    st.plotly_chart(fig)

def plot_CM_VGG16():

    confusion_lines = [
        [178, 5, 9, 2],
        [4, 152, 17, 0],
        [2, 11, 160, 3],
        [0, 0, 4, 175]
    ]

    confusion_matrix = np.array(confusion_lines)
    class_names = ['Covid', 'Lung opacity', 'Normal', 'Viral Pneumonia']
    st.title('Matrice de Confusion')
    fig = ff.create_annotated_heatmap(z=confusion_matrix, x=class_names, y=class_names, colorscale='Cividis')
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis=dict(title='Classe Prédite'),
        yaxis=dict(title='Classe Réelle')
    )
    st.plotly_chart(fig)

def plot_CM_VGG19():
    confusion_lines = [
        [182, 7, 3, 0],
        [7, 158, 8, 0],
        [8, 21, 142, 5],
        [1, 1, 3, 174]
    ]

    confusion_matrix = np.array(confusion_lines)
    class_names = ['Covid', 'Lung opacity', 'Normal', 'Viral Pneumonia']
    st.title('Matrice de Confusion')
    fig = ff.create_annotated_heatmap(z=confusion_matrix, x=class_names, y=class_names, colorscale='Cividis')
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis=dict(title='Classe Prédite'),
        yaxis=dict(title='Classe Réelle')
    )
    st.plotly_chart(fig)

def plot_CM_ConvnextTiny():
    confusion_lines = [
        [122, 11, 19, 0],
        [13, 142, 15, 0],
        [17, 14, 144, 1],
        [4, 1, 7, 168]
    ]

    confusion_matrix = np.array(confusion_lines)
    class_names = ['Covid', 'Lung opacity', 'Normal', 'Viral Pneumonia']
    st.title('Matrice de Confusion')
    fig = ff.create_annotated_heatmap(z=confusion_matrix, x=class_names, y=class_names, colorscale='Cividis')
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis=dict(title='Classe Prédite'),
        yaxis=dict(title='Classe Réelle')
    )
    st.plotly_chart(fig)

def plot_CM_ConvnextBase():
    confusion_lines = [
        [168, 9, 15, 0],
        [9, 152, 12, 0],
        [12, 8, 153, 3],
        [2, 0, 8, 169]
    ]

    confusion_matrix = np.array(confusion_lines)
    class_names = ['Covid', 'Lung opacity', 'Normal', 'Viral Pneumonia']
    st.title('Matrice de Confusion')
    fig = ff.create_annotated_heatmap(z=confusion_matrix, x=class_names, y=class_names, colorscale='Cividis')
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis=dict(title='Classe Prédite'),
        yaxis=dict(title='Classe Réelle')
    )
    st.plotly_chart(fig)

def plot_CM_EfficientNet_B4():
    confusion_lines = [
        [177, 17, 10, 0],
        [3, 148, 30, 0],
        [1, 13, 151, 0],
        [2, 1, 9, 158]
    ]

    confusion_matrix = np.array(confusion_lines)
    class_names = ['Covid', 'Lung opacity', 'Normal', 'Viral Pneumonia']
    st.title('Matrice de Confusion')
    fig = ff.create_annotated_heatmap(z=confusion_matrix, x=class_names, y=class_names, colorscale='Cividis')
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis=dict(title='Classe Prédite'),
        yaxis=dict(title='Classe Réelle')
    )
    st.plotly_chart(fig)

def plot_CM_VGG16_FT():
    confusion_lines = [
        [229, 7, 6, 0],
        [1, 198, 16, 1],
        [7, 6, 199, 4],
        [0, 0, 1, 225]
    ]

    confusion_matrix = np.array(confusion_lines)
    class_names = ['Covid', 'Lung opacity', 'Normal', 'Viral Pneumonia']
    st.title('Matrice de Confusion')
    fig = ff.create_annotated_heatmap(z=confusion_matrix, x=class_names, y=class_names, colorscale='Cividis')
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis=dict(title='Classe Prédite'),
        yaxis=dict(title='Classe Réelle')
    )
    st.plotly_chart(fig)

def plot_CM_ResNetFT():

    confusion_lines = [
        [278, 6, 3, 1],
        [3, 242, 16, 0],
        [7, 38, 197, 17],
        [2, 1, 0, 265]
    ]

    confusion_matrix = np.array(confusion_lines)
    class_names = ['Covid', 'Lung opacity', 'Normal', 'Viral Pneumonia']
    st.title('Matrice de Confusion')
    fig = ff.create_annotated_heatmap(z=confusion_matrix, x=class_names, y=class_names, colorscale='Cividis')
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis=dict(title='Classe Prédite'),
        yaxis=dict(title='Classe Réelle')
    )
    st.plotly_chart(fig)

def plot_CM_DenseNetFT():
    confusion_lines = [
        [285, 1, 2, 0],
        [3,235 , 23, 0],
        [5, 17, 232, 5],
        [0, 0, 3, 265]
    ]

    confusion_matrix = np.array(confusion_lines)
    class_names = ['Covid', 'Lung opacity', 'Normal', 'Viral Pneumonia']
    st.title('Matrice de Confusion')
    fig = ff.create_annotated_heatmap(z=confusion_matrix, x=class_names, y=class_names, colorscale='Cividis')
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis=dict(title='Classe Prédite'),
        yaxis=dict(title='Classe Réelle')
    )
    st.plotly_chart(fig)

def plot_CM_ENetB4():
    confusion_lines = [
        [282, 2, 3, 1],
        [4, 244, 13, 0],
        [2, 22, 220, 15],
        [1, 0, 0, 267]
    ]

    confusion_matrix = np.array(confusion_lines)
    class_names = ['Covid', 'Lung opacity', 'Normal', 'Viral Pneumonia']
    st.title('Matrice de Confusion')
    fig = ff.create_annotated_heatmap(z=confusion_matrix, x=class_names, y=class_names, colorscale='Cividis')
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis=dict(title='Classe Prédite'),
        yaxis=dict(title='Classe Réelle')
    )
    st.plotly_chart(fig)

# GRAD-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    model_output = model.output if isinstance(model.output, list) else [model.output]
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output] + model_output
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)

    jet = plt.cm.jet

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    return superimposed_img
