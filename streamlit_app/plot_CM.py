import streamlit as st
import numpy as np
import plotly.figure_factory as ff

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

