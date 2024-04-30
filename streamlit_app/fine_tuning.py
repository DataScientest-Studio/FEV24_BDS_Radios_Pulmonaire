import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

with open(r"C:\Users\Gamy\Documents\GitHub\FEV24_BDS_Radios_Pulmonaire\models\history_DenseNet201_finetuned.pkl", "rb") as file1:
    history_densenet = pickle.load(file1)
with open(r"C:\Users\Gamy\Documents\GitHub\FEV24_BDS_Radios_Pulmonaire\models\history_VGG16.pkl", "rb") as file2:
    history_vgg = pickle.load(file2)

def show_fine_tuning():

    tab1, tab2, tab3, tab4 = st.tabs(["🛠️ Preprocessing", "📈 Benchmarks", "💻 Modèles testés", "🤖 Modèle final"])
    
    st.markdown("""
    <style>
        .stTabs [data-baseweb = "tab-list"] {
            gap: 5px;
        }
        .stTabs [data-baseweb = "tab"] {
            height: 25px;
            white-space: pre-wrap;
            background-color: #626C66;
            border-radius: 4px 4px 0px 0px;
            border: 1px solid #fff;
            gap: 5px;
            padding-top: 10px;
            padding-bottom: 10px;
            padding-right: 5px;
        }
        .stTabs [aria-selected = "true"] {
            background-color: #F4FFFD;
            border : 1px solid #626C66;
        }
    </style>""", unsafe_allow_html = True)
    
    ### Premier onglet
    with tab1:
        st.header("Preprocessing des images")
        st.markdown('''
            Une étape très importante de notre projet est l'attention portée au traitement des images d'entrée. Nous avons pu voir précédemment que les images possèdent pour certaines, des dimensions et/ou un nombre de canaux différents. Il est important d'homogénéiser l'ensemble des paramètres de nos images pour assurer une bonne performance de nos modèles, et surtout, des résultats comparables. Les éléments en question sont :
            - Une dimension homogène et carrée, par défaut 256x256 pixels.
            - Un nombre de trois canaux de couleur.
            - Une normalisation de la valeur des pixels.\n
            Une fonction `preproc_img()` est conçue pour simplifier ces étapes, améliorer la reproductibilité et faciliter les ajustements. Elle retourne automatiquement les **ensembles d'entraînement et de test**.
        ''')

        # Style CSS pour listes à puces internes
        st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ul{
            list-style-position: inside;
        }
        </style>
        ''', unsafe_allow_html = True)

        # Définir le code comme une chaîne de caractères longue
        code = """
                def preproc_img(df_images, df_masks, n_img, normalize, files_path, resolution, with_masks):
                    np.random.seed(42)
                    # Gestion des erreurs
                    if resolution[2] != 1 and resolution[2] != 3:
                        return print("Le nombre de canaux doit être de 1 (en nuances de gris) ou de 3 (en couleur)")

                    if resolution[0] != resolution[1]:
                        return print("La largeur de l'image doit être la même que sa hauteur.")
                    
                    if normalize != 'imagenet' and normalize != 'simple':
                        print(Attention : aucune normalisation n'a été appliquée. Utilisez 'imagenet' pour une normalisation standardisée selon le mode opératoire du set ImageNet ou 'simple' pour simplement normaliser la valeur des canaux entre 0 et 1.")

                    df_images_selected_list = []
                    for label, group in df_images.groupby('LABEL'):
                        n_samples = min(len(group), n_img)
                        df_images_selected_list.append(group.sample(n=n_samples, replace=False))
                    df_images_selected = pd.concat(df_images_selected_list)

                    images = []  # Initialiser une liste pour stocker les images prétraitées
                    df_masks_selected = df_masks[df_masks['FILE_NAME'].isin(df_images_selected['FILE_NAME'])] if with_masks else None

                    for i in range(len(df_images_selected)):
                        img_path = df_images_selected[files_path].iloc[i]
                        mask_path = df_masks_selected[files_path].iloc[i] if with_masks else None

                        img = Image.open(img_path)  # Charger l'image avec PIL
                        img = img.convert("L") if resolution[2] == 1 else img.convert("RGB")

                        img_resized = img.resize((resolution[0], resolution[1]))

                        # Normalisation des valeurs des pixels
                        if normalize == 'imagenet':
                            img_normalized = np.array(img_resized) / 255.0
                            img_normalized -= np.array([0.485, 0.456, 0.406]) if resolution[2] == 3 else np.mean([0.485, 0.456, 0.406])
                            img_normalized /= np.array([0.229, 0.224, 0.225]) if resolution[2] == 3 else np.mean([0.229, 0.224, 0.225])
                        elif normalize == 'simple':
                            img_normalized = img_resized / 255
                        else:
                            img_normalized = img_resized

                        images.append(img_normalized)

                    data = np.array(images).reshape(-1, resolution[0], resolution[1], resolution[2])
                    target = df_images_selected['LABEL']
                    return data, target
                """

        with st.expander("Voir le code de la fonction preproc_img()"):
            st.code(code, language = 'python')
    
        st.markdown('''Le processus de prétraitement des données consiste à uniformiser les données en les important via `OpenCV` avec `cv2.IMREAD_GRAYSCALE` et en les convertissant en uint8 pour économiser de la mémoire. 
                       Les images peuvent être redimensionnées à la résolution de notre choix, stockées sous forme d'arrays numpy. 
                       Une normalisation de l'intensité des pixels peut être appliquée selon les besoins et les attentes des modèles, et des méthodes d'équilibrage des classes comme l'undersampling ou l'oversampling peuvent être envisagées en raison de différences significatives dans leur répartition. 
                       Les premiers masques sont utilisés pour limiter la surface aux informations utiles, avec la possibilité de créer de nouveaux masques.
                    ''')
        st.markdown(''' Dernière étape après nos images propres et normalisées, il est nécessaire de transformer nos labels multiclasses en entiers afin d'assurer la compatibilité avec une les modèles de classificiation.
                        Cette étape nécessite seulement un traitement par **One Hot Encoding** grâce à `LabelEncoder()`.
                    ''')
        
        data = {
            'Classes': ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'],
            'Numéros correspondants': [0, 1, 2, 3]
        }
        df = pd.DataFrame(data)

        # Convertir le dataframe en HTML avec les styles CSS
        html_table = df.to_html(index=False, justify='center', classes='styled-table')

        # Afficher le HTML dans Streamlit avec la largeur calculée
        st.markdown(f"<div style='border: 1px solid white; border-radius: 5px; padding: 10px; background-color: #343434; line-height: 1; width: 350px; margin: 0 auto;'>{html_table}</div>", unsafe_allow_html=True)
    
    ### Deuxième onglet
    with tab2:
        st.markdown("Texte filler")

    ### Troisième onglet
    with tab3:
        st.header("Modèles testés")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            accuracy = history_densenet['precision']
            val_accuracy = history_densenet['val_precision']
            epochs = list(range(1, len(accuracy) + 1))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x = epochs, y = accuracy, mode = 'lines+markers', name = 'Accuracy', line = dict(color = 'lightblue')))
            fig.add_trace(go.Scatter(x = epochs, y = val_accuracy, mode = 'lines+markers', name = 'Validation Accuracy', line = dict(color = 'salmon')))

            fig.update_layout(
                title='Courbes de précision pour DenseNet201',
                xaxis_title='Epoques',
                yaxis_title='Précision',
                legend_title='Type de données'
            )

            st.plotly_chart(fig)

        with col2:
            accuracy = history_vgg['accuracy']
            val_accuracy = history_vgg['val_accuracy']
            epochs = list(range(1, len(accuracy) + 1))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x = epochs, y = accuracy, mode = 'lines+markers', name = 'Accuracy', line = dict(color = 'lightblue')))
            fig.add_trace(go.Scatter(x = epochs, y = val_accuracy, mode = 'lines+markers', name = 'Validation Accuracy', line = dict(color = 'salmon')))

            fig.update_layout(
                title='Courbes de précision pour VGG16',
                xaxis_title='Epoques',
                yaxis_title='Précision',
                legend_title='Type de données'
            )

            st.plotly_chart(fig)

    ### Quatrième onglet
    with tab4:
        st.markdown("Texte filler")
