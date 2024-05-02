import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from graph_model_F import plot_auc, plot_f1_score, plot_loss_curve , plot_precision_curve

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
            df = pd.read_csv(r"df_file\Lenet_nb_image.csv")
            df2 = pd.read_csv(r"df_file\Lenet_nb_epoque.csv") 


            col1, col2 = st.columns(2)

            with col1 :

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=df['nombre_images'], y=df['Precision max'], mode='lines+markers', name='Precision max', line=dict(color='lightblue')))
                fig.add_trace(go.Scatter(x=df['nombre_images'], y=df['Precision max validation'], mode='lines+markers', name='Precision max validation', line=dict(color='salmon')))

                fig.add_vline(x=1325, line=dict(color='red', width=1, dash='dash'))
                fig.update_layout(title="Evolution de la précision max en fonction du nombre d'image",
                                xaxis_title='Nombre d\'images',
                                yaxis_title='Précision max',
                                template='plotly_white',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                legend=dict(font=dict(color='white')),
                                xaxis=dict(tickfont=dict(color='white')),
                                yaxis=dict(tickfont=dict(color='white')),
                                title_font=dict(color='white'))
                st.plotly_chart(fig)

            with col2: 

                fig2 = go.Figure()

                fig2.add_trace(go.Scatter(x=df2['Nombre epoque'], y=df2['Precision max'], mode='lines+markers', name='Precision max', line=dict(color='lightblue')))
                fig2.add_trace(go.Scatter(x=df2['Nombre epoque'], y=df2['Precision max validation'], mode='lines+markers', name='Precision max validation', line=dict(color='salmon')))

                fig2.update_layout(title="Evolution de la précision max en fonction du nombre d'époque",
                                xaxis_title='Nombre d\'époque',
                                yaxis_title='Précision max',
                                template='plotly_white',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                legend=dict(font=dict(color='white')),
                                xaxis=dict(tickfont=dict(color='white')),
                                yaxis=dict(tickfont=dict(color='white')),
                                title_font=dict(color='white'))

                st.plotly_chart(fig2)

    ### Troisième onglet
    with tab3:
            Slider = st.select_slider(" ", options = ["Transfer learning" , "Fine Tuning"])

            categorie = {"Transfer learning" :["Modèles testés","InceptionResNetV2","ResNet121V2","DenseNet201","VGG16", "VGG19","ConvNextTiny","ConvNextBase","EfficientNet B4"],
                        "Fine Tuning" : ["EfficientNet", "ResNet", "VGG16_ft" ,"DenseNet"]}

            Choice_cr = st.selectbox("Navigation",
                                    options = categorie[Slider])
            
            csv_path_cr = {"Modèles testés" :r"df_file\df test model.csv",
                        "InceptionResNetV2" :r"df_file\df InceptionRes.csv",
                        "ResNet121V2" : r"df_file\df Res.csv",
                        "DenseNet201": r"df_file\df densenet.csv",
                        "VGG16" : r"df_file\df VGG16.csv", 
                        "VGG19" : r"df_file\df VGG19.csv",
                        "ConvNextTiny" : r"df_file\df Convtiny.csv",
                        "ConvNextBase" : r"df_file\df Convbase.csv",
                        "EfficientNet B4" :r"df_file\df efficient.csv",
                        "EfficientNet" :r"df_file\df efficientnet finetuned.csv",
                        "ResNet" :r"df_file\df resnet finetuned.csv",
                        "VGG16_ft" :r"df_file\df VGG16_finetuned.csv",
                        "DenseNet" :r"df_file\df densenet_finetuned.csv"}
            
            comm_dico = {"Modèles testés" :""" Voici un récapitulatif des modèles que nous avons testé dans le cadre du transfer learning. """,
                        "InceptionResNetV2" :""" Le modèle a une capacité variable à distinguer les différentes classes de radiographies. La classe Viral Pneumonia présente d'excellents scores de précision, de rappel et de F1, indiquant une identification quasi parfaite, tandis que la classe Normal a montré des difficultés plus marquées, avec les scores les plus bas pour ces mêmes métriques. Le score F1, qui équilibre la précision et le rappel, suggère que le modèle est plus apte à identifier correctement les classes COVID et Viral Pneumonia, __mais qu'il pourrait bénéficier d'un rééquilibrage ou d'un ajustement dans la classification des classes Lung_Opacity et Normal.__ """ ,
                        "ResNet121V2" :""" Le modèle a une certaine tendance à confondre la classe COVID avec les classes Lung_Opacity et Normal, comme en témoignent les 11 erreurs dans chaque cas. Néanmoins, la classe Viral Pneumonia est interprétée avec une grande précision, indiquant que les caractéristiques distinctives de cette classe sont bien capturées par le modèle. Les métriques par classe montrent que la classe 3 se distingue avec une précision et un rappel exceptionnels proches de 0.98, menant à un score F1 similaire, qui est une mesure robuste de la précision globale. Les classes COVID, Lung_Opacity, et Normal présentent des scores F1 légèrement plus bas, mais toujours respectables, bien que ces classes pourraient bénéficier d'un réajustement du modèle pour améliorer la distinction entre elles. __La précision globale du modèle à 0.88 est solide, mais l'objectif serait de viser une amélioration dans la classification fine entre les classes similaires.__""" ,
                        "DenseNet201":"""  Les erreurs de classification les plus courantes semblent se produire entre les classes Lung_Opacity et Normal, sous-entendant des similarités entre les caractéristiques des radiographies que le modèle confond certainement. Selon le tableau de métriques, le modèle a une excellente précision pour la classe COVID et des scores exceptionnels de rappel et de F1 pour la classe Viral Pneumonia, indiquant une classification presque parfaite pour ces catégories. Les classes Lung_Opacity et Normal ont des scores F1 légèrement inférieurs mais comparables. __Tout ceci indique une bonne performance de classification qui reste uniforme entre ces catégories.__""" ,
                        "VGG16" : """ Le modèle parvient à très bien classer les radiographies des classes Viral (Viral pneumonia) et COVID. Également, même si les résultats restent bons, le modèle commet plus d'erreurs de classification entre les catégories Normal et Lung (Lung_opacity). __Sans ajustement particulier, ce modèle semble déjà prometteur quant à ses capacités à classifier nos radiographies correctement.__""", 
                        "VGG19" : """ Les résultats obtenus semblent également très bons et superposables  à ceux que nous avons obtenus pour que pour VGG16. __Cependant ce modèle étant un peu plus profond, il demande des ressources computationnelles plus importantes sans que cela ne se répercute de façon évidente sur ses performances.__ """,
                        "ConvNextTiny" : """ Avec ce modèle il apparaît que la classification est significativement meilleure pour la catégorie  Viral (Viral pneumonia) que pour les autres. Ceci donne un score global en deçà de ce que nous avons pu observer sur d’autres modèles dans les mêmes conditions de test. Les courbes d’apprentissage suggèrent que le modèle pourrait bénéficier d’un nombre d'époques supérieur pour continuer à s’améliorer. """,
                        "ConvNextBase" : """ La classe Viral pneumonia reste toujours la mieux détectée, suivie de la classe COVID. Les résultats obtenus ici sont donc comparables à ceux obtenus avec le modèle ConvNeXtTiny. Encore une fois le modèle semble pouvoir bénéficier d’un allongement de la durée d'entraînement. Cependant il est à noter que ce modèle peut se montrer gourmand en termes de ressource computationnelle, __une époque de ConvNeXtBase pouvant prendre entre deux et trois fois plus de temps que le modèle ConvNeXtTiny sans montrer une différence flagrante de performance.__""",
                        "EfficientNet B4" :"""La précision du modèle chute à 0.88 sur l’ensemble test. La détection de la classe COVID n'est pas au niveau de ce que l’on espérait avec une précision de 0.91. __Globalement, le modèle avec ce paramétrage donne de bons résultats.__ Dans la section suivante, nous allons essayer un ajustement plus fin pour avoir de  meilleures performances avec ce modèle. """,
                        "EfficientNet" :""" __Avec une précision globale de 0.94, c’est le meilleur modèle que nous avons eu pour cette partie concernant la famille de modèles EfficientNet.__ De plus, le modèle semble bien plus performant concernant la classe qui nous intéresse ici (classe COVID), avec une reconnaissance des radiographies COVID à 0.98 avec précision. Pour la suite de nos travaux, le meilleur modèle sera adopté et utilisé pour l’interprétabilité et la suite de cette étude.""",
                        "ResNet" :""" Bien que performant, le modèle tend à être freiné dans ses performances par la classe Lung_Opacity, dans laquelle il classe des poumons sains et vice-versa. Quelques poumons sains sont aussi incorrectement classés en Viral Pneumonia. __Ce modèle est donc performant, et supprimer la classe Lung_Opacity bénéficierait certainement beaucoup au InceptionResNetV2.__""",
                        "VGG16_ft" :""" Le modèle a donc été entraîné avec ces paramètres ce qui nous permet d’améliorer encore l’efficacité du modèle par rapport à ce que nous avions obtenu sans finetuning. Les classes les mieux prédites sont COVID et Viral (Viral pneumonia), suivies par les classes Lung Opacity et Normal. De façon intéressante , toutes nos métriques sont au-dessus de 90% et suite à l'entraînement du modèle avec les meilleurs paramètres nous obtenons une accuracy globale de 95%. __Ce modèle semble donc capable de fournir des résultats plus qu’acceptables tout en ayant un coût computationnel très contenu.__""",
                        "DenseNet" :""" Le rapport de classification montre des valeurs élevées pour la précision, le rappel et le F1 Score pour chaque classe ce qui indique que le modèle est particulièrement performant dans la distinction entre les différentes conditions. A noter cependant qu’il performe tout particulièrement dans la distinction de la classe COVID et de la classe Viral Pneumonia mais est un peu moins efficace dans la détection des classes Normal et Lung_Opacity. Pour le COVID, le modèle a très bien performé, avec seulement 3 faux positifs et faux négatifs. Les résultats pour les autres conditions sont également bons, mais on note quelques erreurs, par exemple, 23 cas de Lung_Opacity ont été confondus avec la classe Normal. __Néanmoins, ces erreurs semblent être faibles en comparaison avec le nombre total de prédictions correctes.__"""}


            df = pd.read_csv(csv_path_cr[Choice_cr])
            df= df.fillna("")

            # Convertir le dataframe en HTML avec les styles CSS
            html_table = df.to_html(index=False, justify='center', classes='styled-table')

            # Définir le style CSS pour centrer l'affichage du DataFrame et le fond
            css_style = """
            <style>
                .background-div {
                    max-width: fit-content; /* Largeur adaptée au contenu */
                    margin: 0 auto; /* Centrer horizontalement */
                    padding: 10px;
                    background-color: #343434;
                    border-radius: 5px;
                }
                .inner-div {
                    text-align: center; /* Centrer le contenu */
                }
            </style>
            """

            # Ajouter le style CSS à la balise div
            styled_html_table = f"<div class='background-div'><div class='inner-div'>{html_table}</div></div>"

            col1, col2 = st.columns(2)

            with col1 :

                st.markdown(css_style, unsafe_allow_html=True)
                st.markdown(styled_html_table, unsafe_allow_html=True)
            
            with col2 :
                 
                st.markdown(comm_dico[Choice_cr])


    ### Quatrième onglet
    with tab4:

        model_f = st.selectbox ('Meilleurs modèles', options = ["VGG16" , "DenseNet"] ) 

        path_pickle = {"VGG16" : r"pickle_file\model_historyVGG16BP_test.pkl",
                    "DenseNet" : r"pickle_file\history_DenseNet201_finetuned_0_95_20epochs.pkl"}
        
        best_hp = {"VGG16" : """ 
                   - Dernière couche dense : 1024 neurones
                   - Dropout : 0
                   - Learningrate : 10e-4 """,
                   "DenseNet" : """ 
                    - Dernière couche dense : 256 neurones (Regularisation L2 : 0.01)
                    - Dropout : 0.4,
                    - Learning rate : 10e-4 """}
        
        with open(path_pickle[model_f], 'rb') as fichier:
        # Charger les données à partir du fichier
            history = pickle.load(fichier)
        
    Col1 , Col2 = st.columns(2)

    with Col1:
        plot_loss_curve(history)
        plot_auc(history)
    
    with Col2:
        plot_precision_curve(history)
        plot_f1_score(history)

    st.markdown(best_hp[model_f])
