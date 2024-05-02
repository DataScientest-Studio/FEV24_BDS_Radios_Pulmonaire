import streamlit as st
from PIL import Image

def show_interpretability():
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

    tab1, tab2 = st.tabs(["🧰 Suivi des métriques", "👀 Analyse de la GRAD-CAM"])


    with tab1:
        st.header("Suivi des métriques")
        st.markdown('''
        Dans le domaine du deep learning appliqué à la santé, l'évaluation des modèles joue un rôle crucial pour mesurer leur performance et leur pertinence clinique. 
        Les métriques utilisées fournissent des informations essentielles sur la capacité du modèle à généraliser à de nouvelles données et à fournir des prédictions précises et fiables.
                    ''')
    
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Précision (accuracy)")
            st.markdown('''
            La précision est l'une des métriques les plus fondamentales en deep learning. 
            Elle mesure la proportion de prédictions correctes parmi toutes les prédictions effectuées par le modèle. 
            Bien qu'elle soit souvent utilisée comme mesure globale de performance, elle peut être trompeuse dans le contexte médical lorsque les classes sont déséquilibrées. 
            Par exemple, dans le diagnostic médical, où les cas positifs sont rares par rapport aux cas négatifs, une haute précision peut être obtenue simplement en prédisant toujours la classe majoritaire (négative), ce qui masquerait l'incapacité du modèle à détecter les cas positifs.
                        ''')

            st.subheader("F1 Score")
            st.markdown('''
            Le score F1 est une mesure qui combine à la fois la précision et la sensibilité en un seul nombre. 
            Il est particulièrement utile lorsque le déséquilibre entre les classes est important, car il prend en compte à la fois les faux positifs et les faux négatifs. 
            Dans le domaine médical, où les conséquences des erreurs de prédiction peuvent être graves, le score F1 est souvent préféré pour évaluer la performance des modèles de diagnostic et de détection des maladies.
                        ''')
        
        with col2:
            st.subheader("Sensibilité et Spécificité")
            st.markdown('''
            La sensibilité (recall) mesure la capacité du modèle à identifier correctement les cas positifs parmi tous les cas réellement positifs. 
            Elle est particulièrement importante dans les applications médicales où la détection précoce des anomalies ou des maladies est cruciale. 
            D'un autre côté, la spécificité mesure la capacité du modèle à identifier correctement les cas négatifs parmi tous les cas réellement négatifs. 
            Ensemble, la sensibilité et la spécificité fournissent une image plus complète de la capacité du modèle à discriminer entre les classes et à minimiser les faux positifs et les faux négatifs.
                        ''')

            st.subheader("Courbe ROC et aire sous la courbe (RAC-AUC)")
            st.markdown('''
            La courbe ROC (Receiver Operating Characteristic) est un graphique qui représente la performance d'un modèle de classification à différents seuils de classification. 
            Elle compare la sensibilité (taux de vrai positif) au taux de faux positif (1 - spécificité) à différents seuils de décision. 
            L'aire sous la courbe (AUC) ROC quantifie la capacité du modèle à discriminer entre les classes et fournit une mesure agrégée de sa performance. 
            Dans le contexte médical, une AUC élevée indique une capacité de diagnostic élevée et une meilleure capacité à séparer les classes.
                        ''')
        

    with tab2:
        st.header("Analyse de la GRAD-CAM")