import streamlit as st
from custom_functions import create_styled_box

def show_presentation():
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
        
    tab1, tab2 = st.tabs(["🗣️ Contexte", "🎯 Objectifs"])

    with tab1:
        st.header("Contexte du projet")
        st.markdown("Les radiographies pulmonaires en particulier, sont couramment utilisées pour évaluer l'état des poumons et des structures avoisinantes. Elles sont précieuses pour le diagnostic et la surveillance de diverses affections pulmonaires, telles que les pneumonies, les tuberculoses, les tumeurs pulmonaires et les maladies respiratoires obstructives. Dans le contexte de la pandémie mondiale de COVID-19, les radiographies pulmonaires sont devenues **un outil essentiel pour le dépistage et le suivi** des patients atteints de cette maladie virale respiratoire.")
        st.markdown("Le COVID-19, causé par le coronavirus SARS-CoV-2, peut provoquer une pneumonie virale sévère chez certains patients, **caractérisée par des lésions pulmonaires spécifiques visibles sur les radiographies**. Ces anomalies radiographiques comprennent généralement des opacités pulmonaires diffuses, des infiltrats interstitiels et des consolidations alvéolaires. La capacité à identifier et à interpréter ces caractéristiques radiographiques est **cruciale pour le diagnostic rapide et la gestion clinique efficace** des patients atteints de COVID-19.")

    with tab2:
        st.header("Objectifs")
        st.markdown("L'objectif principal de ce projet est de développer un système d'analyse automatisée de radiographies pulmonaires pour la détection efficace du COVID-19 sur moins de deux mois en parallèle de notre formation.")

        create_styled_box(text = "📑 Le rapport complet est disponible sur le GitHub du projet.", 
                        text_color = '#A9A9A9', 
                        background_color = '#444444')
