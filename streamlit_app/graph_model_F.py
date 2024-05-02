import plotly.graph_objects as go
import streamlit as st 

def plot_loss_curve(history):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(history['loss']))),
                             y=history['loss'],
                             mode='lines+markers',
                             name='Perte d\'entraînement',
                             marker=dict(color='lightblue')))

    fig.add_trace(go.Scatter(x=list(range(len(history['val_loss']))),
                             y=history['val_loss'],
                             mode='lines+markers',
                             name='Perte de validation',
                             marker=dict(color='salmon')))

    fig.update_layout(title=dict(text="Courbe de Perte", font=dict(color='white')),
                      xaxis_title=dict(text="Époque", font=dict(color='white')),
                      yaxis_title=dict(text="Perte", font=dict(color='white')),
                      template='plotly_white',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(color='white')))
    st.plotly_chart(fig)

def plot_precision_curve(history):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(history['precision']))),
                             y=history['precision'],
                             mode='lines+markers',
                             name="Précision d'entraînement",
                             marker=dict(color='lightblue')))

    fig.add_trace(go.Scatter(x=list(range(len(history['val_precision']))),
                             y=history['val_precision'],
                             mode='lines+markers',
                             name='Précision de validation',
                             marker=dict(color='salmon')))

    fig.update_layout(title=dict(text="Courbe de Précision", font=dict(color='white')),
                      xaxis_title=dict(text="Époque", font=dict(color='white')),
                      yaxis_title=dict(text="Précision", font=dict(color='white')),
                      template='plotly_white',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(color='white')))
    st.plotly_chart(fig)

def plot_auc(history):
    # Créer une figure pour la courbe AUC-ROC
    fig = go.Figure()

    # Tracer la courbe AUC-ROC d'entraînement
    fig.add_trace(go.Scatter(x=list(range(len(history['auc']))),
                              y=history['auc'],
                              mode='lines+markers',
                              name="AUC moyen d'entraînement",
                              marker=dict(color='lightblue')))

    # Tracer la courbe AUC-ROC de validation
    fig.add_trace(go.Scatter(x=list(range(len(history['val_auc']))),
                              y=history['val_auc'],
                              mode='lines+markers',
                              name='AUC moyen de validation',
                              marker=dict(color='salmon')))

    # Mettre à jour les titres et les axes
    fig.update_layout(title="Courbe de AUC-ROC",
                      xaxis_title="Époque",
                      yaxis_title="Area Under Curve",
                      template='plotly_white',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(color='white')),
                      xaxis=dict(tickfont=dict(color='white')),
                      yaxis=dict(tickfont=dict(color='white')),
                      title_font=dict(color='white'))

    # Afficher la figure
    st.plotly_chart(fig)

def plot_f1_score(history):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(history['f1_score']))),
                              y=np.mean(history['f1_score'], axis=1),
                              mode='lines+markers',
                              name="F1 Score d'entraînement",
                              marker=dict(color='lightblue')))

    fig.add_trace(go.Scatter(x=list(range(len(history['val_f1_score']))),
                              y=np.mean(history['val_f1_score'], axis=1),
                              mode='lines+markers',
                              name='F1 Score moyen de validation',
                              marker=dict(color='salmon')))

    fig.update_layout(title="Courbe de F1 Score",
                      xaxis_title="Époque",
                      yaxis_title="F1 Score",
                      template='plotly_white',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(color='white')),
                      xaxis=dict(tickfont=dict(color='white')),
                      yaxis=dict(tickfont=dict(color='white')),
                      title_font=dict(color='white'))
    st.plotly_chart(fig)