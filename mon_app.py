# importer toutes les librairies
import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.metrics import classification_report_imbalanced

st.set_page_config(layout="wide")

st.title("Taux de désabonnements des clients de télécommunication")



choix_partie = st.sidebar.radio("Sommaire",["I - Introduction","II - Exploration des données","III - Data visualization","IV - Modèle de prédiction du taux de désabonnement des clients","V - Segmentation client", "VI - Conclusion et Perspectives"])

                                                   # récupérer les données
@st.cache_data
def open_df(dataframe):
    return pd.read_csv(dataframe)

df = open_df('WA_Fn-UseC_-Telco-Customer-Churn.csv')


df_cleaned = pd.read_pickle('df_telco_customer_churn.pkl')

                                        #Partie 1 : Introduction
    
if choix_partie == 'I - Introduction':
    st.subheader('I - Introduction')
    st.text('''Ce jeu de données fournit un aperçu complet des clients d'une entreprise de télécommunications, en se concentrant sur les facteurs qui influencent leur décision de quitter le service (désabonnement ou "churn"). Il contient des informations démographiques, des détails sur leurs comptes, les services qu'ils utilisent et leurs interactions avec l'entreprise. \n\nL'objectif principal de ce jeu de données est de permettre l'analyse du comportement des clients et la prédiction du taux de désabonnement, afin de développer des stratégies de fidélisation efficaces.''')

if choix_partie == 'II - Exploration des données':
    st.subheader('II - Exploration des données')

    pd.set_option('display.max_columns',None)
    st.dataframe(df.head(100))

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Afficher les informations détaillées"): # Permet de masquer/afficher les détails
            import io
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

    with col2:
        with st.expander("Afficher les statistiques descriptives"):
            st.dataframe(df.describe()) # Affiche les statistiques d

if choix_partie == 'III - Data visualization':
    st.subheader('III - Data visualization')

    type_graphique = ['Distribution','Relation avec la variable churn','Proportion','Corrélation']
    graphique_choisi = st.radio("Quel est le type d'analyse graphique souhaitée?", type_graphique)
    features_list_df_cleaned = list(df_cleaned.columns)
    features_list_df = list(df.drop('customerID', axis = 1).columns)

    if graphique_choisi == 'Distribution':
        
        x_choisi = st.selectbox(label = 'Choisir une variable en abscisse',options = features_list_df_cleaned)
        nb_bins =   st.number_input('Nombre de séparations sur le graphique',2,50)
        ax = sns.displot(df_cleaned[x_choisi], bins = nb_bins, stat = 'percent', aspect = 3)
        plt.title('Distribution de la variable {}'.format(x_choisi))
        st.pyplot(plt)

    if graphique_choisi == 'Relation avec la variable churn':

        x_choisi = st.selectbox(label = '''Choisir une variable d'intérêt''',options = features_list_df_cleaned)

        if len(df_cleaned[x_choisi].unique()) > 5:
            ax = sns.catplot(data = df_cleaned, x = x_choisi, hue = 'Churn', kind = st.selectbox(label = 'Type de graphique', options = ['box','violin','boxen']), aspect = 3)
            plt.title('Taux de désabonnement en fonction de la variable {}'.format(x_choisi),fontdict = {'fontsize' : 20})
            ax.set_xticklabels(rotation = 90)
            st.pyplot(plt)

        else:
            ax = sns.countplot(data = df_cleaned, x = x_choisi, hue = 'Churn', stat = 'percent')
            plt.title('Taux de désabonnement en fonction de la variable {}'.format(x_choisi), fontdict = {'fontsize' : 20})
            for container in ax.containers:
                labels = [f'{float(np.round(x,1))}' for x in container.datavalues]
                ax.bar_label(container, labels=labels, label_type='edge', padding=3)
            st.pyplot(plt)

    if graphique_choisi == 'Proportion':

        x_choisi = st.selectbox(label = '''Choisir une variable d'intérêt''',options = features_list_df)
        plt.figure(figsize = (2,2))
        plt.title('Proportion de la variable {}'.format(x_choisi))
        plt.pie(x = df[x_choisi].value_counts(), labels= df[x_choisi].value_counts().index, autopct='%1.1f%%', textprops={'fontsize': 7})
        st.pyplot(plt)

    if graphique_choisi == 'Corrélation':
        ax = sns.heatmap(data = df_cleaned.select_dtypes(['int','float']).corr(), annot = True, fmt = '0.2f', annot_kws = {'size' : 10})
        plt.title('Corrélation entre les différentes variables du jeu de données', fontdict = {'fontsize' : 20})
        st.pyplot(plt)

if choix_partie == 'IV - Modèle de prédiction du taux de désabonnement des clients':
    st.subheader('IV - Modèle de prédiction du taux de désabonnement des clients')

    # Fonction pour charger le modèle
    @st.cache_resource  # Utilise st.cache_resource pour ne charger le modèle qu'une seule fois
    def load_model(model_path):
        loaded_model = joblib.load(model_path)
        return loaded_model

    # Chargement du modèle
    model_lr = load_model('my_model_Logistic Regression.pkl')
    model_svm = load_model('my_model_Support Vector Machine.pkl')
    model_dt = load_model('my_model_Decision Tree.pkl')
    model_rf = load_model('my_model_Random Forest.pkl')


    def plot_confusion_matrix(y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Prédictions')
        plt.ylabel('Vraies valeurs')
        return fig

    def display_model_info(col, title, model, X_test, y_test, labels):
        col.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 1.2em'>{title}</div>", unsafe_allow_html = True)

        # Affichage du meilleur score (si disponible)
        if hasattr(model, 'best_score_'):
            col.markdown(f"<div style='text-align: center;'>Meilleur score : {np.round(model.best_score_, 2)}</div>", unsafe_allow_html=True)
        else:
            col.write('Meilleur score : Non disponible')

        # Prédictions sur les données de test
        y_pred = model.predict(X_test)

        # Affichage de la matrice de confusion
        col.markdown("<div style='text-align: center; font-weight: bold;'>Matrice de Confusion :</div>", unsafe_allow_html=True)
        cm_fig = plot_confusion_matrix(y_test, y_pred, labels)
        col.pyplot(cm_fig)

        # Affichage du classification report imbalanced
        col.write('**Classification Report Imbalanced:**')
        target_names_str = [str(label) for label in labels]
        report_imbalanced_str = classification_report_imbalanced(y_test, y_pred, target_names=target_names_str)
        col.code(report_imbalanced_str)

        # Affichage des hyperparamètres
        col.markdown("<div style='text-align: center; font-weight: bold;'>Hyperparamètres :</div>", unsafe_allow_html=True)
        if hasattr(model, 'best_params_'):
            for param, value in model.best_params_.items():
                col.write(f'{param}: {value}')
        elif hasattr(model, 'get_params'):
            params = model.get_params()
            for param, value in params.items():
                col.write(f'{param}: {value}')
        else:
            col.write('Hyperparamètres : Non disponibles')

    # --- Configuration principale de Streamlit ---
    st.title("Comparaison des Modèles de Classification")

    # Charger les données de test (assure-toi que les chemins sont corrects)
    try:
        x_test = joblib.load('x_test')
        y_test = joblib.load('y_test')
    except FileNotFoundError:
        st.error("Les fichiers de données ou de labels sont introuvables. Veuillez vérifier les chemins.")
        st.stop()

    # Charger les modèles (assure-toi que les chemins sont corrects)
    try:
        model_lr = joblib.load('my_model_Logistic Regression.pkl')
        model_svm = joblib.load('my_model_Support Vector Machine.pkl')
        model_dt = joblib.load('my_model_Decision Tree.pkl')
        model_rf = joblib.load('my_model_Random Forest.pkl')
    except FileNotFoundError:
        st.error("Un ou plusieurs fichiers de modèle sont introuvables. Veuillez vérifier les chemins.")
        st.stop()

    # Créer les quatre colonnes
    col1, col2, col3, col4 = st.columns(4)

    # Afficher les informations pour chaque modèle
    display_model_info(col1, 'Logistic Regression', model_lr, x_test, y_test, labels = [0,1])
    display_model_info(col2, 'Support Vector Machine', model_svm, x_test, y_test, labels = [0,1])
    display_model_info(col3, 'Decision Tree', model_dt, x_test, y_test, labels = [0,1])
    display_model_info(col4, 'Random Forest', model_rf, x_test, y_test, labels = [0,1])

    st.markdown("<div style='text-align: center; font-size: 1.2em; font-weight: bold'>Feature Importances</div>", unsafe_allow_html = True)
    st.image('Feature_importances.png', use_container_width = False, width = 600)



if choix_partie == 'V - Segmentation client':
    st.subheader('V - Segmentation client')

if choix_partie == 'VI - Conclusion et Perspectives':
    st.subheader('VI - Conclusion et Perspectives')