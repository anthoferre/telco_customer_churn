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



choix_partie = st.sidebar.radio("Sommaire",["I - Introduction","II - Exploration des données","III - Data visualization","IV - Modèle de prédiction du taux de désabonnement des clients", "V - Conclusion et Perspectives"])

                                                   # récupérer les données
@st.cache_data
def open_df(dataframe):
    return pd.read_csv(dataframe)

df = open_df('WA_Fn-UseC_-Telco-Customer-Churn.csv')


df_cleaned = pd.read_pickle('df_telco_customer_churn.pkl')

                                        #Partie 1 : Introduction
    
if choix_partie == 'I - Introduction':
    st.subheader('I - Introduction')
    st.image('https://vertone.com/wp-content/uploads/2018/12/adobestock_436800241-scaled.jpeg', use_container_width=False)
    
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

    import streamlit as st

    st.header("Aperçu Initial du Jeu de Données")

    st.subheader("1. Structure du Jeu de Données")
    st.markdown(f" **Taille :** L'ensemble de données se compose de **{7043} lignes** d'informations, réparties sur **{21} colonnes** distinctes.")
    st.markdown("   La majorité des colonnes, soit **18**, contiennent des données de type texte (`object`). **2** colonnes contiennent des nombres entiers (`int`). **1** colonne contient des nombres décimaux (`float`).")

    st.subheader("2. Points Requérant une Attention Particulière (Qualité des Données)")

    st.markdown("**- Colonne 'TotalCharges'**")
    st.markdown("Actuellement identifiée comme contenant du texte (`object`), cette colonne devrait normalement contenir des nombres décimaux (`float`) représentant le montant total facturé aux clients. L'examen a révélé que **11 lignes** présentent des valeurs manquantes dans cette colonne, ce qui est probablement la raison de son type de données incorrect. ")
    st.markdown("\n Solution : supprimer les 11 lignes où TotalCharges est manquant semble être une approche raisonnable pour commencer, étant donné le faible nombre de lignes concernées. Ce qui permettra de convertir la colonne en nombres décimaux.")

    st.markdown("**- Colonnes Binaires**")
    st.markdown("Plusieurs colonnes représentent des choix binaires (par exemple, 'Oui'/'Non', 'Vrai'/'Faux'). Bien qu'actuellement de type texte (`object`), il serait avantageux de les convertir en valeurs numériques (0 et 1) pour faciliter les analyses quantitatives et la modélisation.")

    st.subheader("3. Informations Clés sur les Données")

    st.markdown("**- Absence de Doublons :**")
    st.markdown("L'analyse de la colonne 'customerID' n'a révélé aucune valeur dupliquée, ce qui indique qu'il n'y a pas de lignes complètement identiques dans l'ensemble du jeu de données.")

    st.markdown("**- Déséquilibre de la Variable Cible ('Churner')**")
    st.markdown("La variable que nous cherchons à prédire ('Churner', indiquant si un client s'est désabonné ou non) est déséquilibrée. Environ **25%** des clients se sont débasonnés, tandis que la majorité est restée. Ce déséquilibre devra être pris en compte lors de la construction de modèles prédictifs (technique de sous ou sur échantillonage).")

    st.markdown("**- Absence de Valeurs Aberrantes Numériques**")
    st.markdown("Une première vérification des colonnes contenant des nombres n'a pas révélé de valeurs anormalement éloignées des autres, suggérant une certaine cohérence dans les données numériques.")

    st.header("Préprocessing")
    st.code('''# sélectionner les variables binaires Yes or No et les remplacer respectivement par 1 et 0
def selectionner_colonnes_binaires (df):
    colonnes_binaires = []
    for col in df.select_dtypes('object').columns:
        if set(df[col].unique()) == {"Yes","No"}:
            colonnes_binaires.append(col)
    return colonnes_binaires

df[selectionner_colonnes_binaires] = df[selectionner_colonnes_binaires(df)].replace({"Yes" : 1, "No" : 0})

# remplacer female par 0 et male par 1
df['gender'] = df['gender'].replace({"Female" : 0, "Male" : 1})

# one hot encoding
df = pd.get_dummies(data = df)

# supprimer toutes les colonnes contenant no internet service sauf une car toutes identiques
df = df.drop([col for col in df.columns if "No internet service" in col][1:], axis = 1)

#Supprimer la colonne no_phone_service de multiplelines car info déjà dans la variable PhoneService
df.drop('MultipleLines_No phone service', axis = 1, inplace=True)

# Renommer la colonne "No_internet_service"
df = df.rename(columns={"OnlineSecurity_No internet service" : "No_internet_service"})''')
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
    st.markdown("\n")

   
    col1,col2 = st.columns(2)
    with col1:
        st.image('Feature_importances.png', use_container_width = False, width = 600)
    with col2:
        st.markdown("Les variables les plus importantes dans le choix que réalise le modèle sur la classification binaire semblent être **streamingmovies_yes** (le client a l'option film en streaming), **tenure** (nb de mois que le client est abonné), **totalcharges** et **monthly charges** (les charges mensuelles et depuis le début de l'abonnement).")
        st.markdown("Ces 4 variables représentent plus de **50%** de l'explication du choix du modèle dans la classification churner. Les autres variables ont un impact plus faible (- de 6%)")




if choix_partie == 'V - Conclusion et Perspectives':
    st.subheader('V - Conclusion et Perspectives')

    st.markdown("#### Conclusion")
    st.markdown("""
    Cette exploration et modélisation du taux de désabonnement des clients de télécommunication a permis de mettre en lumière plusieurs aspects cruciaux. L'analyse des données brutes a révélé des défis initiaux concernant le type de certaines variables et la présence de quelques valeurs manquantes, qui ont été adressés par un pré-traitement ciblé.

    La visualisation des données a ensuite offert des insights précieux sur la distribution des différentes caractéristiques et leur relation avec la variable cible 'Churn'. Des tendances claires se sont dégagées, suggérant que certains facteurs comme la durée d'abonnement, les charges mensuelles et totales, ainsi que l'utilisation de certains services (comme le streaming de films) sont fortement corrélés au désabonnement.

    La comparaison de différents modèles de classification (Régression Logistique, Support Vector Machine, Arbre de Décision et Forêt Aléatoire) a permis d'évaluer leur performance dans la prédiction du 'Churn'. L'examen des matrices de confusion, des rapports de classification et des hyperparamètres optimisés fournit une base solide pour choisir le modèle le plus adapté aux objectifs de l'entreprise. Il est notable que certaines variables, telles que la présence de l'option 'streamingmovies_yes', la durée d'abonnement ('tenure') et les charges ('totalcharges' et 'monthly charges'), se sont avérées être des indicateurs clés dans les décisions des modèles.
    """)

    st.markdown("#### Perspectives")
    st.markdown("""
    Forts de ces résultats, plusieurs pistes d'action et de développement peuvent être envisagées :

    **1. Stratégies de Rétention Ciblées :**
    - Les insights tirés de la visualisation et de l'importance des variables peuvent être utilisés pour identifier les clients à haut risque de désabonnement. Des offres personnalisées ou des interventions proactives pourraient être mises en place pour les fidéliser.
    - Une attention particulière devrait être portée aux clients ayant des durées d'abonnement courtes, des charges mensuelles élevées ou qui n'utilisent pas certains services spécifiques.

    **2. Amélioration des Services et de l'Expérience Client :**
    - Comprendre pourquoi les clients qui utilisent certains services sont plus susceptibles de rester (ou de partir) peut orienter l'amélioration de ces services ou l'offre de services complémentaires pertinents.
    - L'analyse des commentaires clients (si disponibles dans d'autres sources de données) pourrait enrichir la compréhension des raisons du désabonnement au-delà des variables quantitatives.

    **3. Optimisation des Modèles de Prédiction :**
    - L'exploration de techniques d'équilibrage des classes plus avancées pourrait potentiellement améliorer la performance des modèles, en particulier pour la détection des clients qui vont se désabonner.
    - L'intégration de nouvelles variables (par exemple, les interactions du service client, les données de navigation sur le site web) pourrait enrichir le pouvoir prédictif des modèles.
    - Un suivi régulier des performances des modèles en production et un réentraînement périodique sont essentiels pour maintenir leur précision dans le temps.

    **4. Développement d'Outils d'Aide à la Décision :**
    - L'application Streamlit elle-même pourrait être transformée en un outil interactif pour les équipes marketing et commerciales, leur permettant de visualiser les risques de désabonnement pour des segments de clients spécifiques et de planifier des actions en conséquence.

    **5. Exploration de Modèles Plus Avancés :**
    - L'expérimentation avec des modèles de machine learning plus complexes (par exemple, réseaux de neurones, méthodes d'ensemble avancées) pourrait être envisagée pour tenter d'améliorer encore la précision des prédictions.

    En conclusion, cette analyse fournit une base solide pour comprendre et potentiellement réduire le taux de désabonnement des clients. En mettant en œuvre des stratégies basées sur ces insights et en continuant à affiner les modèles de prédiction, l'entreprise de télécommunications peut améliorer significativement la fidélisation de sa clientèle et optimiser ses opérations commerciales.
    """)