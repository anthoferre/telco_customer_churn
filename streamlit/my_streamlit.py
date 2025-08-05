# Importer toutes les librairies
import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import shap

# Configuration de Seaborn
sns.set_theme(style="whitegrid")

st.set_page_config(layout="wide", page_title = 'Customer_Churn')

# Masquer le menu de Streamlit et le bouton de déploiement
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Fonctions de chargement des données avec cache ---

# Utilisation de st.cache_data pour les DataFrames qui ne changent pas
@st.cache_data
def load_dataframe(filepath):
    """Charge un DataFrame à partir d'un fichier CSV ou Pickle."""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.pkl'):
        return pd.read_pickle(filepath)
    else:
        raise ValueError("Format de fichier non pris en charge. Utilisez .csv ou .pkl")

# Charger les DataFrames une seule fois
df = load_dataframe('../database/raw_data.csv')
df_cleaned = load_dataframe('../database/cleaned_data.pkl')

# --- Préparation des listes de colonnes
@st.cache_data
def get_cleaned_features_lists(dataframe):
    """Prépare les listes de variables numériques et catégorielles."""
    features_num = [col for col in dataframe.select_dtypes(['int','float']) if dataframe.select_dtypes(['int','float'])[col].nunique() > 2]
    features_cat = [col for col in dataframe.columns if dataframe[col].nunique() <= 10 and col != 'Churn']
    return features_num, features_cat

features_list_df_cleaned_num, features_list_df_cleaned_cat = get_cleaned_features_lists(df_cleaned)

# Fonction pour charger le modèle
@st.cache_resource
def load_ml_model(model_path):
    return joblib.load(model_path)

@st.cache_data
def load_test_data(data_path):
    return joblib.load(data_path)

try:
    x_test = load_test_data('../train_model/x_test.pkl')
    y_test = load_test_data('../train_model/y_test.pkl')
except FileNotFoundError:
    st.error("Les fichiers `x_test` ou `y_test` sont introuvables. Assurez-vous qu'ils sont dans le même répertoire que votre script.")
    st.stop()

# Charger les modèles
try:
    churn_model = load_ml_model('../train_model/best_overall_churn_model.pkl')
except FileNotFoundError:
    st.error("Le fichier du modèle est introuvable. Veuillez vérifier le chemin.")
    st.stop()

# Noms des caractéristiques après prétraitement pour SHAP
def get_feature_names_after_preprocessing(model, original_df_columns):
    preprocessor = model.named_steps['preprocessor']
    
    # Créer un DataFrame avec une ligne dummy pour obtenir les noms de colonnes transformées
    dummy_df = pd.DataFrame(columns=original_df_columns, data=np.zeros((1, len(original_df_columns))))
    
    # Appliquer la transformation pour obtenir les noms des features traitées
    # Assurez-vous que l'input du préprocesseur est un DataFrame
    transformed_features = preprocessor.get_feature_names_out()
    
    if 'selector' in model.named_steps:
        selector = model.named_steps['selector']
        selected_indices = selector.get_support(indices=True)
        final_feature_names = [transformed_features[i] for i in selected_indices]
    else:
        final_feature_names = list(transformed_features)
    
    return final_feature_names

all_original_features = [col for col in df_cleaned.columns if col != 'Churn']
final_shap_feature_names = get_feature_names_after_preprocessing(churn_model, all_original_features)


# --- Préparer les noms de features pour l'affichage SHAP (raccourcis et clairs) ---
feature_display_names = {
    'tenure': 'Ancienneté (mois)',
    'MonthlyCharges': 'Frais Mensuels (€)',
    'TotalCharges': 'Frais Totaux (€)',
    'gender_Female': 'Genre: Femme',
    'gender_Male': 'Genre: Homme',
    'SeniorCitizen': 'Senior',
    'Partner_No': 'Partenaire: Non',
    'Partner_Yes': 'Partenaire: Oui',
    'Dependents_No': 'Dépendants: Non',
    'Dependents_Yes': 'Dépendants: Oui',
    'PhoneService_No': 'Tel: Non', # Plus court
    'PhoneService_Yes': 'Tel: Oui', # Plus court
    'MultipleLines_No': 'Lignes: Non', # Plus court
    'MultipleLines_No phone service': 'Lignes: Pas Tel', # Plus court
    'MultipleLines_Yes': 'Lignes: Oui', # Plus court
    'InternetService_Fiber optic': 'Internet: Fibre', # Plus court
    'InternetService_DSL': 'Internet: DSL',
    'InternetService_No': 'Internet: Aucun', # Plus court
    'OnlineSecurity_No': 'Sécurité: Non', # Plus court
    'OnlineSecurity_No internet service': 'Sécurité: Pas Internet', # Plus court
    'OnlineSecurity_Yes': 'Sécurité: Oui', # Plus court
    'OnlineBackup_No': 'Sauvegarde: Non', # Plus court
    'OnlineBackup_No internet service': 'Sauvegarde: Pas Internet', # Plus court
    'OnlineBackup_Yes': 'Sauvegarde: Oui', # Plus court
    'DeviceProtection_No': 'Protection App: Non', # Plus court
    'DeviceProtection_No internet service': 'Protection App: Pas Internet', # Plus court
    'DeviceProtection_Yes': 'Protection App: Oui', # Plus court
    'TechSupport_No': 'Support Tech: Non', # Plus court
    'TechSupport_No internet service': 'Support Tech: Pas Internet', # Plus court
    'TechSupport_Yes': 'Support Tech: Oui', # Plus court
    'StreamingTV_No': 'TV Streaming: Non', # Plus court
    'StreamingTV_No internet service': 'TV Streaming: Pas Internet', # Plus court
    'StreamingTV_Yes': 'TV Streaming: Oui', # Plus court
    'StreamingMovies_No': 'Films Streaming: Non', # Plus court
    'StreamingMovies_No internet service': 'Films Streaming: Pas Internet', # Plus court
    'StreamingMovies_Yes': 'Films Streaming: Oui', # Plus court
    'Contract_Month-to-month': 'Contrat: Mensuel',
    'Contract_One year': 'Contrat: 1 An',
    'Contract_Two year': 'Contrat: 2 Ans',
    'PaperlessBilling_No': 'Facturation Elec: Non', # Plus court
    'PaperlessBilling_Yes': 'Facturation Elec: Oui', # Plus court
    'PaymentMethod_Electronic check': 'Paiement: Chèque Elec', # Plus court
    'PaymentMethod_Mailed check': 'Paiement: Chèque Poste', # Plus court
    'PaymentMethod_Bank transfer (automatic)': 'Paiement: Virement Auto', # Plus court
    'PaymentMethod_Credit card (automatic)': 'Paiement: CB Auto' # Plus court
}
display_feature_names_for_shap = [feature_display_names.get(f, f) for f in final_shap_feature_names]


# --- Préparer les données pour SHAP 
@st.cache_resource
def prepare_shap_explainer(_model_pipeline, x_data_for_explainer):
    """
    Crée l'explainer SHAP et prépare les données d'arrière-plan si nécessaire.
    """
    model = _model_pipeline.named_steps['model']
    
    # Transformer les données complètement avec le pipeline, sauf le modèle final
    # Cela garantit que x_data_transformed a le bon nombre de colonnes
    # pour le sélecteur et l'explainer.
    
    # Crée une version "tronquée" du pipeline pour les transformations uniquement
    pipeline_for_transform = _model_pipeline[:-1] # Exclut la dernière étape (le modèle)
    
    x_data_transformed = pipeline_for_transform.transform(x_data_for_explainer)
    
    # Obtenir les noms des features après le préprocesseur et le sélecteur
    # C'est la partie délicate, et `get_feature_names_out` doit être appelé
    # sur la dernière étape du pipeline de transformation si elle existe.
    
    # Initialisez avec les noms de colonnes originaux de x_data_for_explainer
    current_feature_names = x_data_for_explainer.columns.tolist()

    # Parcourez les étapes du pipeline de transformation pour suivre les noms de features
    for step_name, transformer in pipeline_for_transform.steps:
        if hasattr(transformer, 'get_feature_names_out'):
            try:
                # Si le transformer a input_features, utilisez-le
                if 'input_features' in transformer.get_feature_names_out.__code__.co_varnames:
                    current_feature_names = transformer.get_feature_names_out(input_features=current_feature_names)
                else: # Sinon, appelez sans arguments ou avec l'appel par défaut
                    current_feature_names = transformer.get_feature_names_out()
            except Exception:
                # Fallback si get_feature_names_out échoue pour une raison quelconque
                # Ceci est une gestion d'erreur simplifiée, adaptez si nécessaire
                current_feature_names = [f"{step_name}_{i}" for i in range(x_data_transformed.shape[1])]
        elif step_name == 'preprocessor': # Gérer le ColumnTransformer spécifiquement si besoin
             # Un ColumnTransformer peut avoir ses propres noms de sortie
             # Cette partie est plus complexe car elle dépend de vos transformers internes
             # Pour l'instant, on se base sur les noms des colonnes de sortie de la transformation
            pass # On va générer les noms après la transformation si on ne peut pas les obtenir directement
        elif step_name == 'selector':
            # Le sélecteur filtre les colonnes, on doit utiliser get_support
            selected_indices = transformer.get_support(indices=True)
            current_feature_names = [current_feature_names[i] for i in selected_indices]
        else:
            # Pour d'autres transformers qui n'ont pas get_feature_names_out,
            # on doit s'assurer que current_feature_names correspond toujours aux colonnes
            # Si le transformer modifie le nombre de colonnes, cela peut devenir compliqué.
            # Pour la plupart des cas simples (scaler, etc.), les noms ne changent pas,
            # mais leur ordre ou leur nombre si des colonnes sont supprimées/ajoutées
            # doit être géré.
            pass # Ici, on suppose que le nombre de colonnes est cohérent

    # Si le pipeline final n'a pas de get_feature_names_out ou si c'est un sélecteur
    # et que la logique ci-dessus n'a pas mis à jour current_feature_names correctement,
    # on doit se rabattre sur une méthode plus générique
    if len(current_feature_names) != x_data_transformed.shape[1]:
        # Fallback générique si le nombre de features ne correspond pas
        final_feature_names = [f"feature_{i}" for i in range(x_data_transformed.shape[1])]
    else:
        final_feature_names = current_feature_names
            

    # Initialisation de l'explainer SHAP
    if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
        explainer = shap.TreeExplainer(model)
    else:
        sample_size = min(100, x_data_transformed.shape[0])
        background_data = shap.utils.sample(x_data_transformed, sample_size, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background_data)
        
    return explainer, final_feature_names # Retourne explainer et les noms de features finaux


# Initialisation de l'explainer SHAP une seule fois au début du script
explainer, final_shap_feature_names = prepare_shap_explainer(churn_model, x_test)


# --- Barre latérale pour la navigation ---
choix_partie = st.sidebar.radio(
    "Sommaire",
    [
        "I - Introduction",
        "II - Exploration des données",
        "III - Data visualization",
        "IV - Le meilleur modèle de prédiction du taux de désabonnement des clients",
        'V - Etude de cas',
        "VI - Conclusion et Perspectives"
    ]
)

# --- Contenu principal de l'application ---

# Partie 1 : Introduction
if choix_partie == 'I - Introduction':
    st.title("Taux de désabonnements des clients de télécommunication")
    st.subheader('I - Introduction')
    st.image('https://vertone.com/wp-content/uploads/2018/12/adobestock_436800241-scaled.jpeg', use_container_width=False)
    st.markdown("""
    Ce jeu de données fournit un aperçu complet des clients d'une entreprise de télécommunications, en se concentrant sur les facteurs qui influencent leur décision de quitter le service (désabonnement ou "churn"). Il contient des informations démographiques, des détails sur leurs comptes, les services qu'ils utilisent et leurs interactions avec l'entreprise. 
    \n\nL'objectif principal de ce jeu de données est de permettre l'analyse du comportement des clients et la prédiction du taux de désabonnement, afin de développer des stratégies de fidélisation efficaces.
    """)

# Partie 2 : Exploration des données
elif choix_partie == 'II - Exploration des données':
    st.title('II - Exploration des données') 
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Afficher les informations détaillées"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

    with col2:
        with st.expander("Afficher les statistiques descriptives"):
            st.dataframe(df.describe())

    st.header("Aperçu Initial du Jeu de Données")
    st.subheader("1. Structure du Jeu de Données")
    st.markdown(f"**Taille :** L'ensemble de données se compose de **{df.shape[0]} lignes** d'informations, réparties sur **{df.shape[1]} colonnes** distinctes.")
    st.markdown("La majorité des colonnes, soit **18**, contiennent des données de type texte (`object`). **2** colonnes contiennent des nombres entiers (`int`). **1** colonne contient des nombres décimaux (`float`).")

    st.subheader("2. Points Requérant une Attention Particulière (Qualité des Données)")
    st.markdown("**- Colonne 'TotalCharges'**")
    st.markdown("""
    Actuellement identifiée comme contenant du texte (`object`), cette colonne devrait normalement contenir des nombres décimaux (`float`) représentant le montant total facturé aux clients. L'examen a révélé que **11 lignes** présentent des valeurs manquantes dans cette colonne, ce qui est probablement la raison de son type de données incorrect.
    \nSolution : supprimer les 11 lignes où TotalCharges est manquant semble être une approche raisonnable pour commencer, étant donné le faible nombre de lignes concernées. Ce qui permettra de convertir la colonne en nombres décimaux.
    """)

    st.markdown("**- Colonnes Binaires**")
    st.markdown("Plusieurs colonnes représentent des choix binaires (par exemple, 'Oui'/'Non', 'Vrai'/'Faux'). Bien qu'actuellement de type texte (`object`), il serait avantageux de les convertir en valeurs numériques (0 et 1) pour faciliter les analyses quantitatives et la modélisation.")

    st.subheader("3. Informations Clés sur les Données")
    st.markdown("**- Absence de Doublons :**")
    st.markdown("L'analyse de la colonne 'customerID' n'a révélé aucune valeur dupliquée, ce qui indique qu'il n'y a pas de lignes complètement identiques dans l'ensemble du jeu de données.")

    st.markdown("**- Déséquilibre de la Variable Cible ('Churner')**")
    st.markdown(f"La variable que nous cherchons à prédire ('Churn', indiquant si un client s'est désabonné ou non) est déséquilibrée. Environ **{np.round(df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100)}%** des clients se sont désabonnés, tandis que la majorité est restée. Ce déséquilibre devra être pris en compte lors de la construction de modèles prédictifs (technique de sous ou sur échantillonnage).")

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

# Partie 3 : Data Visualization
elif choix_partie == 'III - Data visualization':
    st.title('III - Data visualization')

    type_graphique = [
        "Distribution d'une variable",
        "Distribution des variables numériques par statut de désabonnement",
        "Taux de désabonnement par variable catégorielle",
        "Proportion",
        "Corrélation"
    ]
    
    col1, col2 = st.columns([1.5,2.5])
    with col1:
        graphique_choisi = st.radio("Quel est le type d'analyse graphique souhaitée?", type_graphique)
    
    with col2:
        if graphique_choisi == "Distribution d'une variable":

            x_choisi = st.selectbox(label='Choisir une variable en abscisse', options=df_cleaned.columns)
        
            fig, ax = plt.subplots(figsize=(10, 6))
            if df_cleaned[x_choisi].nunique() > 5:
                sns.histplot(df_cleaned[x_choisi], stat='percent', ax=ax)
                ax.set_title(f'Distribution de la variable {x_choisi}', fontsize=16)
                ax.set_ylabel('Pourcentage')
            else:
                sns.countplot(x=df_cleaned[x_choisi], hue=df_cleaned[x_choisi], stat='percent', palette='viridis', ax=ax, legend=False)
                ax.set_title(f'Distribution de la variable {x_choisi}', fontsize=16)
                ax.set_ylabel('Pourcentage')
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3)
            st.pyplot(fig)
            plt.close(fig)

        if graphique_choisi == 'Distribution des variables numériques par statut de désabonnement':
            x_choisi = st.selectbox(label='Choisir une variable en abscisse', options=features_list_df_cleaned_num)
            
            fig, axes = plt.subplots(1, 2, figsize=(18, 7))

            sns.histplot(data=df_cleaned, x=x_choisi, hue='Churn', multiple='fill', bins=20, ax=axes[0], palette='coolwarm')
            axes[0].set_title(f'Distribution du Churn par {x_choisi}', fontsize=14)
            axes[0].set_xlabel(f'{x_choisi}')
            axes[0].set_ylabel('Proportion de Churn')
            axes[0].set_yticks(np.arange(0, 1.1, 0.2))
            axes[0].set_yticklabels([f'{int(p*100)}%' for p in np.arange(0, 1.1, 0.2)])
            axes[0].legend(title='Churn', labels=['Non', 'Oui'])

            sns.boxplot(data=df_cleaned, y=x_choisi, hue='Churn', ax=axes[1], palette='Set2')
            axes[1].set_title(f'Distribution de {x_choisi} par Statut de Churn', fontsize=14)
            axes[1].set_xlabel('Statut de Churn')
            axes[1].set_ylabel(f'{x_choisi}')
            axes[1].set_xticks([0, 1])
            axes[1].set_xticklabels(['Non-Churn', 'Churn'])
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        elif graphique_choisi == 'Taux de désabonnement par variable catégorielle':
            x_choisi = st.selectbox(label='Choisir une variable d\'intérêt', options=features_list_df_cleaned_cat)

            fig, ax = plt.subplots(figsize=(12, 7))

            df_crosstab = pd.crosstab(df_cleaned[x_choisi], df_cleaned.Churn, normalize='index') * 100
            df_crosstab = df_crosstab.reset_index().melt(id_vars = x_choisi, var_name = 'Churn', value_name = 'Percentage')
            sns.barplot(data = df_crosstab, x = x_choisi, y = 'Percentage', hue = 'Churn', ax = ax) 
            ax.set_title(f"Taux de désabonnement en fonction de la variable {x_choisi}")
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3)

            if set(df_cleaned[x_choisi].unique()) == {0, 1}:
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Non', 'Oui'])
            elif set(df_cleaned[x_choisi].unique()) == {1, 0}:
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Oui', 'Non'])

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        elif graphique_choisi == 'Proportion':
            x_choisi = st.selectbox(label='Choisir une variable d\'intérêt', options=features_list_df_cleaned_cat)
            
            fig, ax = plt.subplots(figsize=(2, 2))
            plt.title(f'Proportion de la variable {x_choisi}', fontsize=16)
            
            counts = df_cleaned[x_choisi].value_counts()
            labels = counts.index
            
            ax.pie(x=counts, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 7}, pctdistance=0.7)
            ax.axis('equal')
            
            st.pyplot(fig)
            plt.close(fig)

        elif graphique_choisi == 'Corrélation':
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(data=df_cleaned.select_dtypes(['int', 'float']).corr(), annot=True, fmt='.2f', annot_kws={'size': 9}, cmap='coolwarm', ax=ax)
            plt.title('Corrélation entre les différentes variables du jeu de données', fontdict={'fontsize': 18})
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# Partie 4 : Modèle de prédiction du taux de désabonnement des clients
elif choix_partie == 'IV - Le meilleur modèle de prédiction du taux de désabonnement des clients':
    st.title('IV - Le meilleur modèle de prédiction du taux de désabonnement des clients')

    def plot_confusion_matrix(y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Prédictions')
        ax.set_ylabel('Vraies valeurs')
        ax.set_title('Matrice de Confusion')
        return fig
    
    def display_model_info(title, model, X_test_data, y_test_data, labels_data):
        y_pred = model.predict(X_test_data)
        recall = recall_score(y_test_data, y_pred)
        f1 = f1_score(y_test_data, y_pred)

        st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 1.2em'>{title}</div>", unsafe_allow_html=True)

        y_pred = model.predict(X_test_data)

        cm_fig = plot_confusion_matrix(y_test_data, y_pred, labels_data)
        st.pyplot(cm_fig)
        plt.close(cm_fig)

        st.metric(label = 'F1-Score', value = np.round(f1,3), border = True)
        st.metric(label = 'Recall-Score (Churners)', value = np.round(recall,3), border = True)
        
    
    labels_churn = [0, 1]
    display_model_info('Meilleur modèle', churn_model, x_test, y_test, labels_churn)


# Partie 5 : Etude de cas
elif choix_partie == 'V - Etude de cas':
    st.title('V - Etude de cas : Prédiction et Explication du Désabonnement')

    st.markdown("""
    Dans cette section, vous pouvez simuler les caractéristiques d'un client et obtenir une prédiction de son risque de désabonnement.
    Plus important encore, nous utilisons les **valeurs SHAP** pour vous expliquer **quelles caractéristiques influencent le plus la prédiction** et dans quelle direction.
    """)

    # --- 3. Définir les features et leurs plages/options dynamiquement ---
    features_info = {}
    for col in all_original_features:
        if df_cleaned[col].dtype in ['int64', 'float64']:
            if df_cleaned[col].nunique() > 10:
                features_info[col] = {
                    'type': 'number_input',
                    'min': float(df_cleaned[col].min()),
                    'max': float(df_cleaned[col].max()),
                    'default': float(df_cleaned[col].mean()),
                    'step': 1.0,
                    'label': col
                }
            else:
                features_info[col] = {
                    'type': 'radio',
                    'options': sorted([str(x) for x in df_cleaned[col].unique()]),
                    'default': str(df_cleaned[col].mode()[0]),
                    'label': col
                }
        else: # Type object (chaîne de caractères), probablement catégorielle
            if df_cleaned[col].nunique() <= 5:
                features_info[col] = {
                    'type': 'radio',
                    'options': sorted([str(x) for x in df_cleaned[col].unique()]),
                    'default': str(df_cleaned[col].mode()[0]),
                    'label': col
                }
            else:
                features_info[col] = {
                    'type': 'selectbox',
                    'options': sorted([str(x) for x in df_cleaned[col].unique()]),
                    'default': str(df_cleaned[col].mode()[0]),
                    'label': col
                }

    # Corrections spécifiques pour des colonnes
    if 'tenure' in features_info:
        features_info['tenure'].update({'type': 'slider', 'min': 0, 'max': 72, 'default': 30, 'step': 1, 'label': 'Ancienneté (mois)'})
    if 'MonthlyCharges' in features_info:
        features_info['MonthlyCharges'].update({'type': 'slider', 'min': 18.0, 'max': 120.0, 'default': 50.0, 'step': 0.5, 'label': 'Frais Mensuels (€)'})
    if 'TotalCharges' in features_info:
        features_info['TotalCharges'].update({'type': 'number_input', 'min': 0.0, 'max': 9000.0, 'default': 1500.0, 'step': 10.0, 'label': 'Frais Totaux (€)'})
    if 'gender' in features_info:
        features_info['gender'].update({'type': 'radio', 'options': ['Male', 'Female'], 'default': 'Male', 'label': 'Genre'})
    if 'InternetService' in features_info:
        features_info['InternetService'].update({'type': 'selectbox', 'options': ['DSL', 'Fiber optic', 'No'], 'default': 'Fiber optic', 'label': 'Service Internet'})
    if 'Contract' in features_info:
        features_info['Contract'].update({'type': 'selectbox', 'options': ['Month-to-month', 'One year', 'Two year'], 'default': 'Month-to-month', 'label': 'Contrat'})
    if 'Partner' in features_info:
        features_info['Partner'].update({'type': 'radio', 'options': ['Yes', 'No'], 'default': 'No', 'label': 'Partenaire'})
    if 'Dependents' in features_info:
        features_info['Dependents'].update({'type': 'radio', 'options': ['Yes', 'No'], 'default': 'No', 'label': 'Dépendants'})


    # --- 4. Créer les widgets dans la barre latérale ---
    st.sidebar.header("Paramètres du Client Hypothétique")
    input_data = {}

    for feature_name in all_original_features:
        info = features_info.get(feature_name, {})

        if not info: # Déduire le type de widget si l'info n'est pas spécifiée
            if df_cleaned[feature_name].dtype in ['int64', 'float64']:
                if df_cleaned[feature_name].nunique() > 10:
                    info = { 'type': 'number_input', 'min': float(df_cleaned[feature_name].min()), 'max': float(df_cleaned[feature_name].max()), 'default': float(df_cleaned[feature_name].mean()), 'step': 1.0, 'label': feature_name }
                else:
                    info = { 'type': 'radio', 'options': sorted([str(x) for x in df_cleaned[feature_name].unique()]), 'default': str(df_cleaned[feature_name].mode()[0]), 'label': feature_name }
            else: # object (string)
                unique_values = df_cleaned[feature_name].unique()
                if len(unique_values) <= 5:
                    info = { 'type': 'radio', 'options': sorted([str(x) for x in unique_values]), 'default': str(df_cleaned[feature_name].mode()[0]), 'label': feature_name }
                else:
                    info = { 'type': 'selectbox', 'options': sorted([str(x) for x in unique_values]), 'default': str(df_cleaned[feature_name].mode()[0]), 'label': feature_name }

        widget_value = None
        if info['type'] == 'slider':
            widget_value = st.sidebar.slider(label=info['label'], min_value=info['min'], max_value=info['max'], value=info['default'], step=info['step'])
        elif info['type'] == 'number_input':
            widget_value = st.sidebar.number_input(label=info['label'], min_value=info['min'], max_value=info['max'], value=info['default'], step=info['step'])
        elif info['type'] == 'selectbox':
            widget_value = st.sidebar.selectbox(label=info['label'], options=info['options'], index=info['options'].index(info['default']))
        elif info['type'] == 'radio':
            widget_value = st.sidebar.radio(label=info['label'], options=info['options'], index=info['options'].index(info['default']))

        try:
            original_dtype = df_cleaned[feature_name].dtype
            if original_dtype == 'int64':
                input_data[feature_name] = int(widget_value)
            elif original_dtype == 'float64':
                input_data[feature_name] = float(widget_value)
            else:
                input_data[feature_name] = str(widget_value)
        except ValueError:
            st.error(f"Erreur de conversion de type pour la colonne '{feature_name}'. La valeur '{widget_value}' ne peut pas être convertie au type attendu.")
            st.stop()


    # --- 5. Construire le DataFrame pour la prédiction ---
    input_df = pd.DataFrame([input_data])
    input_df = input_df[all_original_features]


    # --- 6. Faire la prédiction et afficher les SHAP values pour l'état initial ---
    st.subheader("Situation Actuelle du Client")
    predict_button_label = "Prédire et Expliquer le Désabonnement (État Initial)"
    if st.button(predict_button_label, key='initial_predict_btn'):
        try:
            prediction_proba = churn_model.predict_proba(input_df)[0]
            prediction_class = churn_model.predict(input_df)[0]

            churn_probability = prediction_proba[1]
            non_churn_probability = prediction_proba[0]

            st.write(f"Probabilité de **Non-Désabonnement** : **`{non_churn_probability:.2%}`**")
            st.write(f"Probabilité de **Désabonnement (Churn)** : **`{churn_probability:.2%}`**")

            if prediction_class == 1:
                st.error("Ce client hypothétique est **susceptible de se désabonner**.")
            else:
                st.success("Ce client hypothétique est **peu susceptible de se désabonner**.")

            st.markdown("---")
            st.subheader("Explication de la Prédiction Actuelle (Valeurs SHAP)")
            st.markdown("""
            Le graphique ci-dessous (Force Plot SHAP) montre comment chaque caractéristique contribue
            à la prédiction de désabonnement pour ce client spécifique.
            * Les **valeurs rouges** augmentent la probabilité de désabonnement.
            * Les **valeurs bleues** diminuent la probabilité de désabonnement.
            """)

            instance_transformed = churn_model.named_steps['preprocessor'].transform(input_df)
            if 'selector' in churn_model.named_steps:
                instance_transformed = churn_model.named_steps['selector'].transform(instance_transformed)

            shap_values_instance = explainer.shap_values(instance_transformed)[0][:, 1]

            shap.force_plot(explainer.expected_value[1], shap_values_instance,
                            feature_names=display_feature_names_for_shap, matplotlib=True)
            fig_initial_shap = plt.gcf()
            fig_initial_shap.set_size_inches(18, 6) # Largeur augmentée pour Force Plot
            plt.tight_layout()
            st.pyplot(fig_initial_shap)
            plt.close(fig_initial_shap)

            st.markdown("---")
            st.write("Détails de l'instance client générée :")
            st.dataframe(input_df)

            st.session_state.initial_input_df = input_df
            st.session_state.initial_prediction_proba = prediction_proba
            st.session_state.initial_shap_values = shap_values_instance

        except Exception as e:
            st.error(f"Une erreur est survenue lors de la prédiction ou de l'explication SHAP pour l'état initial : {e}")
            st.write("Veuillez vérifier que les types de données et les noms de colonnes correspondent à ceux attendus par le modèle.")
            st.write(f"Colonnes attendues par le modèle (issues de df_cleaned): {all_original_features}")
            st.write(f"Colonnes générées par les inputs : {input_df.columns.tolist()}")

    st.markdown("---")
    st.subheader("Simulation d'Impact : Que se passe-t-il si nous changeons une caractéristique ?")

    if 'initial_input_df' in st.session_state and st.session_state.initial_input_df is not None:
        st.info("Sélectionnez une caractéristique ci-dessous pour simuler un changement et observer l'impact sur le risque de désabonnement.")

        modifiable_features = sorted([f for f in all_original_features if f not in ['customerID']])

        feature_to_modify = st.selectbox("Choisir la caractéristique à modifier :", modifiable_features, key='feature_select')

        if feature_to_modify:
            info_to_modify = features_info.get(feature_to_modify, {})

            st.write(f"Valeur actuelle de **{feature_display_names.get(feature_to_modify, feature_to_modify)}** : `{st.session_state.initial_input_df[feature_to_modify].iloc[0]}`")

            new_value = None
            if info_to_modify['type'] == 'slider':
                new_value = st.slider(label=f"Nouvelle valeur pour '{feature_display_names.get(feature_to_modify, feature_to_modify)}'", min_value=info_to_modify['min'], max_value=info_to_modify['max'], value=info_to_modify['default'], step=info_to_modify['step'], key=f'mod_slider_{feature_to_modify}')
            elif info_to_modify['type'] == 'number_input':
                new_value = st.number_input(label=f"Nouvelle valeur pour '{feature_display_names.get(feature_to_modify, feature_to_modify)}'", min_value=info_to_modify['min'], max_value=info_to_modify['max'], value=info_to_modify['default'], step=info_to_modify['step'], key=f'mod_number_{feature_to_modify}')
            elif info_to_modify['type'] == 'selectbox':
                new_value = st.selectbox(label=f"Nouvelle valeur pour '{feature_display_names.get(feature_to_modify, feature_to_modify)}'", options=info_to_modify['options'], index=info_to_modify['options'].index(info_to_modify['default']), key=f'mod_select_{feature_to_modify}')
            elif info_to_modify['type'] == 'radio':
                new_value = st.radio(label=f"Nouvelle valeur pour '{feature_display_names.get(feature_to_modify, feature_to_modify)}'", options=info_to_modify['options'], index=info_to_modify['options'].index(info_to_modify['default']), key=f'mod_radio_{feature_to_modify}')

            if st.button("Simuler l'impact", key='simulate_btn'):
                try:
                    modified_input_df = st.session_state.initial_input_df.copy()

                    original_dtype_mod = df_cleaned[feature_to_modify].dtype
                    if original_dtype_mod == 'int64':
                        modified_input_df[feature_to_modify] = int(new_value)
                    elif original_dtype_mod == 'float64':
                        modified_input_df[feature_to_modify] = float(new_value)
                    else:
                        modified_input_df[feature_to_modify] = str(new_value)

                    st.subheader(f"Résultat de la Simulation : '{feature_display_names.get(feature_to_modify, feature_to_modify)}' modifié")

                    prediction_proba_modified = churn_model.predict_proba(modified_input_df)[0]
                    prediction_class_modified = churn_model.predict(modified_input_df)[0]

                    churn_probability_modified = prediction_proba_modified[1]
                    non_churn_probability_modified = prediction_proba_modified[0]

                    st.markdown("#### Comparaison avant/après modification")

                    initial_churn_probability = st.session_state.initial_prediction_proba[1]
                    initial_non_churn_probability = st.session_state.initial_prediction_proba[0]
                    initial_prediction_class = np.argmax(st.session_state.initial_prediction_proba)

                    delta_churn_prob = churn_probability_modified - initial_churn_probability
                    delta_non_churn_prob = non_churn_probability_modified - initial_non_churn_probability

                    col_current, col_simulated, col_delta = st.columns(3)

                    with col_current:
                        st.metric(label="Probabilité Churn (Initial)", value=f"{initial_churn_probability:.2%}")
                        st.metric(label="Probabilité Non-Churn (Initial)", value=f"{initial_non_churn_probability:.2%}")
                    with col_simulated:
                        st.metric(label="Probabilité Churn (Simulé)", value=f"{churn_probability_modified:.2%}")
                        st.metric(label="Probabilité Non-Churn (Simulé)", value=f"{non_churn_probability_modified:.2%}")
                    with col_delta:
                        st.metric(label="Changement Churn Probabilité",
                                  value=f"{abs(delta_churn_prob):.2%}",
                                  delta=f"{delta_churn_prob:.2%}",
                                  delta_color="inverse" if delta_churn_prob > 0 else "normal")

                        st.metric(label="Changement Non-Churn Probabilité",
                                  value=f"{abs(delta_non_churn_prob):.2%}",
                                  delta=f"{delta_non_churn_prob:.2%}",
                                  delta_color="normal" if delta_non_churn_prob > 0 else "inverse")

                    if prediction_class_modified != initial_prediction_class:
                        if prediction_class_modified == 1:
                            st.warning(f"**Attention :** Le client est passé de **{'Non-Désabonnement'}** à **{'Désabonnement'}** après cette modification.")
                        else:
                            st.success(f"**Succès :** Le client est passé de **{'Désabonnement'}** à **{'Non-Désabonnement'}** après cette modification !")
                    else:
                        st.info("Le statut de désabonnement prédit n'a pas changé après cette modification.")

                    st.markdown("---")

                    st.write(f"Probabilité de **Non-Désabonnement (Modifiée)** : **`{non_churn_probability_modified:.2%}`**")
                    st.write(f"Probabilité de **Désabonnement (Churn) (Modifiée)** : **`{churn_probability_modified:.2%}`**")

                    if prediction_class_modified == 1:
                        st.error("Ce client simulé est **susceptible de se désabonner**.")
                    else:
                        st.success("Ce client simulé est **peu susceptible de se désabonner**.")

                    st.markdown("---")
                    st.subheader("Explication de la Prédiction Modifiée (Valeurs SHAP)")

                    instance_transformed_modified = churn_model.named_steps['preprocessor'].transform(modified_input_df)
                    if 'selector' in churn_model.named_steps:
                        instance_transformed_modified = churn_model.named_steps['selector'].transform(instance_transformed_modified)

                    shap_values_instance_modified = explainer.shap_values(instance_transformed_modified)[0][:, 1]

                    shap.force_plot(explainer.expected_value[1], shap_values_instance_modified,
                                    feature_names=display_feature_names_for_shap, matplotlib=True)
                    fig_modified_shap = plt.gcf()
                    fig_modified_shap.set_size_inches(18, 6) # Largeur augmentée
                    plt.tight_layout()
                    st.pyplot(fig_modified_shap)
                    plt.close(fig_modified_shap)

                    st.markdown("---")
                    st.write("Détails de l'instance client simulée :")
                    st.dataframe(modified_input_df)

                except Exception as e:
                    st.error(f"Une erreur est survenue lors de la simulation : {e}")
                    st.write("Veuillez vérifier les valeurs entrées et les types de données.")
    else:
        st.warning("Veuillez d'abord cliquer sur 'Prédire et Expliquer le Désabonnement (État Initial)' pour initialiser le client avant de lancer une simulation.")


# Partie 6 : Conclusion et Perspectives
elif choix_partie == 'VI - Conclusion et Perspectives':
    st.title('VI - Conclusion et Perspectives')
    st.markdown("---")
    st.markdown("#### Conclusion")
    st.markdown("""
    Cette exploration et modélisation du taux de désabonnement des clients de télécommunication a permis de mettre en lumière plusieurs aspects cruciaux. L'analyse des données brutes a révélé des défis initiaux concernant le type de certaines variables et la présence de quelques valeurs manquantes, qui ont été adressés par un pré-traitement ciblé.

    La visualisation des données a ensuite offert des insights précieux sur la distribution des différentes caractéristiques et leur relation avec la variable cible 'Churn'. Des tendances claires se sont dégagées, suggérant que certains facteurs comme la durée d'abonnement, les charges mensuelles et totales, ainsi que l'utilisation de certains services (comme le streaming de films) sont fortement corrélés au désabonnement.

    La comparaison de différents modèles de classification (Régression Logistique, Support Vector Machine, Arbre de Décision et Forêt Aléatoire) a permis d'évaluer leur performance dans la
    prédiction du désabonnement. La Forêt Aléatoire s'est distinguée comme le modèle le plus performant, grâce à sa capacité à gérer la complexité des données et à fournir des prédictions robustes.

    Enfin, l'intégration des explications SHAP dans l'application Streamlit a transformé un simple outil de prédiction en un puissant levier de compréhension. En permettant d'analyser l'impact de chaque caractéristique sur le risque de désabonnement, tant au niveau global du modèle qu'au niveau d'un client individuel, cette application offre des insights actionnables. La fonctionnalité de simulation "ce que si" permet aux entreprises de tester l'efficacité de potentielles stratégies de rétention avant de les déployer.

    En somme, cette approche combine la puissance du Machine Learning avec l'interprétabilité pour fournir un outil complet et pratique d'aide à la décision pour la fidélisation client.
    """)

    st.markdown("#### Perspectives Futures")
    st.markdown("""
    Pour l'avenir, plusieurs pistes d'amélioration pourraient être envisagées :
    * **Intégration de nouvelles données :** Inclure des données supplémentaires comme les interactions clients avec le service support, l'historique des pannes, ou des données socio-économiques pour affiner encore la précision du modèle.
    * **Modèles plus avancés :** Explorer des modèles plus complexes comme les réseaux de neurones profonds, tout en veillant à maintenir leur interprétabilité.
    * **Optimisation des stratégies de rétention :** Utiliser les informations SHAP pour concevoir des offres de rétention personnalisées et tester leur efficacité à travers des A/B tests.
    * **Suivi de l'impact des actions :** Mettre en place un système pour suivre si les actions de rétention basées sur ces prédictions réduisent effectivement le churn.
    * **Déploiement en production :** Mettre en place l'application dans un environnement de production avec des mises à jour automatiques des données et du modèle.
    """)