import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from typing import List, Dict, Any, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_curve, auc, recall_score, make_scorer, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer

from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import Pipeline as ImbPipeline

# --- Configuration et Constantes ---
sns.set_theme()
pd.set_option('display.max_columns', None)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


DF_PICKLE_FILE = 'df_telco_customer_churn.pkl'
X_TEST_FILE = 'x_test.pkl'
Y_TEST_FILE = 'y_test.pkl'
BEST_OVERALL_CHURN_MODEL_FILE = 'best_overall_churn_model.pkl'

TEST_SIZE = 0.2
RANDOM_STATE = 66
CHURN_THRESHOLD = 0.4
K_BEST_FEATURES_INITIAL = 25 # Cette constante n'est plus directement utilisée pour un plot initial, mais peut être conservée si 'all' est trop coûteux pour certains modèles dans les hyperparams
CV_FOLDS = 5

# --- Fonctions ---

def load_and_prepare_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Charge les données, effectue un nettoyage initial et sépare les features de la cible.
    """
    logging.info(f"Chargement et préparation des données depuis {file_path}")
    try:
        df = pd.read_csv(file_path, na_values=[' ', None])
    except FileNotFoundError:
        logging.error(f"Erreur: Le fichier '{file_path}' n'a pas été trouvé. Veuillez vérifier le chemin.")
        raise

    df.drop('customerID', axis=1, inplace=True)
    df = df.dropna(how='any')
    df.columns = df.columns.str.title()

    df.to_pickle(DF_PICKLE_FILE)

    df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1}).astype('int')

    target = df['Churn']
    features = df.drop(['Churn'], axis=1)

    logging.info("Données chargées et préparées avec succès.")
    return features, target, df

def get_column_types(features_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Identifie et retourne les noms des colonnes par type pour le prétraitement.
    """
    logging.info("Identification des types de colonnes.")
    numerical_cols = features_df.select_dtypes(['int', 'float']).drop('Seniorcitizen', axis=1).columns.tolist()
    
    binary_yes_no_cols = [col for col in features_df.columns
                          if features_df[col].nunique() == 2 and 'Yes' in features_df[col].unique() and 'No' in features_df[col].unique()]
    
    gender_col = ['Gender']

    all_object_cols = features_df.select_dtypes('object').columns.tolist()
    
    no_service_values = ['No internet service', 'No phone service']
    no_internet_service_cols = [col for col in all_object_cols
                                if any(val in features_df[col].unique() for val in no_service_values)]
    
    other_cat_cols = [col for col in all_object_cols
                      if col not in gender_col and col not in binary_yes_no_cols and col not in no_internet_service_cols]

    logging.info("Types de colonnes identifiés.")
    return numerical_cols, binary_yes_no_cols, gender_col, no_internet_service_cols, other_cat_cols

def create_preprocessor(x_train_df: pd.DataFrame,
                        numerical_cols: List[str],
                        binary_yes_no_cols: List[str],
                        gender_col: List[str],
                        no_internet_service_cols: List[str],
                        other_cat_cols: List[str]) -> ColumnTransformer:
    """
    Crée et configure le ColumnTransformer pour le prétraitement des données.
    """
    logging.info("Création du préprocesseur de données.")

    numerical_transformer = StandardScaler()
    binary_yes_no_transformer = OrdinalEncoder(categories=[['No', 'Yes']] * len(binary_yes_no_cols),
                                               handle_unknown='use_encoded_value', unknown_value=-1)
    gender_transformer = OrdinalEncoder(categories=[['Female', 'Male']],
                                        handle_unknown='use_encoded_value', unknown_value=-1)
    
    all_categories_no_internet = []
    for col in no_internet_service_cols:
        current_categories = [cat for cat in x_train_df[col].unique() if cat not in ['No internet service', 'No phone service']]
        all_categories_no_internet.append(current_categories)

    categorical_transformer_no_internet_service = OneHotEncoder(categories=all_categories_no_internet, handle_unknown='ignore', drop=None)
    categorical_transformer_other = OneHotEncoder(handle_unknown='ignore', drop='if_binary')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('bin_yes_no', binary_yes_no_transformer, binary_yes_no_cols),
            ('gender', gender_transformer, gender_col),
            ('cat_no_int', categorical_transformer_no_internet_service, no_internet_service_cols),
            ('cat_other', categorical_transformer_other, other_cat_cols)
        ],
        remainder='drop'
    )
    logging.info("Préprocesseur créé avec succès.")
    return preprocessor


def train_and_evaluate_models(x_train: pd.DataFrame, y_train: pd.Series,
                              x_test: pd.DataFrame, y_test: pd.Series,
                              preprocessor: ColumnTransformer,
                              models_to_evaluate: List[Tuple[str, Any]],
                              scorer: Any) -> Dict[str, Dict[str, Any]]:
    """
    Entraîne et évalue une liste de modèles de classification en utilisant RandomizedSearchCV
    pour l'optimisation des hyperparamètres. Stocke tous les résultats pour une analyse ultérieure.

    Args:
        x_train (pd.DataFrame): Données d'entraînement pour les caractéristiques.
        y_train (pd.Series): Cible d'entraînement.
        x_test (pd.DataFrame): Données de test pour les caractéristiques.
        preprocessor (ColumnTransformer): Le préprocesseur de données.
        models_to_evaluate (List[Tuple[str, Any]]): Liste de tuples (nom du modèle, instance du modèle).
        scorer (Any): La fonction de score à utiliser pour l'optimisation (ex: make_scorer(recall_score)).

    Returns:
        Dict[str, Dict[str, Any]]: Un dictionnaire contenant les meilleurs scores,
                                   les meilleurs paramètres, les meilleurs modèles,
                                   et les résultats d'évaluation détaillés pour chaque classifieur.
    """
    logging.info("Début de l'entraînement et de l'évaluation comparative des modèles.")
    results = {}
    under_sampler = RandomUnderSampler(random_state=RANDOM_STATE)

    # Note: transformed_feature_names n'est plus nécessaire ici car feature_importances est retiré.
    # Si d'autres analyses nécessitent les noms de features transformées, elles devront être recalculées localement.

    for name_model, model in models_to_evaluate:
        logging.info(f"\n--- Optimisation des hyperparamètres pour : {name_model} ---")
        
        model_pipeline_all = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('under_sampler', under_sampler),
            ('selector', SelectKBest()), # KBest sera toujours utilisé
            ('model', model)
        ])

        params = {}
        if name_model == 'Logistic Regression':
            params = {'selector__k': [10, 15, 20, 'all'],
                      'model__solver': ['liblinear', 'lbfgs'],
                      'model__C': [0.01, 0.1, 1]}
        elif name_model == 'Support Vector Machine':
            params = {'selector__k': [10, 15, 20, 'all'],
                      'model__C': [0.1, 1],
                      'model__gamma': ['scale', 'auto'],
                      'model__kernel': ['rbf']}
        elif name_model == 'Random Forest':
            params = {'selector__k': [10, 15, 20, 25, 30, 'all'],
                      'model__n_estimators': [100, 200, 300],
                      'model__max_depth': [5, 10, None],
                      'model__min_samples_split': [2, 5, 10],
                      'model__min_samples_leaf': [1, 2, 4]}
        elif name_model == 'Decision Tree':
            params = {'selector__k': [10, 15, 20, 'all'],
                      'model__criterion': ['gini', 'entropy'],
                      'model__max_depth': [None, 10],
                      'model__min_samples_split': [2, 5]}

        grid = RandomizedSearchCV(model_pipeline_all, params, cv=CV_FOLDS, scoring=scorer,
                                  n_jobs=-1, random_state=RANDOM_STATE, verbose=1)
        grid.fit(x_train, y_train)

        y_pred_proba = grid.predict_proba(x_test)
        y_preds = np.where(y_pred_proba[:, 1] > CHURN_THRESHOLD, 1, 0)
        
        # --- Stockage de tous les résultats pour ce modèle ---
        current_model_results = {
            'best_score': grid.best_score_,
            'best_params': grid.best_params_,
            'best_estimator': grid.best_estimator_, # Le pipeline optimisé
            'y_pred_proba': y_pred_proba,
            'y_preds': y_preds,
            'confusion_matrix': pd.crosstab(y_test, y_preds, rownames=['Réel'], colnames=['Prédit'], margins=True, margins_name='Total').to_markdown(),
            'classification_report': classification_report_imbalanced(y_test, y_preds),
            'roc_curve_data': {'fpr': roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)[0],
                               'tpr': roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)[1],
                               'auc': auc(roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)[0], roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)[1])}
        }
        
        # La logique de récupération des importances de caractéristiques pour Random Forest est retirée.
        
        results[name_model] = current_model_results

        logging.info(f"Meilleurs hyperparamètres pour {name_model}: {grid.best_params_}")
        logging.info(f"Meilleur score (validation croisée) pour {name_model}: {grid.best_score_:.4f}")
        logging.info(f"Modèle optimisé pour {name_model} entraîné et résultats stockés.")

    logging.info("Évaluation comparative des modèles terminée. Tous les résultats sont stockés.")
    return results

def main():
    """
    Fonction principale orchestrant l'ensemble du processus de modélisation du churn.
    """
    logging.info("Démarrage du script de prédiction du churn.")

    features, target, _ = load_and_prepare_data(RAW_DATA_FILE)
    x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE,
                                                        stratify=target)
    logging.info(f"Données divisées en ensembles d'entraînement ({len(x_train)} échantillons) "
                 f"et de test ({len(x_test)} échantillons).")
    
    joblib.dump(x_test, X_TEST_FILE)
    joblib.dump(y_test, Y_TEST_FILE)
    logging.info("Ensembles de test sauvegardés.")

    numerical_cols, binary_yes_no_cols, gender_col, no_internet_service_cols, other_cat_cols = get_column_types(features)
    preprocessor = create_preprocessor(x_train, numerical_cols, binary_yes_no_cols, gender_col, no_internet_service_cols, other_cat_cols)

    models_to_evaluate = [
        ('Logistic Regression', LogisticRegression(random_state=RANDOM_STATE, solver='liblinear')),
        ('Support Vector Machine', SVC(probability=True, random_state=RANDOM_STATE)),
        ('Decision Tree', DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ('Random Forest', RandomForestClassifier(random_state=RANDOM_STATE))
    ]
    
    recall_scorer = make_scorer(recall_score, pos_label=1)

    all_models_results = train_and_evaluate_models(x_train, y_train, x_test, y_test, preprocessor, models_to_evaluate, recall_scorer)

    logging.info("\n--- Résultats comparatifs finaux de tous les modèles ---")
    for name, result_info in all_models_results.items():
        logging.info(f"{name}: Meilleur score de rappel = {result_info['best_score']:.4f}, Meilleurs paramètres = {result_info['best_params']}")
    
    # --- Identification du meilleur modèle global ---
    logging.info("\n--- Identification du meilleur modèle global ---")

    best_overall_model_name = None
    best_overall_score = -1
    best_overall_model_info = None

    for model_name, model_info in all_models_results.items():
        if model_info['best_score'] > best_overall_score:
            best_overall_score = model_info['best_score']
            best_overall_model_name = model_name
            best_overall_model_info = model_info

    logging.info(f"Le meilleur modèle global est : {best_overall_model_name} avec un score de rappel de {best_overall_score:.4f}")
    logging.info(f"Ses meilleurs paramètres sont : {best_overall_model_info['best_params']}")

    joblib.dump(best_overall_model_info['best_estimator'], BEST_OVERALL_CHURN_MODEL_FILE)
    logging.info(f"Le meilleur modèle global (pipeline) sauvegardé sous '{BEST_OVERALL_CHURN_MODEL_FILE}'")

    # --- Analyse détaillée du modèle champion ---
    logging.info(f"\n--- Analyse détaillée du modèle champion : {best_overall_model_name} ---")

    print(f"\n--- Matrice de confusion pour {best_overall_model_name} (Seuil={CHURN_THRESHOLD}) ---")
    print(best_overall_model_info['confusion_matrix'])
    print(f"\n--- Rapport de classification pour {best_overall_model_name} (Seuil={CHURN_THRESHOLD}) ---")
    print(best_overall_model_info['classification_report'])

    fpr_best = best_overall_model_info['roc_curve_data']['fpr']
    tpr_best = best_overall_model_info['roc_curve_data']['tpr']
    roc_auc_best = best_overall_model_info['roc_curve_data']['auc']

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_best, tpr_best, color='blue', lw=2, label=f'{best_overall_model_name} (AUC = {roc_auc_best:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title(f'Courbe ROC Finale - Meilleur Modèle ({best_overall_model_name})')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'ROC_Curve_Best_Model_{best_overall_model_name.replace(" ", "_")}.png')
    plt.show()
    plt.close()

    # --- La logique de l'affichage de l'importance des caractéristiques est entièrement retirée ici ---
    # if 'feature_importances' in best_overall_model_info:
    # ... (le code pour le graphique des importances est supprimé) ...

    logging.info("Analyse du meilleur modèle terminée.")
    logging.info("Processus de modélisation du churn terminé avec succès.")


if __name__ == "__main__":
    main()