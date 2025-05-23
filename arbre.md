```mermaid
graph TD
    A[Démarrage : Quel est votre objectif principal ?] --> B{Souhaitez-vous prédire une valeur (ex: prix, température) ?}
    B -- Oui --> C[Problème de Régression]
    B -- Non --> D{Souhaitez-vous classer des données en catégories (ex: spam/non-spam, race d'animal) ?}
    D -- Oui --> E[Problème de Classification]
    D -- Non --> F{Souhaitez-vous regrouper des données similaires ou trouver des patterns (sans étiquettes) ?}
    F -- Oui --> G[Problème Non Supervisé]
    F -- Non --> H{Autre objectif spécifique (réduction, génération, recommandation...) ?}
    H -- Oui --> I[Objectif Spécifique]

    %% Régression
    C --> C1{Interprétabilité cruciale ET données non linéaires/complexes ?}
    C1 -- Oui --> C2{Préférez la simplicité et la visualisation ?}
    C2 -- Oui --> C3[Arbres de Décision (Régression)]
    C2 -- Non --> C4{Recherchez la performance et la robustesse ?}
    C4 -- Oui --> C5[Forêts Aléatoires (Régression)]
    C5 -- Non --> C6[Gradient Boosting (XGBoost, LightGBM, CatBoost)]
    C1 -- Non --> C7{Relation linéaire simple ?}
    C7 -- Oui --> C8[Régression Linéaire / Régression Polynomiale]
    C7 -- Non --> C9{Performance maximale et données complexes/hautes dim ?}
    C9 -- Oui --> C10{Très grandes données et GPU disponibles ?}
    C10 -- Oui --> C11[Deep Learning (MLP)]
    C10 -- Non --> C12[Gradient Boosting / SVR]

    %% Classification
    E --> E1{Type de données prédominant ?}
    E1 -- Images/Vidéos --> E2[Deep Learning (CNN)]
    E1 -- Texte/Séries Temporelles --> E3[Deep Learning (RNN/LSTM/GRU)]
    E1 -- Tabulaires --> E4{Quantité de données très grande ET GPU disponibles ?}
    E4 -- Oui --> E5[Deep Learning (MLP)]
    E4 -- Non --> E6{Interprétabilité cruciale OU faibles données ?}
    E6 -- Oui --> E7[Régression Logistique / Arbres de Décision / Naïve Bayes]
    E6 -- Non --> E8{Performance maximale recherchée ?}
    E8 -- Oui --> E9[Forêts Aléatoires / Gradient Boosting / SVM / Méthodes d'Ensemble]
    E8 -- Non --> E10[K-NN / Régression Logistique]

    %% Non Supervisé
    G --> G1{Quel type de problème non supervisé ?}
    G1 -- Clustering (Regroupement) --> G2{Nombre de clusters connu à l'avance ?}
    G2 -- Oui --> G3[K-Means]
    G2 -- Non --> G4{Clusters de formes complexes / détection de bruit ?}
    G4 -- Oui --> G5[DBSCAN]
    G4 -- Non --> G6[Clustering Hiérarchique]
    G1 -- Réduction de Dimensionnalité --> G7{Souhaitez-vous préserver la variance linéaire ?}
    G7 -- Oui --> G8[PCA (Analyse en Composantes Principales)]
    G7 -- Non --> G9[Auto-encodeurs (DL)]
    G1 -- Détection d'Anomalies --> G10{Anomalies isolées ou frontières denses ?}
    G10 -- Isolées --> G11[Isolation Forest]
    G10 -- Frontière dense --> G12[One-Class SVM]

    %% Objectifs Spécifiques
    I --> I1{Quel objectif spécifique ?}
    I1 -- Génération de données --> I2[GANs / Auto-encodeurs variationnels]
    I1 -- Recommandation --> I3{Basé sur utilisateurs similaires OU contenu ?}
    I3 -- Utilisateurs --> I4[Filtrage Collaboratif]
    I3 -- Contenu --> I5[Filtrage Basé sur le Contenu]
    I1 -- Modélisation de dépendances/diagnostic --> I6[Réseaux Bayésiens]

    C3 -.-> Fin
    C6 -.-> Fin
    C8 -.-> Fin
    C11 -.-> Fin
    C12 -.-> Fin
    E2 -.-> Fin
    E3 -.-> Fin
    E5 -.-> Fin
    E7 -.-> Fin
    E9 -.-> Fin
    E10 -.-> Fin
    G3 -.-> Fin
    G5 -.-> Fin
    G6 -.-> Fin
    G8 -.-> Fin
    G9 -.-> Fin
    G11 -.-> Fin
    G12 -.-> Fin
    I2 -.-> Fin
    I4 -.-> Fin
    I5 -.-> Fin
    I6 -.-> Fin

    subgraph Modèles Recommandés
        Fin[Modèle(s) / Famille(s) de Modèles Recommandé(s)]
    end