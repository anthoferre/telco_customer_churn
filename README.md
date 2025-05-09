# Jeu de données sur le taux de désabonnement des clients de télécommunications

## 1. Description générale

Ce jeu de données fournit un aperçu complet des clients d'une entreprise de télécommunications, en se concentrant sur les facteurs qui influencent leur décision de quitter le service (désabonnement ou "churn"). Il contient des informations démographiques, des détails sur leurs comptes, les services qu'ils utilisent et leurs interactions avec l'entreprise. L'objectif principal de ce jeu de données est de permettre l'analyse du comportement des clients et la prédiction du taux de désabonnement, afin de développer des stratégies de fidélisation efficaces.

Le jeu de données comprend **7 043 lignes**, représentant chacune un client unique, et **21 colonnes**, décrivant diverses caractéristiques de ces clients. La variable cible est la colonne "**Churn**", qui indique si le client a quitté le service ou non.

## 2. Contenu du jeu de données

Les colonnes du jeu de données peuvent être regroupées dans les catégories suivantes :

* **Informations sur le client :**
    * `CustomerID` : Identifiant unique du client.
    * `Gender` : Sexe du client (homme ou femme).
    * `SeniorCitizen` : Indique si le client est une personne âgée (1 pour oui, 0 pour non).
    * `Partner` : Indique si le client a un partenaire (Yes ou No).
    * `Dependents` : Indique si le client a des personnes à charge (Yes ou No).

* **Informations sur le compte du client :**
    * `tenure` : Nombre de mois pendant lesquels le client est resté avec l'entreprise.
    * `Contract` : Type de contrat du client (Month-to-month, One year, Two year).
    * `PaperlessBilling` : Indique si le client a opté pour la facturation électronique (Yes ou No).
    * `PaymentMethod` : Mode de paiement du client (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).
    * `MonthlyCharges` : Montant facturé mensuellement au client.
    * `TotalCharges` : Montant total facturé au client jusqu'à présent.

* **Services souscrits par le client :**
    * `PhoneService` : Indique si le client a un service téléphonique (Yes ou No).
    * `MultipleLines` : Indique si le client a plusieurs lignes téléphoniques (Yes, No, No phone service).
    * `InternetService` : Type de service Internet du client (DSL, Fiber optic, No).
    * `OnlineSecurity` : Indique si le client a une sécurité en ligne (Yes, No, No internet service).
    * `OnlineBackup` : Indique si le client a une sauvegarde en ligne (Yes, No, No internet service).
    * `DeviceProtection` : Indique si le client a une protection de l'appareil (Yes, No, No internet service).
    * `TechSupport` : Indique si le client bénéficie d'une assistance technique (Yes, No, No internet service).
    * `StreamingTV` : Indique si le client regarde la télévision en streaming (Yes, No, No internet service).
    * `StreamingMovies` : Indique si le client regarde des films en streaming (Yes, No, No internet service).

* **Variable cible :**
    * `Churn` : Indique si le client s'est désabonné au cours du dernier mois (Yes ou No).

## 3. Cas d'utilisation et applications potentielles

Ce jeu de données offre de nombreuses possibilités d'analyse et d'application, notamment :

* **Analyse exploratoire des données (EDA) :** Comprendre la distribution des différentes caractéristiques, identifier les relations entre elles et visualiser les différences entre les clients qui se sont désabonnés et ceux qui sont restés.
* **Modélisation prédictive du taux de désabonnement :** Construire des modèles de classification (par exemple, régression logistique, arbres de décision, forêts aléatoires, réseaux neuronaux) pour prédire la probabilité qu'un client se désabonne.
* **Identification des facteurs clés de désabonnement :** Déterminer les variables qui ont le plus d'influence sur la décision d'un client de quitter le service.
* **Segmentation de la clientèle :** Regrouper les clients en fonction de leurs caractéristiques et de leur comportement pour adapter les stratégies de fidélisation.
* **Développement de stratégies de rétention ciblées :** Concevoir des interventions spécifiques pour les clients à haut risque de désabonnement.
* **Évaluation de l'impact des offres et des promotions :** Analyser si certaines offres ou promotions ont un effet sur la réduction du taux de désabonnement.

 
## 4. Références

[Lien vers le jeu de données sur Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
