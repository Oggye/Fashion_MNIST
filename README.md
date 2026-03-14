# Projet de Classification d'Images Fashion-MNIST avec MLP et CNN

Ce projet explore la classification d'images du jeu de données **Fashion-MNIST** à l'aide de réseaux de neurones profonds. Il compare deux architectures majeures :

1.  **Un Perceptron Multicouche (MLP)** : Implémenté comme solution de base, puis optimisé pour améliorer ses performances.
2.  **Un Réseau de Neurones Convolutionnel (CNN)** : Basé sur l'architecture historique **LeNet-5**, il est ensuite optimisé pour maximiser la précision de classification.

L'objectif principal est de démontrer l'impact des différents choix architecturaux et hyperparamètres sur les performances d'un modèle, en comparant une approche classique (MLP) à une approche moderne (CNN).

---

## Table des Matières

1.  [Aperçu du Projet](#aperçu-du-projet)
2.  [Jeu de Données : Fashion-MNIST](#jeu-de-données--fashion-mnist)
3.  [Architecture du Projet](#architecture-du-projet)
4.  [Méthodologie et Expériences](#méthodologie-et-expériences)
    *   [1. Modèle MLP de Base](#1-modèle-mlp-de-base)
    *   [2. Optimisation du MLP](#2-optimisation-du-mlp)
    *   [3. Modèle CNN de Base (LeNet-5)](#3-modèle-cnn-de-base-lenet-5)
    *   [4. Optimisation du CNN](#4-optimisation-du-cnn)
    *   [5. Modèle CNN Final](#5-modèle-cnn-final)
5.  [Résultats](#résultats)
6.  [Installation et Utilisation](#installation-et-utilisation)
7.  [Technologies Utilisées](#technologies-utilisées)
8.  [Conclusion](#conclusion)

---

## Aperçu du Projet

Ce projet de *machine learning* compare deux approches fondamentales pour la classification d'images :
*   **Le MLP (Perceptron Multicouche)** : Une architecture de réseau de neurones "fully connected" où chaque neurone d'une couche est connecté à tous les neurones de la couche suivante. Il ne tient pas compte de la structure spatiale des images.
*   **Le CNN (Réseau de Neurones Convolutionnel)** : Une architecture spécialisée pour les données spatiales (comme les images) qui utilise des couches de convolution pour apprendre automatiquement des caractéristiques hiérarchiques (bords, textures, formes).

Le projet suit un processus complet :
1.  **Analyse exploratoire** : Visualisation des données.
2.  **Développement de modèles de base** : MLP simple et CNN LeNet-5.
3.  **Optimisation** : Tests d'architectures, de fonctions d'activation, de taux d'apprentissage et de cycles d'entraînement pour améliorer les modèles de base.
4.  **Évaluation finale** : Entraînement des modèles optimisés et comparaison de leurs performances.

## Jeu de Données : Fashion-MNIST

**Fashion-MNIST** est un jeu de données moderne conçu comme un remplacement direct du classique MNIST. Il est constitué d'images en niveaux de gris de 28x28 pixels, représentant des articles de mode répartis en 10 classes.

*   **Taille de l'ensemble d'entraînement** : 60 000 images
*   **Taille de l'ensemble de test** : 10 000 images
*   **Classes** :
    0. T-shirt/top
    1. Pantalon
    2. Pull
    3. Robe
    4. Manteau
    5. Sandale
    6. Chemise
    7. Basket
    8. Sac
    9. Botte

Le notebook commence par charger et visualiser un échantillon de ces images.

## Architecture du Projet

Le projet est structuré dans un notebook Jupyter (`Fashion_MNIST.ipynb`) qui se décompose en plusieurs sections principales :

*   **Partie 1 : MLP (Perceptron Multicouche)**
    *   **1.1 Modèle de Base** : Mise en place d'un MLP simple (1 couche cachée).
    *   **1.2 Optimisation du MLP** :
        *   Tests de différentes architectures (2, 3, 4 couches cachées).
        *   Tests de différentes fonctions d'activation (`relu`, `sigmoid`, `tanh`).
        *   Tests de différents taux d'apprentissage.
        *   Tests de différents nombres d'époques.
    *   **1.3 Modèle MLP Final** : Entraînement du MLP avec la meilleure configuration trouvée et évaluation de ses performances.

*   **Partie 2 : CNN (Réseau de Neurones Convolutionnel)**
    *   **2.1 Préparation des Données pour CNN** : Remise en forme des données pour les couches `Conv2D`.
    *   **2.2 Modèle de Base (LeNet-5)** : Implémentation et évaluation du LeNet-5.
    *   **2.3 Optimisation du CNN** :
        *   Tests de différentes architectures basées sur LeNet-5 (avec plus de filtres, batch normalization, dropout).
        *   Tests de différents taux d'apprentissage.
        *   Tests de différents nombres d'époques.
    *   **2.4 Modèle CNN Final** : Entraînement du CNN optimisé et évaluation finale.

*   **Analyse des Résultats** :
    *   Comparaison des performances du MLP de base, du MLP amélioré et du CNN final.
    *   Visualisation de l'historique d'entraînement (accuracy et loss).
    *   Analyse des erreurs de classification et matrice de confusion.

## Méthodologie et Expériences

### 1. Modèle MLP de Base
*   **Architecture** : Une couche cachée avec 50 neurones et une fonction d'activation `relu`, suivie d'une couche de sortie avec 10 neurones et `softmax`.
*   **Résultat** : Ce modèle a servi de baseline, atteignant une précision de test d'environ **86.89%**.

### 2. Optimisation du MLP
*   **Optimisation de l'architecture** : Nous avons testé des architectures plus profondes. L'architecture **à 3 couches (128-64-32 neurones)** a offert le meilleur compromis performance/complexité, atteignant une précision de validation de 88.43%.
*   **Optimisation de la fonction d'activation** : La fonction `relu` a surperformé `sigmoid` et `tanh`, en raison de sa capacité à atténuer le problème de *vanishing gradient*.
*   **Optimisation du taux d'apprentissage** : Une recherche a montré qu'un taux d'apprentissage de **0.0005** pour l'optimiseur Adam était optimal, permettant une convergence stable.
*   **Optimisation du nombre d'époques** : L'entraînement avec 15 epochs a été jugé optimal, offrant le meilleur équilibre entre performance et surapprentissage.

### 3. Modèle CNN de Base (LeNet-5)
*   **Architecture** : Implémentation fidèle du LeNet-5 avec deux couches convolutionnelles (suivies de *average pooling*) et trois couches fully-connected.
*   **Résultat** : Ce modèle a atteint une précision de test de **87.65%**, surpassant déjà le MLP de base, démontrant la puissance des convolutions pour ce type de tâche.

### 4. Optimisation du CNN
*   **Optimisation de l'architecture** : Nous avons conçu et testé trois architectures améliorées. L'architecture "**LeNet-5 Enhanced**" (deux couches `Conv2D` avec plus de filtres) a donné les meilleurs résultats, atteignant une précision de validation de 91.83% avec `lr=0.001`.
*   **Optimisation du taux d'apprentissage** : Les tests ont confirmé qu'un taux d'apprentissage de **0.001** était le plus efficace pour cette architecture.
*   **Optimisation du nombre d'époques** : L'analyse des courbes d'apprentissage a montré que 15-20 epochs suffisent pour atteindre de très bonnes performances, au-delà desquelles le surapprentissage commence à apparaître.

### 5. Modèle CNN Final
Le modèle final combine tous les optima trouvés :
*   **Architecture** : LeNet-5 Enhanced (`Conv2D(64)` -> `MaxPool` -> `Conv2D(128)` -> `MaxPool` -> `Flatten` -> `Dense(256)` -> `Dense(10)`).
*   **Fonction d'activation** : `relu` pour les couches cachées.
*   **Optimiseur** : Adam avec un taux d'apprentissage de `0.001`.
*   **Nombre d'époques** : 15.

## Résultats

Le tableau ci-dessous résume les performances des différents modèles développés, de la baseline la plus simple au modèle final optimisé. On observe une nette progression à chaque étape d'optimisation.

| Modèle | Accuracy (Test) | Loss (Test) | Amélioration vs. MLP de Base |
| :--- | :--- | :--- | :--- |
| **MLP de Base** | 86.89% | 0.3599 | - |
| **MLP Amélioré** | 88.54% | 0.3336 | +1.89% |
| **CNN de Base (LeNet-5)** | 87.65% | 0.3299 | +0.76% |
| **CNN Final (Optimisé)** | **91.66%** | **0.4730** | **+4.77%** |

*   Le **MLP Amélioré** a montré une nette progression par rapport à la baseline, prouvant l'efficacité des optimisations hyperparamétriques et architecturales.
*   Le **CNN de Base (LeNet-5)** surpasse déjà légèrement le MLP de base, démontrant la puissance des convolutions pour la classification d'images, même sans optimisation poussée.
*   Le **CNN Final** a atteint une précision de **91.66%**, confirmant la supériorité des architectures convolutionnelles en capturant efficacement les caractéristiques spatiales. L'augmentation de la loss peut indiquer un début de surapprentissage ou une différence dans la métrique de loss au moment de la meilleure accuracy.
*   L'analyse de la **matrice de confusion** du CNN a révélé que les erreurs de classification se produisent principalement entre des classes visuellement similaires (ex : T-shirt vs. Chemise vs. Pull), ce qui est compréhensible.

## Installation et Utilisation

1.  **Clonez le dépôt** (si applicable) ou ouvrez le notebook.
2.  **Installez les dépendances nécessaires** :
    ```bash
    pip install tensorflow numpy matplotlib scikit-learn seaborn
