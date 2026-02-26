# TP3 - Statistiques Descriptives et Analyse en Composantes Principales

Travaux pratiques de statistiques descriptives avec Python et choix des axes factoriels,
réalisé sur le jeu de données des Iris de Fisher.

---

## Contexte

Les Iris de Fisher sont un jeu de données de référence introduit en 1933 par le
statisticien Ronald Aylmer Fisher. Il contient des mesures morphologiques sur 3 espèces
de fleurs : Iris setosa, Iris versicolor et Iris virginica.

Les 4 variables mesurées sont :

| Variable | Description        | Unité |
|----------|--------------------|-------|
| SE_L     | Longueur du sépale | mm    |
| SE_W     | Largeur du sépale  | mm    |
| PE_L     | Longueur du pétale | mm    |
| PE_W     | Largeur du pétale  | mm    |

---

## Structure du projet

```
tp3/
├── tp3_iris.py                   # Script principal Python
├── IRIS.xlsx                     # Données source (ne pas déplacer)
├── README.md                     
└── tp3_output/                   # Généré automatiquement à l'exécution
    ├── output.txt                               # Output terminal complet
    ├── hist_density_SE_L_global.png             # Histogramme + densité globale
    ├── hist_density_SE_L_par_espece.png         # Histogramme + densité par espèce
    ├── boxplots_especes.png                     # Boxplots comparatifs
    ├── scatter_SE_L_PE_L.png                    # Nuage de points SE_L vs PE_L
    ├── scatter_matrix.png                       # Matrice des nuages de points
    ├── correlation_heatmap.png                  # Heatmap de corrélation
    ├── scree_plot.png                           # Scree plot (valeurs propres)
    ├── cercle_correlation.png                   # Cercle de corrélation ACP
    └── projection_individus_plan_factoriel.png  # Projection ACP
```

Le dossier `tp3_output/` est créé automatiquement par le script. Il n'est pas nécessaire
de le créer manuellement.

---

## Prérequis

Python 3.10 ou supérieur.

Installer les dépendances avec :

```bash
pip install pandas numpy matplotlib openpyxl
```

Aucune autre dépendance externe n'est requise. 

---

## Lancer le script

Placer `tp3_iris.py` et `IRIS.xlsx` dans le même dossier, puis exécuter :

```bash
python3 tp3_iris.py
```

Tous les graphiques et l'output terminal seront sauvegardés dans `tp3_output/`.

---

## Contenu du TP

### 1. Chargement et description des données

- Chargement du fichier `IRIS.xlsx` avec reconstruction de l'index complet
  au format `i001_SETOSA`, `i027_VIRGINICA`, etc.
- Statistiques descriptives globales et par espèce
- Interprétation des distributions via moyenne et écart-type

### 2. Graphiques

- Histogramme et courbe de densité (KDE) d'une variable, globalement puis par espèce
- Boxplots comparatifs des 3 espèces pour les 4 variables
- Nuage de points pour une paire de variables
- Matrice complète des nuages de points pour toutes les paires de variables

### 3. Choix des axes factoriels

- Calcul et visualisation de la matrice de corrélation
- Calcul des valeurs propres et vecteurs propres via `numpy.linalg.eigh`
- Scree plot avec critère de Kaiser
- Pourcentage de variance expliquée par axe

### 4. Coordonnées et qualité de représentation

- Projection de tous les individus sur les axes principaux
- Coordonnées de l'individu `i027_VIRGINICA` sur PC1 et PC2
- Identification de la fleur la plus contributrice sur PC1
- Calcul du cos2 (qualité de représentation) individuelle et globale
- Cercle de corrélation et projection sur le plan factoriel PC1-PC2

---

## Résultats principaux

- PC1 et PC2 expliquent 94.64% de la variance totale (critère de Kaiser retient 2 axes)
- La dimension du sous-espace vectoriel de projection est 2
- i027_VIRGINICA : PC1 = -1.2789, PC2 = -0.0769, cos2 = 91.1% (bien représentée)
- Fleur la plus contributrice sur PC1 : i010_SETOSA (PC1 = 3.6495)
- Corrélation PC1/PC2 = 0.000000 (orthogonalité confirmée)
- PE_L et PE_W sont les variables les plus discriminantes entre espèces

---

## Interprétation des axes

PC1 représente la taille globale de la fleur. Il est fortement corrélé avec PE_L, PE_W
et SE_L dans le même sens. Il sépare nettement Iris setosa (petits pétales) des deux
autres espèces.

PC2 représente un contraste entre la largeur du sépale et la taille du pétale. Il est
corrélé positivement avec SE_W et négativement avec PE_L et PE_W. Il permet de distinguer
partiellement Versicolor et Virginica.
