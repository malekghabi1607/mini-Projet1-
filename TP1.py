# =========================================================
# TP1 — Science des Données : Titanic
# Objectif : explorer les données et analyser la survie
# GHABI Malek & TAKDJERAD Meriem
# =========================================================

# -----------------------------
# Préalables : packages
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Chargement des données
# -----------------------------
# Lecture du fichier CSV dans un DataFrame pandas
titanic = pd.read_csv("Donnees/titanic.csv")  # adapte le chemin si besoin
# On OBSERVE d’abord tout ce qu’on a...

print(titanic)
# -> Affiche tout le DataFrame.
# Comme il est grand, pandas montre seulement le début et la fin + les dimensions.

print(titanic.head(15))
# -> Affiche les 15 premières lignes.
# Sans 15, head() affiche 5 lignes par défaut.

print(titanic.tail())
# -> Affiche les 5 dernières lignes.
# Oui, on peut mettre un argument : tail(10) affiche les 10 dernières lignes.

print(titanic.describe())
# -> Affiche les statistiques descriptives des colonnes numériques
# (count, mean, std, min, 25%, 50%, 75%, max).

print(titanic.info())
# -> Donne un résumé général  / des informations de  :  nombre de lignes, colonnes, types de données et nombre de valeurs non nulles par colonne.

df = pd.DataFrame(titanic)
# -> Sert à transformer des données en DataFrame " table de données" .
# Mais ici ce n’est pas nécessaire car titanic est déjà un DataFrame.

print(df)
# -> Affiche le DataFrame


print("Dimensions n,p=", titanic.shape)
# -> Affiche les dimensions du tableau :
# n = nombre de lignes (891 passagers)
# p = nombre de colonnes (12 variables).
# La table possède 2 dimensions (lignes et colonnes).

print(titanic.describe(include='all'))
# -> Cette commande sert à résumer l’ensemble du jeu de données.
# Elle affiche des statistiques pour toutes les colonnes, numériques et catégorielles.
# Cela permet d’avoir une vue globale des données (valeurs principales, fréquences, etc.).

print(titanic.isnull().sum())
# Cette commande permet de compter le nombre de valeurs manquantes dans chaque colonne du tableau.
# Elle sert à identifier les variables incomplètes avant l’analyse.

print(titanic.columns)
# Cette instruction affiche la liste des noms des colonnes du jeu de données.
# Elle permet de savoir quelles variables sont disponibles (Age, Sex, Survived, etc.)
# et d’utiliser leurs noms exacts dans la suite de l’analyse sans faire d’erreurs.


for col in titanic.columns:
    print(col, titanic[col].nunique())
# Cette sortie montre le nombre de valeurs différentes dans chaque colonne.
# Elle permet d’identifier les variables catégorielles (peu de valeurs),
# les identifiants (valeurs presque uniques) et les variables numériques.
# Cela aide à choisir quelles colonnes sont pertinentes pour l’analyse.




plt.hist(titanic['Age'], edgecolor = 'black', color=(0.2,0.7,1))
plt.title('Distribution d’ages')
plt.xlabel('Age')
plt.ylabel('Compte')

# Avant d’afficher le graphique, on s’attend à observer la répartition des âges
# des passagers, avec une majorité d’adultes et moins d’enfants ou de personnes âgées

plt.show()

# Que pensez vous que ceci montrera ? (réponse avant d’habiliter plt.show())
#plt.hist(titanic['Survived'], edgecolor = 'black', color=(0.2,0.7,1))
#plt.title('Distribution de survivants')
#plt.xlabel('Age')
#plt.ylabel('Compte')

# Avant d’afficher le graphique, on s’attend à observer la répartition
# entre survivants (1) et non-survivants (0), avec probablement plus de décès.

#plt.show()

print(titanic.drop(columns=['Ticket', 'Cabin','Name','SibSp']))
# On a supprimé (à l’affichage) certaines colonnes jugées peu pertinentes
# pour notre question sur la survie des passagers.
# Cela permet de simplifier le tableau et de se concentrer sur les variables utiles.
# Les données originales ne sont pas modifiées.

print(titanic.groupby(['Sex','Survived']).count()['PassengerId'])
# Cette commande montre le nombre de passagers selon le sexe et la survie.
# On observe que la majorité des femmes ont survécu,
# alors que la majorité des hommes sont décédés.
# Cela suggère que le sexe a fortement influencé les chances de survie.

passengers = titanic.groupby('Sex')['PassengerId'].count()
#print(passengers)

# On regroupe les passagers par sexe et on compte le nombre total
# de passagers hommes et femmes.

survivors = titanic.groupby('Sex')['Survived'].sum()
#print(survivors)
# On utilise sum() car Survived vaut 1 pour un survivant et 0 sinon.
# La somme permet donc d’obtenir directement le nombre de survivants par sexe.


summary = pd.DataFrame({
    "Survivants": survivors,
    "Passagers": passengers,
    "%": round(100 * survivors / passengers, 1)
})
# On a créé un tableau récapitulatif contenant, pour chaque sexe,
# le nombre de passagers, le nombre de survivants et le pourcentage de survie.
# Cela permet de comparer facilement les chances de survie entre hommes et femmes.

#print(summary)

print(titanic['Survived'].sum() / titanic['PassengerId'].count())
# Cette valeur représente le taux de survie global des passagers du Titanic.
# On observe qu’environ 38 % des passagers ont survécu au naufrage.

summary[["Survivants", "Passagers"]].plot(kind='bar')
# Avant d’afficher le graphique, on s’attend à obtenir un diagramme en barres
# comparant, pour chaque sexe, le nombre total de passagers et le nombre de survivants.
# On s’attend à voir que les femmes ont proportionnellement plus de survivants que les hommes.

plt.xlabel('Sexe')
plt.ylabel('Total')
plt.title('Comparaison de la survie selon le sexe')
#plt.show()

#print("Si j'enlève toutes les lignes contenant un 'NaN': ", titanic.dropna().shape)
# Cette commande supprime toutes les lignes contenant au moins une valeur manquante
# et affiche les dimensions du tableau restant.

print("\nSi je n'enlève que les 'NaN' de la colonne Age : ",
      titanic.loc[titanic['Age'].notna(), :].shape)
# Ici, on supprime uniquement les lignes où l’âge est manquant,
# ce qui permet de conserver plus de données que la méthode précédente.

titanic['Adult'] = titanic['Age'] >= 18
# On crée une nouvelle variable booléenne qui indique
# si le passager est adulte (âge >= 18) ou non.
# Cela facilite l’analyse de la survie selon l’âge.

#print(titanic.head())
# Affiche les premières lignes pour vérifier la création de la nouvelle colonne.


titanic_filt_age = titanic.loc[titanic['Age'].notna(), :]
passengers = titanic_filt_age.groupby(['Adult','Sex']).count()['PassengerId']
print(passengers)
# Cette sortie montre le nombre de passagers dont l’âge est connu,
# répartis selon l’âge (adulte ou non) et le sexe.
# On observe qu’il y a peu d’enfants par rapport aux adultes,
# et que les hommes adultes sont beaucoup plus nombreux que les femmes adultes.


survivors = titanic_filt_age.groupby(['Adult','Sex'])['Survived'].sum()
# On regroupe les passagers selon l’âge (adulte ou non) et le sexe,
# puis on calcule le nombre de survivants dans chaque groupe.
# La somme est utilisée car Survived vaut 1 pour un survivant et 0 sinon.

print(survivors)




passengers = titanic_filt_age.groupby(['Adult','Sex'])['PassengerId'].count()
summary = pd.DataFrame({"Survivants": survivors,
"Passagers": passengers,
"%": round(survivors/passengers*100, 1)})
summary.index=['Girl','Boy','Woman','Man']
print(summary)
# On représente ici un barplot car on a des catégories
summary.plot(kind='bar')
plt.xlabel("Personnes classées selon l'age et le sexe")
plt.ylabel('Total')
plt.title("Comparaison de la survie selon l'age et le sexe");
#plt.show()
# Ce tableau montre la survie selon l’âge et le sexe.
# On observe que les femmes adultes ont le taux de survie le plus élevé (77.2 %),
# tandis que les hommes adultes ont le plus faible (17.7 %).
# Les enfants survivent mieux que les hommes adultes,
# et les filles ont un taux de survie plus élevé que les garçons.



# Conclusions :
# Observation :
# Environ 38.4 % des passagers ont survécu au naufrage.
# Plus précisément, environ 74.2 % des femmes ont survécu
# contre seulement 18.9 % des hommes.
# Il y avait plus d’hommes que de femmes sur le paquebot.

# Interprétation :
# Les femmes ont eu plus de chances de survivre que les hommes.

# Oui, nous sommes d’accord avec cette interprétation,
# car les résultats montrent un écart très important
# entre le taux de survie des femmes et celui des hommes.


# ----- ETUDE SELON LE PORT
print(f'Nombre de ports:',titanic['Embarked'].nunique())
print(f'Liste des ports:',titanic.loc[:,'Embarked'].unique())
# Ces commandes montrent qu’il existe 3 ports d’embarquement différents :
# S (Southampton), C (Cherbourg) et Q (Queenstown).
# On remarque également la présence de valeurs manquantes (NaN) dans la colonne Embarked.
print (titanic.groupby(['Embarked']).count() )
# Ici, on regroupe les passagers selon leur port d’embarquement
# et on compte le nombre de passagers pour chaque port.
# On observe que la majorité des passagers ont embarqué à Southampton (S),
# suivi de Cherbourg (C), puis de Queenstown (Q).


#--- CHOIX
titanic["Embarked"] = titanic["Embarked"].fillna('S')
print(titanic["Embarked"])
# Ici, on remplace les valeurs manquantes (NaN) de la colonne Embarked
# par la valeur 'S', correspondant au port de Southampton.
# Ce choix est justifié par le fait que Southampton est le port
# d’embarquement majoritaire. Il s’agit toutefois d’une hypothèse
# qui peut légèrement influencer les résultats.



survivors_per_port = titanic.groupby('Embarked')['Survived'].sum()
# On regroupe les passagers par port d’embarquement
# et on calcule le nombre de survivants pour chaque port.
passengers_per_port = titanic.groupby('Embarked')['PassengerId'].count()
# On regroupe les passagers par port d’embarquement
# et on compte le nombre total de passagers pour chaque port



comparaison_port_survie = pd.DataFrame({"Survivants": survivors_per_port,
"Passagers": passengers_per_port,
"%":
round(survivors_per_port/passengers_per_port*100, 1)})
print(comparaison_port_survie)
# Ici, on crée un tableau récapitulatif qui compare, pour chaque port d’embarquement,
# le nombre de survivants, le nombre total de passagers
# et le pourcentage de survie.

# Quest-ce qu’on affiche ?
comparaison_port_survie.plot(kind='bar')
plt.xlabel("Port d'Embarquement")
plt.ylabel("Nombre d'individus")
plt.title('Comparaison de survie selon le port')
#plt.show()

# Ce graphique affiche un diagramme en barres montrant,
# pour chaque port d’embarquement (C, Q, S),
# le nombre de passagers, le nombre de survivants
# ainsi que le pourcentage de survie.


# 2 HYPOTHESES
# Avant d’explorer les données, on peut penser qu’il y a plus de femmes à Cherbourg,
# car ce port présente un taux de survie plus élevé.

# Tester l'Hypothèse 1: Il y a plus de femmes à Cherbourg
female_per_port = titanic[titanic['Sex']=='female'].groupby('Embarked')['PassengerId'].count()
male_per_port = titanic[titanic['Sex']=='male'].groupby('Embarked')['PassengerId'].count()

pd.DataFrame({
    "Female": female_per_port,
    "Male": male_per_port,
    "Total": passengers_per_port,
    "% Female": female_per_port / passengers_per_port
})

print(female_per_port)
print(male_per_port)
# Après observation des résultats, on constate qu’il y a
# plus de femmes ayant embarqué à Cherbourg (73)
# qu’à Queenstown (36).
# Cette comparaison est faite à partir du nombre total de passagères par port.
# L’hypothèse 1 est donc vérifiée.


# Tester l’Hypothese 2
survivors_per_class = titanic.groupby('Pclass')['Survived'].sum()
passengers_per_class = titanic['Pclass'].value_counts()

print(
    pd.DataFrame({
        "Survivants": survivors_per_class,
        "Passagers": passengers_per_class,
        "%": round(survivors_per_class/passengers_per_class*100, 1)
    })
)
# Après observation des résultats, on constate que le taux de survie
# est le plus élevé en première classe, intermédiaire en deuxième classe
# et le plus faible en troisième classe.
# Il existe donc une corrélation entre la classe du billet
# et la probabilité de survie.




# Explorons encore plus...
# Explorons encore plus...
# Analyse de la répartition des classes selon le port d’embarquement

# Nombre de passagers de 1ère classe par port
pclass1_per_port = titanic[titanic['Pclass'] == 1].groupby('Embarked')['PassengerId'].count()

# Nombre de passagers de 2ème classe par port
pclass2_per_port = titanic[titanic['Pclass'] == 2].groupby('Embarked')['PassengerId'].count()

# Nombre de passagers de 3ème classe par port
pclass3_per_port = titanic[titanic['Pclass'] == 3].groupby('Embarked')['PassengerId'].count()

# Tableau récapitulatif
print(
    pd.DataFrame({
        'Classe 1': pclass1_per_port,
        'Classe 2': pclass2_per_port,
        'Classe 3': pclass3_per_port,
        'Passengers': passengers_per_port,
        '% Classe 1': round(pclass1_per_port / passengers_per_port * 100, 1)
    })
)
print(pclass1_per_port)
print(pclass2_per_port)
print(pclass3_per_port)
# Ce tableau analyse la répartition des classes sociales selon le port d’embarquement.
# On observe que Cherbourg (C) regroupe beaucoup de passagers de première classe,
# ce qui suggère une population globalement plus aisée.
# À l’inverse, Queenstown (Q) est majoritairement composé de passagers
# de troisième classe, principalement des migrants.
# Southampton (S) présente une situation intermédiaire,
# mais reste dominé par la troisième classe.
# La classe sociale est donc fortement liée au port d’embarquement.

# -----------------------------------
# Conversion CSV -> JSON
# -----------------------------------
titanic.to_json("Donnees/titanic.json", orient="records", indent=2)

# Le fichier titanic.json est créé dans le dossier Donnees


# -----------------------------------
# Conversion CSV -> XML
# -----------------------------------
titanic.to_xml("Donnees/titanic.xml", index=False)

# Le fichier titanic.xml est créé dans le dossier Donnees



#Cette commande crée une archive tar.gz contenant le code Python et les fichiers de données.
#tar -czvf TP1_titanic.tar.gz TP1.py Donnees/


#Cette commande crée une archive tar.bz2, utilisant une compression plus forte que gzip.
#tar -cjvf TP1_titanic.tar.bz2 TP1.py Donnees/


#Vérifier la taille des archives:
#ls -lh TP1_titanic.tar.gz TP1_titanic.tar.bz2


# Après vérification de la taille des archives, on constate que l’archive
# au format .tar.bz2 (46 Ko) est plus petite que l’archive au format .tar.gz (88 Ko).
# Cela s’explique par le fait que l’algorithme de compression bzip2
# offre une meilleure compression que gzip.
# En revanche, gzip est généralement plus rapide à la compression.
# Dans notre cas, le format .tar.bz2 est préféré car il permet
# de réduire davantage la taille de l’archive, ce qui facilite
# le stockage et le transfert des fichiers.