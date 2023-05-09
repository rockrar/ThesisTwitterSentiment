import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Charger les données du fichier CSV
df = pd.read_csv("scores_tweets.csv", sep='|')

# Liste des mots-clés les plus utilisés (à remplacer par votre liste)
top_keywords = ['crypto', 'https', 'nft', 'bitcoin', 'btc', 'eth', 'nfts', 'blockchain', 'cryptocurrency', 'done']

# Créer des colonnes pour chaque mot-clé indiquant si le mot-clé est présent dans le tweet ou non (1 si présent, 0 sinon)
for keyword in top_keywords:
    df[keyword] = df['tweet_text'].str.contains(keyword, case=False).astype(int)

# Extraire les scores de sentiment TextBlob et VADER
textblob_scores = df["sentiment_textblob"].values
vader_scores = df["sentiment_vader"].values

# Extraire les données de mots-clés
keywords_data = df[top_keywords].values

# Diviser les données en ensembles d'entraînement et de test (80% d'entraînement, 20% de test)
X_train, X_test, y_train_textblob, y_test_textblob = train_test_split(keywords_data, textblob_scores, test_size=0.2, random_state=42)
X_train, X_test, y_train_vader, y_test_vader = train_test_split(keywords_data, vader_scores, test_size=0.2, random_state=42)

# Créer un modèle de forêt aléatoire pour TextBlob
random_forest_textblob = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraîner le modèle en utilisant les données de mots-clés et les scores de sentiment TextBlob
random_forest_textblob.fit(X_train, y_train_textblob)

# Prédire les scores TextBlob en utilisant les données de mots-clés
textblob_predicted = random_forest_textblob.predict(X_test)

# Créer un modèle de forêt aléatoire pour VADER
random_forest_vader = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraîner le modèle en utilisant les données de mots-clés et les scores de sentiment VADER
random_forest_vader.fit(X_train, y_train_vader)

# Prédire les scores VADER en utilisant les données de mots-clés
vader_predicted = random_forest_vader.predict(X_test)

# Calculer le coefficient de détermination (R²) pour chaque modèle
r2_textblob = r2_score(y_test_textblob, textblob_predicted)
r2_vader = r2_score(y_test_vader, vader_predicted)

print(f"Coefficient de détermination (R²) pour TextBlob avec forêt aléatoire: {r2_textblob}")
print(f"Coefficient de détermination (R²) pour VADER avec forêt aléatoire: {r2_vader}")



import matplotlib.pyplot as plt

# Afficher les résultats pour TextBlob
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test_textblob)), y_test_textblob, color='blue', label='Valeurs réelles')
plt.scatter(range(len(textblob_predicted)), textblob_predicted, color='red', label='Prédictions')
plt.xlabel('Index des échantillons de test')
plt.ylabel('Scores de sentiment TextBlob')
plt.title('Comparaison des valeurs réelles et des prédictions pour TextBlob')
plt.legend()
plt.show()

# Afficher les résultats pour VADER
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test_vader)), y_test_vader, color='blue', label='Valeurs réelles')
plt.scatter(range(len(vader_predicted)), vader_predicted, color='red', label='Prédictions')
plt.xlabel('Index des échantillons de test')
plt.ylabel('Scores de sentiment VADER')
plt.title('Comparaison des valeurs réelles et des prédictions pour VADER')
plt.legend()
plt.show()
