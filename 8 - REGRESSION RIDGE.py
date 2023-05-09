import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Charger les données du fichier CSV
df = pd.read_csv("scores_tweets.csv", sep='|')

# Extraire les scores de sentiment TextBlob et VADER
textblob_scores = df["sentiment_textblob"].values.reshape(-1, 1)
vader_scores = df["sentiment_vader"].values

# Diviser les données en ensembles d'entraînement et de test (80% pour l'entraînement, 20% pour le test)
X_train, X_test, y_train_textblob, y_test_textblob = train_test_split(textblob_scores, textblob_scores, test_size=0.2, random_state=42)
_, _, y_train_vader, y_test_vader = train_test_split(textblob_scores, vader_scores, test_size=0.2, random_state=42)

# Créer un modèle de régression Ridge
ridge_model_textblob = Ridge(alpha=1.0)
ridge_model_vader = Ridge(alpha=1.0)

# Entraîner les modèles en utilisant les scores de TextBlob et VADER
ridge_model_textblob.fit(X_train, y_train_textblob)
ridge_model_vader.fit(X_train, y_train_vader)

# Prédire les scores TextBlob et VADER en utilisant les modèles Ridge
y_pred_textblob = ridge_model_textblob.predict(X_test)
y_pred_vader = ridge_model_vader.predict(X_test)

# Calculer les coefficients de détermination (R²) pour les modèles Ridge
r2_textblob = r2_score(y_test_textblob, y_pred_textblob)
r2_vader = r2_score(y_test_vader, y_pred_vader)

print(f"Coefficient de détermination (R²) pour TextBlob avec régression Ridge: {r2_textblob}")
print(f"Coefficient de détermination (R²) pour VADER avec régression Ridge: {r2_vader}")

# Afficher la régression Ridge pour TextBlob
plt.scatter(X_test, y_test_textblob, color="blue", label="Données réelles")
plt.plot(X_test, y_pred_textblob, color="red", label="Régression Ridge")
plt.xlabel("Scores de sentiment TextBlob (test)")
plt.ylabel("Scores de sentiment TextBlob (prédiction)")
plt.legend()
plt.title("Régression Ridge pour TextBlob")
plt.show()

# Afficher la régression Ridge pour VADER
plt.scatter(X_test, y_test_vader, color="blue", label="Données réelles")
plt.plot(X_test, y_pred_vader, color="red", label="Régression Ridge")
plt.xlabel("Scores de sentiment TextBlob (test)")
plt.ylabel("Scores de sentiment VADER (prédiction)")
plt.legend()
plt.title("Régression Ridge pour VADER")
plt.show()
