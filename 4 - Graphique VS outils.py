import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Charger les scores de sentiment à partir du fichier CSV
df = pd.read_csv("scores_tweets.csv", sep="|")

# Calculer les statistiques pour TextBlob et VADER
mean_textblob = np.mean(df["sentiment_textblob"])
mean_vader = np.mean(df["sentiment_vader"])
std_textblob = np.std(df["sentiment_textblob"])
std_vader = np.std(df["sentiment_vader"])

print(f"TextBlob - Moyenne: {mean_textblob}, Écart-type: {std_textblob}")
print(f"VADER - Moyenne: {mean_vader}, Écart-type: {std_vader}")

# Créer un graphique de dispersion pour visualiser la corrélation entre TextBlob et VADER
plt.figure(figsize=(10, 6))
plt.scatter(df["sentiment_textblob"], df["sentiment_vader"], alpha=0.5)
plt.title("Corrélation entre les scores de sentiment TextBlob et VADER")
plt.xlabel("Scores de sentiment TextBlob")
plt.ylabel("Scores de sentiment VADER")
plt.show()

# Créer des histogrammes pour visualiser la distribution des scores de sentiment
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

axes[0].hist(df["sentiment_textblob"], bins=30, color='blue', alpha=0.5)
axes[0].set_title("Distribution des scores de sentiment TextBlob")
axes[0].set_xlabel("Scores de sentiment")
axes[0].set_ylabel("Nombre de tweets")

axes[1].hist(df["sentiment_vader"], bins=30, color='red', alpha=0.5)
axes[1].set_title("Distribution des scores de sentiment VADER")
axes[1].set_xlabel("Scores de sentiment")

plt.show()
