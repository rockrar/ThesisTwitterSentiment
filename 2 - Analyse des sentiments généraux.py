import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Charger les scores de sentiment à partir du fichier CSV
df = pd.read_csv("scores_tweets.csv", sep="|")

# Calculer les statistiques pour TextBlob et VADER
textblob_stats = df["sentiment_textblob"].describe()
vader_stats = df["sentiment_vader"].describe()

# Afficher les statistiques
print("\nStatistiques pour TextBlob :\n")
print(textblob_stats)
print("\nStatistiques pour VADER :\n")
print(vader_stats)

# Calculer et afficher les catégories de sentiment pour TextBlob et VADER
textblob_sentiment_counts = df["sentiment_category_textblob"].value_counts()
vader_sentiment_counts = df["sentiment_category_vader"].value_counts()

print("\nRépartition des catégories de sentiment pour TextBlob :\n")
print(textblob_sentiment_counts)
print("\nRépartition des catégories de sentiment pour VADER :\n")
print(vader_sentiment_counts)
