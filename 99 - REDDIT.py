import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Charger les publications Reddit à partir du fichier CSV
df = pd.read_csv("reddit_cc.csv", delimiter=",")

# Convertir les dates des publications en objets datetime
df['post_date_time'] = pd.to_datetime(df['timestamp'])

# Initialiser les analyseurs de sentiment
textblob_analyzer = TextBlob
vader_analyzer = SentimentIntensityAnalyzer()

# Fonction pour analyser le sentiment d'un texte avec TextBlob
def analyze_sentiment_textblob(text):
    analysis = textblob_analyzer(text)
    return analysis.sentiment.polarity

# Fonction pour analyser le sentiment d'un texte avec VADER
def analyze_sentiment_vader(text):
    return vader_analyzer.polarity_scores(text)['compound']

# Appliquer l'analyse de sentiment à chaque publication Reddit
df["sentiment_textblob"] = df["title"].apply(analyze_sentiment_textblob)
df["sentiment_vader"] = df["title"].apply(analyze_sentiment_vader)

# Fonction pour catégoriser le score de sentiment selon TextBlob
def categorize_sentiment_textblob(score):
    if score > 0:
        return "Positif"
    elif score < 0:
        return "Négatif"
    else:
        return "Neutre"

# Fonction pour catégoriser le score de sentiment selon VADER
def categorize_sentiment_vader(score):
    if score > 0.05:
        return "Positif"
    elif score < -0.05:
        return "Négatif"
    else:
        return "Neutre"

# Appliquer la catégorisation à chaque score de sentiment
df["sentiment_category_textblob"] = df["sentiment_textblob"].apply(categorize_sentiment_textblob)
df["sentiment_category_vader"] = df["sentiment_vader"].apply(categorize_sentiment_vader)

# Extraire les informations utiles
df = df[["title", "score", "id", "url", "comms_num", "created", "body", "post_date_time", "sentiment_textblob", "sentiment_vader", "sentiment_category_textblob", "sentiment_category_vader"]]

# Enregistrer les résultats dans un fichier CSV
df.to_csv("scores_reddit.csv", sep="|", index=False)
