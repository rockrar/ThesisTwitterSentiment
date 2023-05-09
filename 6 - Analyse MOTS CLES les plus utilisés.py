import pandas as pd
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Charger les tweets depuis le fichier CSV
df = pd.read_csv("crypto-query-tweets.csv", delimiter=",")

# Trier les tweets par ordre chronologique décroissant et ne garder que les 3000 plus récents
df = df.sort_values(by=["date_time"], ascending=False)[:6000]

# Fusionner tous les textes de tweets
all_tweets = ' '.join(df['tweet_text'].tolist())

# Tokeniser le texte
tokens = word_tokenize(all_tweets)

# Supprimer la ponctuation et les caractères spéciaux
words = [word.lower() for word in tokens if word.isalnum()]

# Supprimer les stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

# Compter la fréquence des mots
word_count = Counter(filtered_words)

# Trouver les mots clés les plus fréquents (exemple : les 10 mots les plus fréquents)
top_keywords = word_count.most_common(10)

# Initialiser l'analyseur de sentiment VADER
analyzer = SentimentIntensityAnalyzer()

# Calculer le sentiment global pour chaque mot-clé avec TextBlob et VADER
keyword_sentiments = {}
for keyword, count in top_keywords:
    sentiment_textblob = df[df['tweet_text'].str.contains(keyword, case=False)]['tweet_text'].apply(lambda x: TextBlob(x).sentiment.polarity).mean()
    sentiment_vader = df[df['tweet_text'].str.contains(keyword, case=False)]['tweet_text'].apply(lambda x: analyzer.polarity_scores(x)['compound']).mean()
    keyword_sentiments[keyword] = {
        'count': count,
        'sentiment_textblob': sentiment_textblob,
        'sentiment_vader': sentiment_vader
    }

# Afficher les résultats
print("Mots-clés les plus utilisés et leur sentiment global (TextBlob et VADER) :")
for keyword, info in keyword_sentiments.items():
    print(f"{keyword}: count = {info['count']}, sentiment_textblob = {info['sentiment_textblob']}, sentiment_vader = {info['sentiment_vader']}")
