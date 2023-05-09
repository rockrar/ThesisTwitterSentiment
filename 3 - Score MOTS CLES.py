import pandas as pd

# Charger les résultats à partir du fichier CSV
df = pd.read_csv("scores_tweets.csv", sep='|')

# Liste des mots-clés
keywords = ['#crypto', '#bitcoin', '#Crypto', '#NFT', 'scam', 'id', 'risk']

# Fonction pour vérifier si un mot-clé est présent dans le texte du tweet
def contains_keyword(text, keyword):
    return keyword.lower() in text.lower()

# Calculer les statistiques pour chaque outil
for tool in ['textblob', 'vader']:
    print(f"Statistiques pour {tool.capitalize()} :")
    print(f"Moyenne : {df[f'sentiment_{tool}'].mean()}")
    print(f"Médiane : {df[f'sentiment_{tool}'].median()}")
    print(f"Écart-type : {df[f'sentiment_{tool}'].std()}")
    print()

    # Calculer les statistiques pour chaque mot-clé
    for keyword in keywords:
        print(f"Statistiques pour {tool.capitalize()} - {keyword.capitalize()} :")
        keyword_df = df[df['tweet_text'].apply(lambda x: contains_keyword(x, keyword))]
        print(f"Moyenne : {keyword_df[f'sentiment_{tool}'].mean()}")
        print(f"Médiane : {keyword_df[f'sentiment_{tool}'].median()}")
        print(f"Écart-type : {keyword_df[f'sentiment_{tool}'].std()}")
        print()
