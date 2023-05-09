import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split

# Charger les scores de sentiment à partir du fichier CSV
df = pd.read_csv("scores_tweets.csv", sep="|")

# Préparation des données pour l'apprentissage en profondeur
# Nous ne considérons que les tweets positifs et négatifs
df = df[(df["sentiment_category_vader"] == "Positif") | (df["sentiment_category_vader"] == "Négatif")]

# Conversion des étiquettes en format numérique
df["sentiment_label"] = df["sentiment_category_vader"].apply(lambda x: 1 if x == "Positif" else 0)

# Extraction des tweets et de leurs étiquettes correspondantes
tweets = df["tweet_text"].values
labels = df["sentiment_label"].values

# Initialisation du Tokenizer
tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(tweets)

# Transformation des textes en séquences
sequences = tokenizer.texts_to_sequences(tweets)
data = pad_sequences(sequences, maxlen=50)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Création du modèle LSTM
model = Sequential()

model.add(Embedding(5000, 64, input_length=50)) # Couche d'Embedding
model.add(LSTM(64, dropout=0.2)) # Couche LSTM
model.add(Dense(1, activation='sigmoid')) # Couche de sortie

# Compilation du modèle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

# Affichage de l'historique de l'entraînement
print(history.history)

# Affichage de la précision pour la dernière époque
print(history.history['accuracy'][-1])

# Affichage de la perte pour la dernière époque
print(history.history['loss'][-1])

# Évaluation du modèle sur l'ensemble de test
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Perte sur le test set: {loss}')
print(f'Précision sur le test set: {accuracy}')

