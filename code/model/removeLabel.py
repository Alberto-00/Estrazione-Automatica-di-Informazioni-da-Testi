import pandas as pd
from sklearn.model_selection import train_test_split

# Carica il file CSV
file_path = '../dataset/Corona_NLP_train_clean_start.csv'  # Sostituisci con il percorso reale del tuo file CSV
df = pd.read_csv(file_path)

# Sostituisci 'Extremely Positive' con 'Positive' e 'Extremely Negative' con 'Negative'
df['Sentiment'] = df['Sentiment'].replace({'Extremely Positive': 'Positive', 'Extremely Negative': 'Negative'})

# Dividi il dataframe in base al sentiment
positive_df = df[df['Sentiment'] == 'Positive']
negative_df = df[df['Sentiment'] == 'Negative']
neutral_df = df[df['Sentiment'] == 'Neutral']

# Bilancia il numero di righe per ogni categoria
min_samples = min(len(positive_df), len(negative_df), len(neutral_df))

# Estrai un campione bilanciato per ogni categoria
positive_df = positive_df.sample(min_samples, random_state=42)
negative_df = negative_df.sample(min_samples, random_state=42)
neutral_df = neutral_df.sample(min_samples, random_state=42)

# Unisci i dataframe bilanciati in un unico dataframe
balanced_df = pd.concat([positive_df, negative_df, neutral_df])

# Dividi il dataframe bilanciato in due parti uguali
train_df, test_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['Sentiment'], random_state=42)

# Salva i nuovi dataframe in due file CSV separati
train_df.to_csv('../dataset/unlabeled.csv', index=False)
test_df.to_csv('../dataset/labeled.csv', index=False)
