import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

# Leggi il DataFrame
df_train = pd.read_csv('../dataset/Corona_NLP_train_clean.csv')
df_test = pd.read_csv('../dataset/Corona_NLP_test_clean.csv')

# Tratta i valori mancanti con SimpleImputer
imputer = SimpleImputer(strategy='constant', fill_value='')  # Puoi scegliere una strategia diversa se necessario
df_train['OriginalTweet'] = imputer.fit_transform(df_train['OriginalTweet'].values.reshape(-1, 1)).flatten()

# Creazione di un oggetto TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Applicazione del TF-IDF sulla colonna specifica del DataFrame
tfidf_matrix = tfidf_vectorizer.fit_transform(df_train['OriginalTweet'])

# Ottieni le features (parole/token) del tuo vocabolario
features = tfidf_vectorizer.get_feature_names_out()

# Creazione di un DataFrame Pandas per visualizzare la matrice TF-IDF
tfidf_df = pd.DataFrame(data=tfidf_matrix.toarray(), columns=features)

# Visualizza le prime 5 righe della matrice TF-IDF
print(tfidf_df.head())

# Visualizza le prime 5 colonne della matrice TF-IDF
print(tfidf_df.iloc[:, :5])
