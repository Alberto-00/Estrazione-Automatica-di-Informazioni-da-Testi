import pandas as pd
import numpy as np

from keras import Model
from keras.layers import RepeatVector
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.layers import Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Carica il DataFrame
df_train_labeled = pd.read_csv('../dataset/labeled.csv')
df_train_unlabeled = pd.read_csv('../dataset/unlabeled.csv')
df_test = pd.read_csv('../dataset/Corona_NLP_train_clean.csv')
# Sostituisci 'Extremely Positive' con 'Positive' e 'Extremely Negative' con 'Negative'
df_test['Sentiment'] = df_test['Sentiment'].replace({'Extremely Positive': 'Positive', 'Extremely Negative': 'Negative'})
print(df_test.head(20))
# Definizione del tokenizer
tokenizer = Tokenizer(num_words=50000, lower=True, oov_token='<UNK>')
tokenizer.fit_on_texts(df_train_labeled['OriginalTweet'])
# Definizione di max_length
max_length = 256  # Sostituisci con il valore corretto

# Parte etichettata
encoder = LabelEncoder()
df_train_labeled['Sentiment'] = encoder.fit_transform(df_train_labeled['Sentiment'])
X_train_labeled = pad_sequences(tokenizer.texts_to_sequences(df_train_labeled['OriginalTweet']), maxlen=max_length, padding='post')
y_train_labeled = to_categorical(df_train_labeled['Sentiment'], num_classes=3)

# Parte non etichettata
X_train_unlabeled = pad_sequences(tokenizer.texts_to_sequences(df_train_unlabeled['OriginalTweet']), maxlen=max_length, padding='post')
print("Dimensioni di X_train_unlabeled prima del padding:", X_train_unlabeled.shape)
X_train_unlabeled = pad_sequences(tokenizer.texts_to_sequences(df_train_unlabeled['OriginalTweet']), maxlen=max_length, padding='post')
print("Dimensioni di X_train_unlabeled dopo il padding:", X_train_unlabeled.shape)

# Dati di test
df_test['Sentiment'] = encoder.transform(df_test['Sentiment'])
X_test = pad_sequences(tokenizer.texts_to_sequences(df_test['OriginalTweet']), maxlen=max_length, padding='post')
y_test = to_categorical(df_test['Sentiment'], num_classes=3)

# Costruzione e compilazione del modello autoencoder
embedding_dim = 32

# Encoder
encoder_inputs = Input(shape=(max_length,))
encoder_embedding = Embedding(input_dim=50000, output_dim=embedding_dim, input_length=max_length)(encoder_inputs)
encoder_lstm = Bidirectional(LSTM(16, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(encoder_embedding)
encoder_output = Bidirectional(LSTM(16, dropout=0.5, recurrent_dropout=0.5))(encoder_lstm)

# RepeatVector per ripetere la rappresentazione compressa
decoder_input = RepeatVector(max_length)(encoder_output)


# Decoder
decoder_lstm = RepeatVector(max_length)(encoder_output)  # RepeatVector prima di LSTM
decoder_lstm = LSTM(16, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(decoder_lstm)
decoder_lstm = LSTM(16, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(decoder_lstm)
decoder_output = Dense(50000, activation='softmax')(decoder_lstm)




autoencoder = Model(inputs=encoder_inputs, outputs=decoder_output)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')

# Implementazione dell'Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Addestramento del modello autoencoder sulla parte non etichettata
autoencoder.fit(X_train_unlabeled, X_train_unlabeled, batch_size=64, epochs=50, validation_split=0.2, callbacks=[early_stopping])

# Estrazione della rappresentazione compressa (codifica) dei dati
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_output)
X_train_encoded = encoder_model.predict(X_train_labeled)
X_test_encoded = encoder_model.predict(X_test)

# Aggiunta della rappresentazione compressa ai dati etichettati
X_train_combined = np.hstack([X_train_labeled, X_train_encoded])
X_test_combined = np.hstack([X_test, X_test_encoded])

# Costruzione e compilazione del modello finale per la classificazione supervisionata
model = Sequential([
    Dense(64, activation='relu', input_dim=256),  # Modifica l'input_dim in base alle dimensioni della tua rappresentazione compressa
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Addestramento del modello finale
model.fit(X_train_combined, y_train_labeled, batch_size=64, epochs=50, validation_split=0.2, callbacks=[early_stopping])

# Valutazione finale del modello sui dati di test
y_pred = model.predict(X_test_combined)
y_predicted_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print('\n================= Classification Report ========================\n')
target_names = ['Class Negative', 'Class Neutral', 'Class Positive']
print(classification_report(y_test_labels, y_predicted_labels, target_names=target_names))
