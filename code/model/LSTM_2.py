import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

# Read the DataFrame
df_train = pd.read_csv('/content/drive/MyDrive/Corona_NLP_train.csv')
df_test = pd.read_csv('/content/drive/MyDrive/Corona_NLP_test.csv')

# Encode labels
encoder = LabelEncoder()
df_train['Sentiment'] = encoder.fit_transform(df_train['Sentiment'])
df_test['Sentiment'] = encoder.transform(df_test['Sentiment'])

# Tokenization and padding
max_length = 50
tokenizer = Tokenizer(num_words=50000, lower=True, oov_token='<UNK>')
tokenizer.fit_on_texts(df_train['OriginalTweet'])
X_train = pad_sequences(tokenizer.texts_to_sequences(df_train['OriginalTweet']), maxlen=max_length, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(df_test['OriginalTweet']), maxlen=max_length, padding='post')

# One-hot encode labels
y_train = to_categorical(df_train['Sentiment'], num_classes=5)
y_test = to_categorical(df_test['Sentiment'], num_classes=5)

model_path = ('/home/alberto/Documenti/GitHub/Estrazione-Automatica-di-Informazioni-da-Testi/'
              'code/model/pre-trained/saved_model.h5')

if os.path.exists(model_path):
    model = load_model(model_path)
    print("Modello caricato con successo.")
else:
    print("Il modello non esiste nel percorso specificato.")

    # Build and compile the model
    model = Sequential([
        Embedding(input_dim=50000, output_dim=128, input_length=max_length),
        SpatialDropout1D(0.5),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)),
        Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.5)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # Implement Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, batch_size=64, epochs=50,
                        validation_split=0.2, callbacks=[early_stopping])

    # Salva il modello su disco
    model.save('/home/alberto/Documenti/GitHub/Estrazione-Automatica-di-Informazioni-da-Testi/'
               'code/model/pre-trained/saved_model_5.h5')

# Evaluate the model
y_pred = model.predict(X_test)
y_predicted_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print('\n================= Classification Report ========================\n')
target_names = ['Class Negative', 'Class Neutral', 'Class Positive', 'Extremely Positive', 'Extremely Negative']
print(classification_report(y_test_labels, y_predicted_labels, target_names=target_names))
