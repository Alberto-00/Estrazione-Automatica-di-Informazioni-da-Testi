import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

# Leggi il DataFrame
df_train = pd.read_csv('../dataset/Corona_NLP_train_clean.csv')
df_test = pd.read_csv('../dataset/Corona_NLP_test_clean.csv')
#Merging data columns Extremely Neagtive Sentiment as Negative and Extremely Positive Sentiment as Positive.
def merge(df):
    df['Sentiment'].replace(to_replace='Extremely Negative',value='Negative',inplace=True)
    df['Sentiment'].replace(to_replace='Extremely Positive',value='Positive',inplace=True)

merge(df_train)
merge(df_test)

df_train['Sentiment'].unique()
df_test['Sentiment'].unique()

X_train = df_train.OriginalTweet
Y_train = df_train.Sentiment

print(Y_train.shape)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
#encoder.classes_
Y_train = pd.DataFrame(Y_train,columns=['Sentiment'])
print(Y_train.head())

print(df_test.head())
X_test = df_test.OriginalTweet
Y_test = df_test.Sentiment
print(X_test.head())

print(Y_test.head())
Y_test = encoder.fit_transform(Y_test)
Y_test = pd.DataFrame(Y_test,columns=['Sentiment'])
print(Y_test.head())

print("X_train\n",X_train.head())
print("Y_train\n",Y_train.head())

print("X_test\n",X_test.head())
print("Y_test\n",Y_test.head())

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=50000,
                      lower=True,
                      split=' ',
                      char_level=False,
                      oov_token='<UNK>',
                      document_count=0)
tokenizer.fit_on_texts(X_train)

wordindex = tokenizer.word_index
tokenizer_config = tokenizer.get_config()
print(tokenizer_config.keys())
# tokenizer_config['word_counts']

import tensorflow.keras.preprocessing.sequence as ps

# Sequenze e padding
max_length = 50
train_sequence = tokenizer.texts_to_sequences(X_train)
train_padding = ps.pad_sequences(train_sequence, maxlen=max_length, padding='post')

test_sequence = tokenizer.texts_to_sequences(X_test)
test_padding = ps.pad_sequences(test_sequence, maxlen=max_length, padding='post')

print(train_padding.shape)
print(Y_train.shape)

from sklearn.preprocessing import OneHotEncoder
y_train = to_categorical(Y_train, num_classes=3)
print(y_train.shape)

y_test = to_categorical(Y_test, num_classes=3)
print(y_test.shape)


#Building the model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding,Dropout, Bidirectional
from keras.layers import GlobalAveragePooling1D



# Building the BASELINE MODEL
base_model = Sequential()
base_model.add(Embedding(50000,128,input_length=train_padding.shape[1]))
base_model.add(GlobalAveragePooling1D())
base_model.add(Dense(8,activation='relu'))
base_model.add(Dense(3,activation='softmax'))
base_model.summary()

#Compiling the model

base_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
y_train=np.array(y_train)
train_padding=np.array(train_padding)
#Fitting the model

history_base = base_model.fit(train_padding,y_train ,epochs=10, validation_split=0.2)


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

def plot_graphs(history, metric, save_path=None):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plot_graphs(history_base, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history_base, 'loss')
plt.ylim(0, None)
plt.show()


"""
The model performs well on training data but has a significant difference b/w train data and validation data
The model is overfit in nature
"""

from keras.layers import GlobalAveragePooling1D

regularise = tensorflow.keras.regularizers.l2(0.001)

model_r = Sequential()
model_r.add(Embedding(50000, 128, input_length=train_padding.shape[1]))
model_r.add(Dropout(0.5))
model_r.add(GlobalAveragePooling1D())
model_r.add(Dense(8, activation='relu', kernel_regularizer=regularise))
model_r.add(Dropout(0.5))
model_r.add(Dense(3, activation='softmax'))  # Cambiato il numero di neuroni e la funzione di attivazione
model_r.summary()

#Compiling the model
model_r.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Fitting the model

from sklearn.model_selection import train_test_split

#Fitting the model

history_r = model_r.fit(train_padding,y_train ,epochs=12, validation_split=0.2)



plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plot_graphs(history_r, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history_r, 'loss')
plt.ylim(0, None)

y_test_categorical = to_categorical(Y_test, num_classes=3)
score = model_r.evaluate(test_padding, y_test_categorical,verbose=0)
print("Testing Accuracy(%): ", score[1]*100)

y_pred = model_r.predict(test_padding)
y_predicted_labels = np.array([ np.argmax(i) for i in y_pred])
y_test_labels = Y_test.to_numpy()


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test_labels, y_predicted_labels)

import matplotlib.pyplot as plt
import seaborn as sn

labels=['Negative', 'Neutral','Positive']
plt.figure(figsize=(5,5))
sn.heatmap(cm,  xticklabels=labels, yticklabels=labels, fmt='d', annot=True, cmap=plt.cm.Blues)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

'''# Tratta i valori mancanti con SimpleImputer
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
 '''