import pandas as pd
import numpy as np
import tensorflow
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from keras.constraints import max_norm



# Leggi il DataFrame
df_train = pd.read_csv('../dataset/Corona_NLP_train_clean.csv')
df_test = pd.read_csv('../dataset/Corona_NLP_test_clean.csv')

def merge(df):
    df['Sentiment'].replace(to_replace='Extremely Negative',value='Negative',inplace=True)
    df['Sentiment'].replace(to_replace='Extremely Positive',value='Positive',inplace=True)

merge(df_train)
merge(df_test)

X_train = df_train.OriginalTweet
Y_train = df_train.Sentiment

print(Y_train.shape)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_train = pd.DataFrame(Y_train, columns=['Sentiment'])
print(Y_train.head())

X_test = df_test.OriginalTweet
Y_test = df_test.Sentiment

Y_test = encoder.fit_transform(Y_test)
Y_test = pd.DataFrame(Y_test, columns=['Sentiment'])
print(Y_test.head())

print("X_train\n", X_train.head())
print("Y_train\n", Y_train.head())

print("X_test\n", X_test.head())
print("Y_test\n", Y_test.head())

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=50000,
                      lower=True,
                      split=' ',
                      char_level=False,
                      oov_token='<UNK>',
                      document_count=0)
tokenizer.fit_on_texts(X_train)

wordindex = tokenizer.word_index

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


from keras.constraints import max_norm
#Building the model
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Embedding,Dropout, Bidirectional
from keras.layers import GlobalAveragePooling1D
"""
regularise = tensorflow.keras.regularizers.l2(0.001)
#Building the model
# Building the BASELINE MODEL
model_lstm = Sequential()
model_lstm.add(Embedding(50000, 128, input_length=train_padding.shape[1]))
model_lstm.add(LSTM(32, kernel_constraint=max_norm(3)))
model_lstm.add(Dense(32, activation='relu', kernel_regularizer=regularise))
model_lstm.add(Dropout(0.5))
model_lstm.add(Dense(3, activation='softmax'))
model_lstm.summary()
print(model_lstm.summary())




#Compiling the model
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fitting the model
history1 = model_lstm.fit(train_padding, y_train, epochs=10, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
"""


#Building the model
model2 = Sequential([
    Embedding(50000,128,input_length=train_padding.shape[1]),
    Bidirectional(LSTM(128,return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(32,activation='relu'),
    Dropout(0.5),
    Dense(3,activation='softmax')
])
#Compiling the model
model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#Fitting the model
history2 =  model2.fit(train_padding,y_train ,epochs=8, validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',patience=3)])


y_pred = model2.predict(test_padding)
y_predicted_labels = np.array([ np.argmax(i) for i in y_pred])
y_test_labels = np.array([ np.argmax(i) for i in y_test])
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
print('\n=================Classification Report========================\n')
print(classification_report(y_test_labels, y_predicted_labels, target_names=['Class  Negative',	 'Class Neutral','Class positive']))
