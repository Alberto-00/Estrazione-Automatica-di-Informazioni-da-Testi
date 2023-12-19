import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from torch.optim import Adam
from tqdm import tqdm

def tokenize_data(texts, labels, max_length=128):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Converte le liste in tensori
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Converti le etichette in un array NumPy con tipo di dati Long
    labels = labels.to_numpy().astype(np.int64)
    labels = torch.tensor(labels).long()

    return input_ids, attention_masks, labels


# Leggi i dati di allenamento e test
df_train = pd.read_csv('../dataset/Corona_NLP_test_clean.csv')
df_test = pd.read_csv('../dataset/Corona_NLP_test_clean.csv')

df_train = df_train.loc[~df_train['Sentiment'].isin(['Extremely Positive', 'Extremely Negative'])]
df_test = df_test.loc[~df_test['Sentiment'].isin(['Extremely Positive', 'Extremely Negative'])]

# Codifica delle etichette
encoder = LabelEncoder()
df_train['Sentiment'] = encoder.fit_transform(df_train['Sentiment'])
df_test['Sentiment'] = encoder.transform(df_test['Sentiment'])

# Carica il modello BERT preaddestrato e il tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(encoder.classes_))

# Tokenizzazione e preparazione dei dati di allenamento
input_ids_train, attention_masks_train, labels_train = tokenize_data(df_train['OriginalTweet'], df_train['Sentiment'])

# Definizione del DataLoader per il set di dati di allenamento
train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Parametri di ottimizzazione
optimizer = AdamW(model.parameters(), lr=5e-5)

# Funzione di perdita per la classificazione
loss_fn = torch.nn.CrossEntropyLoss()

# Addestramento del modello
model.train()
optimizer = Adam(model.parameters(), lr=2e-5)
num_epochs = 10  # Puoi regolare il numero di epoche a seconda delle tue esigenze

for epoch in range(num_epochs):
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    total_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}')


# Tokenizzazione e preparazione dei dati di test
input_ids_test, attention_masks_test, labels_test = tokenize_data(df_test['OriginalTweet'], df_test['Sentiment'])

# Definizione del DataLoader per il set di dati di test
test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Valutazione del modello
model.eval()
all_preds = []
total_loss = 0.0

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc='Testing', leave=False):
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        total_loss += outputs.loss.item()

# Calcolo delle metriche di valutazione
accuracy = accuracy_score(df_test['Sentiment'], all_preds)
classification_rep = classification_report(df_test['Sentiment'], all_preds)

average_loss = total_loss / len(test_dataloader)
print(f'Average Loss on Test Data: {average_loss}')
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_rep}')
