import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import textblob as tb
import textblob
from textblob import TextBlob
from textblob.en.inflect import singularize
from textblob import Word

df_train = pd.read_csv('../dataset/Corona_NLP_train.csv')
df_test = pd.read_csv('../dataset/Corona_NLP_test.csv')

print(df_train)
print(f"->{df_train.isin(['NaN', 'None', 'none']).sum()}")
print(df_train)

def clean_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)

    # Remove mentions (@username)
    tweet = re.sub(r'@\w+', '', tweet)

    # Remove hashtags (#)
    tweet = re.sub(r'#', '', tweet)

    # Remove hashtags (#)
    tweet = re.sub(r'-', '', tweet)

    # Remove special characters and punctuations (except alphanumeric and spaces)
    tweet = re.sub(r'[^\w\s]', '', tweet)

    # Remove non-ASCII characters
    tweet = tweet.encode('ascii', 'ignore').decode()

    # Convert to lowercase
    tweet = tweet.lower()

    # Remove extra whitespaces
    tweet = ' '.join(tweet.split())

    return tweet


# Download the stopwords, wordnet(lemmatization), omw-1.4 dataset (only need to run this once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Define a function to remove stopwords from text
stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


# Apply the clean_tweet function to the 'TweetAt' column
df_train['OriginalTweet'] = df_train['OriginalTweet'].apply(clean_tweet)
# Apply stopwords removal to your DataFrame
df_train['OriginalTweet'] = df_train['OriginalTweet'].apply(remove_stopwords)

# Apply the clean_tweet function to the 'TweetAt' column
df_test['OriginalTweet'] = df_test['OriginalTweet'].apply(clean_tweet)
# Apply stopwords removal to your DataFrame
df_test['OriginalTweet'] = df_test['OriginalTweet'].apply(remove_stopwords)

print(df_train.duplicated())#verifico se ci sono duplicati
df_train.dropna(inplace=True) #rimuovo le righe con valore NULL
df_train.drop_duplicates(inplace=True)#elimino duplicati

print(df_train.head(20))
df_train = df_train.drop('UserName', axis=1)
df_train = df_train.drop('ScreenName', axis=1)
df_train = df_train.drop('Location', axis=1)
df_train = df_train.drop('TweetAt', axis=1)
print(df_train.head(20))
print(df_train.info())
print(df_test.head(20))
df_test= df_test.drop('UserName', axis=1)
df_test = df_test.drop('ScreenName', axis=1)
df_test = df_test.drop('Location', axis=1)
df_test = df_test.drop('TweetAt', axis=1)
print(df_test.head(20))

print(df_train.head(20))
df_train['OriginalTweet'] = df_train['OriginalTweet'].apply(lambda x: ' '.join([singularize(item) for item in x.split()]))
print(df_train.head(20))

df_train['OriginalTweet'] = df_train['OriginalTweet'].apply(lambda x: ' '.join([Word(item).lemmatize("v") for item in x.split()]))
print(df_train.head(20))

df_train.to_csv('../dataset/Corona_NLP_train_clean.csv', index=False)
df_test.to_csv('../dataset/Corona_NLP_test_clean.csv', index=False)
