import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords

#nltk.data.path.append("/Users/mattiadargenio/PycharmProjects/EstrazioneInfo/nltk")
#nltk.download('stopwords')
df_train = pd.read_csv('./dataset/Corona_NLP_train.csv')
df_test = pd.read_csv('./dataset/Corona_NLP_test.csv')


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


# Apply the clean_tweet function to the 'TweetAt' column
df_train['OriginalTweet'] = df_train['OriginalTweet'].apply(clean_tweet)
print(df_train['OriginalTweet'])
