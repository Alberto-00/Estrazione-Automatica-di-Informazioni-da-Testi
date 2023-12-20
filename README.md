##  1  Introduction

###  1.1 Problem
More and more people are contributing to social media, and analysis of information available online can be used to reflect on changes in people's perceptions, behavior, and psychology. Using Natural Language Processing (NLP), people's feelings and attitudes can be determined by extracting subjective comments on a given topic (in the following case the context is the pandemic from Covid-19) using different sentiments such as Positive, Extremely Positive, Negative, Extremely Negative and Neutral.
The following paper discusses the development an Automatic Information Extraction system from Text focused on tweets regarding the pandemic from COVID-19. A text analysis of tweets posted on the social media will be performed by going to extract the relative sentiment for each tweet.

### 1.2 Workflow
The first task performed was to identify the dataset to be used for the task introduced in the previous section. The dataset used was: Coronavirus tweets NLP found on kaggle at the following <a href="https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/data">link</a>. Next, a data cleaning was performed on the dataset in order to ensure uniformity in data representation. After that, the cleaned dataset will be given as input to a recurrent neural network (in our case an LSTM) and in output we will have the values related to the model performance (accuracy, precision, recall and f1-score). Specifically, the goal was to perform two different types of classification: 
<ul>
  <li>classification of 3 different classes (Positive, Negative, Neutral); </li>
  <li>classification of 5 different classes (Positive, Extremely Positive, Negative, Extremely Neg ative and Neutral);</li>
</ul>

Finally, BERT was implemented and the following were compared the results of the different models.

##  2  Approach
In this section we will go on to discuss the implementation parts. In particular, we will discuss how data cleaning was done and what models were used for classification.

### 2.1 Pre Processing
For the data cleaning phase, the <code>cleantweet(tweet)</code> function is defined to remove URLs, mentions, hashtags, special characters, non-ASCII characters, and convert text to lowercase. Stopwords and other language data are downloaded via the nltk module. The <code>removestopwords(text)</code> function is defined to remove stopwords from a text. The defined cleaning operations are applied to the 'OriginalTweet' column of both DataFrames (<code>df_train</code>, used for training the mod ello and <code>df_test</code>, used, on the other hand, for testing).

### 2.3 LSTM

### 2.4 BERT

## 3 Author & Contacts
| Name | Description |
| --- | --- |
| <p dir="auto"><strong>Alberto Montefusco</strong> |<br>Developer - <a href="https://github.com/Alberto-00">Alberto-00</a></p><p dir="auto">Email - <a href="mailto:a.montefusco28@studenti.unisa.it">a.montefusco28@studenti.unisa.it</a></p><p dir="auto">LinkedIn - <a href="https://www.linkedin.com/in/alberto-montefusco">Alberto Montefusco</a></p><p dir="auto">My WebSite - <a href="https://alberto-00.github.io/">alberto-00.github.io</a></p><br>|
| <p dir="auto"><strong>Alessandro Aquino</strong> |<br>Developer   - <a href="https://github.com/AlessandroUnisa">AlessandroUnisa</a></p><p dir="auto">Email - <a href="mailto:a.aquino33@studenti.unisa.it">a.aquino33@studenti.unisa.it</a></p><p dir="auto">LinkedIn - <a href="https://www.linkedin.com/in/alessandro-aquino-62b74218a/">Alessandro Aquino</a></p><br>|
| <p dir="auto"><strong>Mattia d'Argenio</strong> |<br>Developer   - <a href="https://github.com/mattiadarg">mattiadarg</a></p><p dir="auto">Email - <a href="mailto:m.dargenio5@studenti.unisa.it">m.dargenio5@studenti.unisa.it</a></p><p dir="auto">LinkedIn - <a href="https://www.linkedin.com/in/mattia-d-argenio-a57849255/)https://www.linkedin.com/in/mattia-d-argenio-a57849255/">Mattia d'Argenio</a></p><br>|
