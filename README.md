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

### 2.1 Dataset

### 2.2 Pre Processing

### 2.3 LSTM

### 2.4 BERT


