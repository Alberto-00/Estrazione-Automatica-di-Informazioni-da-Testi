##  1  Introduction

###  1.1 Problem
More and more people are exchanging text messages through the use of social media, and the analysis of the information can be used to make statistics in the behavior and in people's psychology. Using Natural Language Processing (NLP), we can extrapolate key words from each message that allow us to achieve the proposed goals. The following paper discusses the development of an Automatic Information Extraction system from English-language text messages by using
of the spaCy library that provides a set of pre-trained templates using the NER technique. In the following case, the model considered is RoBERTa which we will go on to analyze in the following paragraphs.

### 1.2 Workflow
The first task performed was to identify the dataset to be used for the task introduced in the previous paragraph. The dataset used was: SMS-NER-Dataset-165-Annotations found on kaggle at the following <a href="https://www.kaggle.com/code/spiralforge/extracting-important-imformation-from-sms/input">link</a>. Next, a data cleaning was performed on the dataset in order to ensure uniformity in data representation. After that, the cleaned dataset was divided into training and testing set and converted to <code>.spacy</code> format so that it could be computed by the chosen model. Next, the <code>config.cfg</code> file was generated, which is nothing but a configuration file with all the hyperparameters and settings that the model has to comply with. After that, the training part was given as input to the pre-trained model and in output were saved two models:
<ul>
  <li><code>model-last</code>: the model trained in the last iteration (it could be used to resume the training at a later time); </li>
  <li><code>model-best</code>: the model that scored highest on the test dataset;</li>
</ul>
Finally, Precision, Recall and F1-Score metrics were reported.
In order to best perform the information extraction task, 3 different pre-trained models were used in accuracy for the prediction of tags and compared with each other. The models used were:
1. en_core_web_sm;
2. en_core_web_md;
3. en_core_web

##  2  Approach
In this section we are going to cover the implementation parts. In particular, we will discuss the structure of the dataset and the configuration files.

### 2.1 Dataset
The dataset is in json format and is structured as follows:
<ul>
  <li>
    "<code>classes</code>": contains the list of tags to be identified within the messages: "MONEY", "TITLE", "OTP", "TRANSAC", "TIME", "PURPOSE".
  </li>
  <li>"<code>annotations</code>": contains the message list and entity class for each message;</li>
  <ul>
    <li>"<code>entities</code>": each entity is an array of tuples where each tuple has within it two integers and a tag (the integers are the coordinates of the tag associated with a specific phrase, e.g. [19,26, "TRANSAC"]). </li>
  </ul>
</ul>
Next, the dataset is divided into two parts: train and test set. If a message has the associated entity class empty, then this is filled with the tuple [(0, 0, 'PEARSON')].

### 2.3 Configuration File
Within the SMS-NER-Dataset-165-Annotations folder we find the <code>base_config.cfg</code> configuration file used to set up the model that will be trained on the previous dataset.
To set up the model structure we run the command:
```
python -m spacy init fill-config dataset/SMS-NER-Dataset-165-Annotations/base_config.cfg config.cfg
```

After that, it will start the training phase and finally the of testing by running the command:
```
python -m spacy train config.cfg -output ./output -paths.train train.spacy -paths.dev test.spacy
```

To conclude, we print the metrics produced by the best model by running the command:
```
python -m spacy benchmark accuracy model/large/model-best model/large/test.spacy -output -code -gold-preproc -gpu-id 0 -displacy-path model/large
```

## 3 Report
The report can be found at the follow link: a href="https://github.com/Alberto-00/Estrazione-Automatica-di-Informazioni-da-Testi">Report</a>.

## 4 Author & Contacts
| Name | Description |
| --- | --- |
| <p dir="auto"><strong>Alberto Montefusco</strong> |<br>Developer - <a href="https://github.com/Alberto-00">Alberto-00</a></p><p dir="auto">Email - <a href="mailto:a.montefusco28@studenti.unisa.it">a.montefusco28@studenti.unisa.it</a></p><p dir="auto">LinkedIn - <a href="https://www.linkedin.com/in/alberto-montefusco">Alberto Montefusco</a></p><p dir="auto">My WebSite - <a href="https://alberto-00.github.io/">alberto-00.github.io</a></p><br>|
| <p dir="auto"><strong>Alessandro Aquino</strong> |<br>Developer   - <a href="https://github.com/AlessandroUnisa">AlessandroUnisa</a></p><p dir="auto">Email - <a href="mailto:a.aquino33@studenti.unisa.it">a.aquino33@studenti.unisa.it</a></p><p dir="auto">LinkedIn - <a href="https://www.linkedin.com/in/alessandro-aquino-62b74218a/">Alessandro Aquino</a></p><br>|
| <p dir="auto"><strong>Mattia d'Argenio</strong> |<br>Developer   - <a href="https://github.com/mattiadarg">mattiadarg</a></p><p dir="auto">Email - <a href="mailto:m.dargenio5@studenti.unisa.it">m.dargenio5@studenti.unisa.it</a></p><p dir="auto">LinkedIn - <a href="https://www.linkedin.com/in/mattia-d-argenio-a57849255/)https://www.linkedin.com/in/mattia-d-argenio-a57849255/">Mattia d'Argenio</a></p><br>|
