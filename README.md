# hatespeechdetection#Importing lib
import pandas as pd
import numpy as np
dataset=pd.read_csv("twitter.csv")
dataset
dataset.isnull()
dataset.isnull().sum()
dataset.info()
dataset.describe()
dataset["labels"]=dataset["class"].map({0: "Offensive language",
                                       1: "Hate Speech",
                                       2: "No hate or Offensive language"})
dataset
data=dataset[["tweet","labels"]]
data
import re
import nltk
nltk.download('stopwords')
import string
#importing of stop words and stemming of words
from nltk.corpus import stopwords
stopwords=set(stopwords.words("english"))
#import stemming
stemmer=nltk.SnowballStemmer("english")
# Data Cleaning
def clean_data(text):
    text = str(text).lower()
    text = re.sub('http?://\s+|www\.s+','',text)
    text = re.sub('\[.*?\]','',text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('[%s]' %re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\v*','',text)
    # Stop words removing
    text = [word for word in text.split(' ') if word not in stopwords]
    text = " ".join(text)
    # Stemming the text
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text
data["tweet"]=data["tweet"].apply(clean_data)
data
X=np.array(data["tweet"])
Y=np.array(data["labels"])
X
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
cv=CountVectorizer()
X=cv.fit_transform(X)
X
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
X_train
Y_train
#building out ML model
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,Y_train)
Y_pred=dt.predict(X_test)
#confusion matrix and accuracy
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)
cm
import seaborn as sns
import matplotlib.pyplot as ply
%matplotlib inline
sns.heatmap(cm, annot = True, fmt = ".1f", cmap = "YlGnBu")
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred)
sample="Let's unite and kill all the people who are protesting against the government"
sample=clean_data(sample)
sample
data1=cv.transform([sample]).toarray()
data1
dt.predict(data1)

