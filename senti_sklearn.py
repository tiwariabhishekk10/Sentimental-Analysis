import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB #Multinomial Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


data=pd.read_csv(r"D:\sentimental analysis\Dataset\Dataset")
data.head()

data.dropna()
data=data[data.Sentiment != 'neutral']#Removing neutral from rows
data

#To run naive bayes classifier in Scikit Learn the categories must be numric

#0:positive 1:negative

data['Label']=data['Sentiment'].apply(lambda x:0 if x=='positive' else 1) 
data

x_train, x_test, y_train, y_test=train_test_split(data['Messages'],data['Sentiment'],random_state=1, train_size=0.8)

#Convert meassages into word count
#A Naive Bayes classifier needs to be able to calculate how many times each word appears in each document and how many times it appears in each category.
#data should look like [0,1,0,1,,,]

#CountVectorizer creates a vector of word counts for each messages to form a matrix. Each index corresponds to a word and every word appearing in the messages is represented.

#strip_accents : {'ascii', 'unicode', None} Remove accents and perform other character normalization during the preprocessing step.
# ascii' is a fast method that only works on characters that have an direct ASCII mapping. 'unicode' is a slightly slower method that works on any characters. None (default) does nothing.
#token_pattern : string Regular expression denoting what constitutes a "token", only used if analyzer == 'word'. The default regexp select tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a token separator).

cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
x_train_cv = cv.fit_transform(x_train)
x_train_cv
x_test_cv = cv.transform(x_test)
x_test_cv

#Fit the model 

naive_bayes=MultinomialNB()
model=naive_bayes.fit(x_train_cv,y_train)
predictions=naive_bayes.predict(x_test_cv)

predictions=pd.DataFrame(predictions,columns=['Prediction'])
predictions

predictions_prob=naive_bayes.predict_proba(x_test_cv)

#predictions_prob=pd.DataFrame(predictions_prob,columns=['Messages','Probability'])
#predictions_prob

#check results
print('Accuracy score: ', accuracy_score(y_test, predictions))

#Confusion Matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False, xticklabels=['positive', 'negative'], yticklabels=['positive', 'negative'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.legend()
plt.show()