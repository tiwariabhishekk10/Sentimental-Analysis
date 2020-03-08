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


data=pd.read_csv(r'D:\sentimental analysis\Dataset\Dataset')
data.head()

data.dropna()
data=data[data.Sentiment != 'neutral']#Removing neutral from rows
data

#To run naive bayes classifier in Scikit Learn the categories must be numric

#0:positive 1:negative

data['Label']=data['Sentiment'].apply(lambda x:0 if x=='positive' else 1) 
data

x_train=data['Messages'][:int(0.8*(len(data['Messages'])))]
x_train
y_train=data['Sentiment'][:int(0.8*(len(data['Sentiment'])))]

#x_train, x_test, y_train, y_test=train_test_split(data['Messages'],data['Sentiment'],random_state=1, train_size=0.8)

cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
x_train_cv = cv.fit_transform(x_train)
x_train_cv

#Fit the model 

naive_bayes=MultinomialNB()
model=naive_bayes.fit(x_train_cv,y_train)

##Saving the model##

# with open('new file name','write bytes=wb') as file:
#   pickle.dump(model.file)

with open('model_sentimental_analysis_new','wb') as file:
    pickle.dump(model,file)
    
#Loading saved model

# with open ('saved file name','readmode=rb') as file:
#   mp=pickle.load(file)

#with open ('model_sentimental_analysis_new','rb') as file:
    #mp_new=pickle.load(file)
