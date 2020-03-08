import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import CountVectorizer

url='https://raw.githubusercontent.com/tiwariabhishekk10/Sentimental-Analysis/master/test.csv'

data=pd.read_csv(url)
data.head()

data=data.dropna(how='all')
data

x_test=data['Messages'][int(0.8*(len(data['Messages']))):]
x_test
y_test=data['Sentiment'][int(0.8*(len(data['Sentiment']))):]


cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
x_test_cv = cv.fit_transform(x_test)
x_test_cv

predict_data_cv=cv.transform(data_new2)
predict_data_cv

#Loading saved model

# with open ('saved file name','readmode=rb') as file:
#   mp=pickle.load(file)

with open ('model_sentimental_analysis_new','rb') as file:
    mp_new=pickle.load(file)

#prediction
model=mp_new.predict(x_test_cv,y_test)
model
