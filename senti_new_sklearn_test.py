import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import CountVectorizer

#new data
data_new=pd.read_csv(r"D:\sentimental analysis\zp_msg.csv")
data_new.head()
data_new=data_new[data_new.columns[6:7]]
data_new
data_new1=data_new.dropna(how='all')
data_new1
data_new2=data_new1['msg']
data_new2

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
model=mp_new.fit(x_test_cv,y_test)
predictions=mp_new.predict(predict_data_cv)
predictions

predictions=pd.DataFrame(predictions,columns=['Prediction'])
predictions

predictions=predictions.reindex_like(data_new1,'ffill')
predictions=predictions.rename(columns={'msg':'Predictions'})
predictions

predict_data=pd.concat([data_new,predictions],axis=1)#axis = 1 for column and axis = 0 for row
predict_data