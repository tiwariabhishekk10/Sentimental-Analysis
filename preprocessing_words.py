import nltk
import string, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.text import Text
#nltk.download('punkt')
from wordcloud import WordCloud

def remove_punctuation(sentence):
    sentence=re.sub(r'[^\w\s]','',sentence)
    return sentence

stop_words=list(set(stopwords.words('english')))
print('Number of stopwords:', len(stop_words))
print(f'First 50 stop words:\n{stop_words[:30]}')

def remove_stopword(sentence):
    return[w for w in sentence if not w in stop_words]

data=pd.read_csv(r"D:\sentimental analysis\Dataset\Dataset.csv")
data.head()

mood=data.groupby('mood')
mood.describe().head()

msg=data.groupby('msg')
msg.describe().head()

review=data.groupby('review')
review.describe().head()

role=data.groupby('role')
role.describe().head()

system=data.groupby('system')
system.describe().head()

thought=data.groupby('thought')
thought.describe().head()

####MSG####

data_msg=data[data.columns[1:2]]
data_msg.head()
data_msg.dropna()
datastr_msg=data_msg.to_string()
datastr_msg=data_msg.msg.tolist()

#Tokenzing the sentences and words
sentences_msg=sent_tokenize(datastr_msg)

cleaned_sent_msg=[remove_punctuation(sentence) for sentence in sentences_msg]
print(cleaned_sent_msg)

clean_speech_words_msg=[word_tokenize(sentence) for sentence in cleaned_sent_msg]
print(clean_speech_words_msg)

#Removing stop words: stop words are small words that can be ignored
#nlkt has inbuilt stop words

filtered_msg=[remove_stopword(s) for s in clean_speech_words_msg]
word_count= len([w for words in clean_speech_words_msg for w in words])
word_count2= len([w for words in filtered_msg for w in words])
print(f'no. of words before:{word_count}')
print(f'no. of words before:{word_count2}')
print(filtered)
filtered_msg=np.array(filtered_msg)
filtered_msg=pd.DataFrame(filtered_msg)
filtered_msg=filtered_msg.to_string()
#Concordance
#Concordance can be used to see all usages of a particular word in context. It returns all occurrences of a word and the parts of sentences it was used in

speech_words_msg=Text(word_tokenize(datastr_msg))
speech_words_msg.concordance('I') #Displaying word sentences were word great is used

#Word cloud

wc_msg=WordCloud(width=1000,height=1000,random_state=1).generate(filtered_msg)
plt.imshow(wc_msg)
plt.show()

####MOOD####

data_mood=data[data.columns[0:1]]
data_mood.head()
data_mood.dropna()
datastr_mood=data_mood.to_string()

#Tokenzing the sentences and words
sentences_mood=word_tokenize(datastr_mood)
sentences_mood=np.array(sentences_mood)
sentences_mood=pd.DataFrame(sentences_mood)
sentences_mood=sentences_mood.to_string()

#Word cloud
wc_mood=WordCloud(width=500,height=500,random_state=1).generate(sentences_mood)
plt.imshow(wc_mood)
plt.show()

####THOUGHT####

data_thought=data[data.columns[6:]]
data_thought.head()
data_thought.dropna()
datastr_thought=data_thought.to_string()

#Tokenzing the sentences and words
sentences_thought=sent_tokenize(datastr_thought)

cleaned_sent_thought=[remove_punctuation(sentence) for sentence in sentences_thought]
print(cleaned_sent_thought)

clean_speech_words_thought=[word_tokenize(sentence) for sentence in cleaned_sent_thought]
print(clean_speech_words_thought)

#Removing stop words: stop words are small words that can be ignored
#nlkt has inbuilt stop words

filtered_thought=[remove_stopword(s) for s in clean_speech_words_thought]
filtered_thought=np.array(filtered_thought)
filtered_thought=pd.DataFrame(filtered_thought)
filtered_thought=filtered_thought.to_string()

#Word cloud
wc_thought=WordCloud(width=1000,height=1000,random_state=1).generate(filtered_thought)
plt.imshow(wc_thought)
plt.show()

