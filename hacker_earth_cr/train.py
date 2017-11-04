from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
import string
import re
nltk.download()
np.random.seed(12)

def tokenize(text):
    text = "".join([ch if ch not in string.punctuation else ' ' for ch in text])
    tokens = nltk.word_tokenize(text)
    return tokens

data = pd.read_csv("/home/mohit/ml_data/train.csv")
data.drop(['Browser_Used','Device_Used','User_ID'], axis=1, inplace=True)
train, test = train_test_split(data, test_size=0.2)
#train = data

lemma = nltk.WordNetLemmatizer()
stopw = set(stopwords.words('english'))

print stopw

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([lemma.lemmatize(w,pos='v') for w in analyzer(doc)])
sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42)
mnb = MultinomialNB(fit_prior=False)
clf = Pipeline([('vect',StemmedCountVectorizer(tokenizer=tokenize, stop_words=stopw)),
                ('clf', mnb)])
clf.fit(train['Description'], train['Is_Response'])

#t_data = pd.read_csv("/home/mohit/Downloads/f2c2f440-8-dataset_he/test.csv")
#ids = t_data['User_ID']
#t_data.drop(['Browser_Used','Device_Used','User_ID'], axis=1, inplace=True)
#preds = clf.predict(t_data['Description'])
#out = pd.DataFrame({'User_ID':ids,'Is_Response':preds})
#out.to_csv('out.csv', index=False)
predict = clf.predict(test['Description'])
print np.mean(predict == test['Is_Response'])
print metrics.classification_report(test['Is_Response'],predict,['not happy', 'happy'])

