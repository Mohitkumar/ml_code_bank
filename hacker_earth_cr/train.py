from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
#nltk.download('wordnet')
np.random.seed(12)

stop = stopwords.words('english')
data = pd.read_csv("/home/mohit/Downloads/f2c2f440-8-dataset_he/train.csv")
data.drop(['Browser_Used','Device_Used','User_ID'], axis=1, inplace=True)
train, test = train_test_split(data, test_size=0.2)
#train = data

stemmer = SnowballStemmer("english", ignore_stopwords=True)
lemma = nltk.WordNetLemmatizer()
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([lemma.lemmatize(w,pos='v') for w in analyzer(doc)])
#CountVectorizer()

clf = Pipeline([('vect',StemmedCountVectorizer(stop_words=stopwords.words('english'))),
                ('tfidf',TfidfTransformer()),
                ('clf', MultinomialNB(fit_prior=False, alpha=0.8))])
clf.fit(train['Description'], train['Is_Response'])

#t_data = pd.read_csv("/home/mohit/Downloads/f2c2f440-8-dataset_he/test.csv")
#ids = t_data['User_ID']
#t_data.drop(['Browser_Used','Device_Used','User_ID'], axis=1, inplace=True)
#preds = clf.predict(t_data['Description'])
#out = pd.DataFrame({'User_ID':ids,'Is_Response':preds})
#out.to_csv('out.csv', index=False)
predict = clf.predict(test['Description'])
print np.mean(predict == test['Is_Response'])
print metrics.classification_report(test['Is_Response'],predict,['happy', 'not happy'])