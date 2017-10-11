from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
data = pd.read_csv("/home/mohit/Downloads/f2c2f440-8-dataset_he/train.csv")
data.drop(['Browser_Used','Device_Used','User_ID'], axis=1, inplace=True)

train, test = train_test_split(data, test_size=0.2)


clf = Pipeline([('vect',CountVectorizer()),
                ('tfidf',TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42))])
clf.fit(train['Description'], train['Is_Response'])

predict = clf.predict(test['Description'])
print np.mean(predict == test['Is_Response'])
print metrics.classification_report(test['Is_Response'],predict,['happy', 'not happy'])
#print metrics.roc_auc_score(test['Is_Response'], predict)