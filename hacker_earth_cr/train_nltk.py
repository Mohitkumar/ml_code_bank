import pandas as pd
from sklearn.model_selection import train_test_split
import string
import nltk
import random
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pickle
import re
from nltk.collocations import *


lemma = nltk.WordNetLemmatizer()
stemmer = SnowballStemmer('english')


def preprocess(text):
    text = "".join([ch if ch not in string.punctuation else ' ' for ch in text])
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.lower() not in stopwords.words('english')]
    return [stemmer.stem(w) for w in tokens]


def clean_and_store():
    data = pd.read_csv("/home/mohit/ml_data/train.csv")
    document =[(preprocess(text),res) for text,res in zip(data['Description'], data['Is_Response'])]
    with open('cleaned_data', 'wb') as f:
        pickle.dump(document,f,2)


def get_word_features():
    data = pd.read_csv("/home/mohit/ml_data/train.csv")
    data.drop(['Browser_Used', 'Device_Used', 'User_ID'], axis=1, inplace=True)
    text = data['Description'].apply(preprocess)
    all_words = nltk.FreqDist(x for w in text for x in w)
    word_features = list(all_words)[:2000]
    return word_features


def document_features(document,feats):
    document_words = set(document)
    features = {}
    for word in feats:
        features['contains({})'.format(word)] = (word in document_words)
    return features

def store_frequent_words():
    with open('cleaned_data', 'r') as f1:
        document = pickle.load(f1)

    all_words = nltk.FreqDist(x for w, label in document for x in w)
    word_features = list(all_words)[:2000]
    print word_features
    with open('top_words', 'wb') as f:
        pickle.dump(word_features,f,2)


def get_frequent_suffixes():
    with open('cleaned_data', 'r') as f1:
        document = pickle.load(f1)
    suffix_fdist = nltk.FreqDist()
    for word_list, label in document:
        for word in word_list:
            suffix_fdist[word[-1:]] += 1
            suffix_fdist[word[-2:]] += 1
            suffix_fdist[word[-3:]] += 1

    common_suffixes = [suffix for (suffix, count) in suffix_fdist.most_common(100)]
    print common_suffixes
    return common_suffixes


def pos_features(word, common_suffixes):
     features = {}
     for suffix in common_suffixes:
         features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
     return features


def train():
    with open('cleaned_data','r') as f:
        document = pickle.load(f)

    with open('top_words','r') as f:
        word_features = pickle.load(f)

    print word_features
    featuresets = [(document_features(d, word_features), c) for (d, c) in document]
    print "done extracting features"

    ferquent_suffixes = get_frequent_suffixes()
    pos_featureset = [(pos_features(d, ferquent_suffixes), c) for (d, c) in document]
    print "done extracting pos features"

    test_split = len(featuresets)/5
    train_set, test_set = featuresets[test_split:], featuresets[:test_split]
    train_set1, test_set1 = pos_featureset[test_split:], pos_featureset[:test_split]
    classifier = nltk.NaiveBayesClassifier.train(train_set+train_set1)
    print(nltk.classify.accuracy(classifier, test_set+test_set1))


if __name__ == '__main__':
    train()
    #doc=[[[1,2,3]]]
    #print [y for t in doc for z in t for y in z]
