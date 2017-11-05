import pandas as pd
from sklearn.model_selection import train_test_split
import string
import nltk
import random
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import mxnet as mx
import re

stemmer = SnowballStemmer('english')

def preprocess(text):
    text = "".join([ch if ch not in string.punctuation else ' ' for ch in text])
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.lower() not in stopwords.words('english')]
    return [stemmer.stem(w) for w in tokens]

def get_sentences(data):
    sentences = [preprocess(text) for text in data['Description']]
    return sentences


if __name__ == '__main__':
    data = pd.read_csv("/home/mohit/ml_data/test.csv")
    ids = data['User_ID']
    sym, arg_params, aux_params = mx.model.load_checkpoint("checkpoint/checkpoint", 10)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (10, 1191))],label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    mod.predict(eval_data=[91,23])
    #out = pd.DataFrame({'User_ID': ids, 'Is_Response': preds})
    #out.to_csv('out.csv', index=False)