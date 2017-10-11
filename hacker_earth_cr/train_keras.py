import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

np.random.seed(12)

data = pd.read_csv("/home/mohit/Downloads/f2c2f440-8-dataset_he/train.csv")
data.drop(['Browser_Used','Device_Used','User_ID'], axis=1, inplace=True)

train, test = train_test_split(data, test_size=0.2)

X_train, y_train = train['Description'], train['Is_Response']
X_test, y_test = test['Description'], test['Is_Response']
y_train_fact = pd.factorize(y_train)
y_test_fact = pd.factorize(y_test)

print y_train_fact[0][0], y_train_fact[1][0]
print y_test_fact[0][0], y_test_fact[1][0]

y_train = y_train_fact[0]
y_test = y_test_fact[0]


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)

tokenizer.fit_on_texts(X_test)
X_test = tokenizer.texts_to_sequences(X_test)

max_length = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)
print X_train.shape
vector_length = 32

model = Sequential()
model.add(Embedding(5000, vector_length, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print model.summary()

model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))