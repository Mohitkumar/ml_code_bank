import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords

vector_length = 32
max_length = 1000
np.random.seed(12)

stop = stopwords.words('english')
data = pd.read_csv("/home/mohit/Downloads/f2c2f440-8-dataset_he/test.csv")
ids = data['User_ID']
data.drop(['Browser_Used','Device_Used','User_ID'], axis=1, inplace=True)
data['Description'] = data['Description'].str.lower().str.split()

X_test = data['Description'].apply(lambda x: [item for item in x if item not in stop]).apply(lambda x: ' '.join(x))

print 'done....'
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_test)
X_test = tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)


model = Sequential()
model.add(Embedding(5000, vector_length, input_length=max_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights('happy.h5')


preds = model.predict_classes(X_test)
out_preds = []
for pr in preds:
    if pr == 0:
        out_preds.append('not happy')
    else:
        out_preds.append('happy')

out = pd.DataFrame({'User_ID':ids,'Is_Response':out_preds})
out.to_csv('out.csv', index=False)