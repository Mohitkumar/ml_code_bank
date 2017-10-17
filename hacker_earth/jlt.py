import pandas as pd

pos_tweets = [('I love this car', 'positive'),
('This view is amazing', 'positive'),
('I feel great this morning', 'positive'),
('I am so excited about the concert', 'positive'),
('He is my best friend', 'positive')]

test = pd.DataFrame(pos_tweets)


test.columns = ["tweet","col2"]

test["tweet"] = test["tweet"].str.lower().str.split()

stop = ['love','car','amazing']

test['tweet'] = test['tweet'].apply(lambda x: [item for item in x if item not in stop]).apply(lambda x: ' '.join(x))
print test
