import pandas as pd
import requests
from sklearn.utils import shuffle


def getToken(id):
    params = {'wooId':id}
    resp = requests.get('http://52.74.49.87:8080/woo/api/admin/generateAuthToken', params)
    token = resp.json().get('WOO_ACCESS_TOKEN')
    print token
    return token

def start():
    df = pd.read_csv('/home/mohit/discover_load_test/female.txt')
    for i, row in df.iterrows():
        df.set_value(i,'wooToken',getToken(row['id']))
    df.to_csv('/home/mohit/discover_load_test/female_token.csv', index=False)


if __name__ == '__main__':
    df = pd.read_csv('/home/mohit/discover_load_test/discover_final_users.csv')
    df = shuffle(df)
    df.to_csv('/home/mohit/discover_load_test/discover_final_users_shuff.csv', index=False)


