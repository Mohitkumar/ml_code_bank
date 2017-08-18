import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder,StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


def get_train_data_raw():
    train = pd.read_csv('/home/mohit/comp_data/train.csv')
    train = train.ix[random.sample(train.index, train.size / 50)]
    return train

#after analysiss
def get_train_data_processed():
    train = pd.read_csv('/home/mohit/comp_data/train.csv')
    train.drop(labels=['siteid','devid','category','offerid'], axis=1, inplace=True)
    train = train.ix[random.sample(train.index,train.size/20)]
    train['datetime'] = pd.to_datetime(train['datetime'])
    train['thour'] = train['datetime'].dt.hour

    train['browserid'].fillna("None", inplace=True)
    for c in list(train.select_dtypes(include=['object']).columns):
        if c != 'ID':
            lbl = LabelEncoder()
            lbl.fit(list(train[c].values))
            train[c] = lbl.transform(list(train[c].values))
    cols_to_use = [x for x in train.columns if x not in list(['ID', 'datetime', 'click'])]
    scaler = StandardScaler().fit(train[cols_to_use])

    strain = scaler.transform(train[cols_to_use])
    return strain, train['click']

def get_test_data_processed():
    train = pd.read_csv('/home/mohit/comp_data/test.csv')
    train.drop(labels=['siteid','devid','category','offerid'], axis=1, inplace=True)
    train = train.ix[random.sample(train.index,train.size/20)]
    train['datetime'] = pd.to_datetime(train['datetime'])
    train['thour'] = train['datetime'].dt.hour

    train['browserid'].fillna("None", inplace=True)
    for c in list(train.select_dtypes(include=['object']).columns):
        if c != 'ID':
            lbl = LabelEncoder()
            lbl.fit(list(train[c].values))
            train[c] = lbl.transform(list(train[c].values))
    cols_to_use = [x for x in train.columns if x not in list(['ID', 'datetime'])]
    scaler = StandardScaler().fit(train[cols_to_use])

    strain = scaler.transform(train[cols_to_use])
    return strain


def check_missing(train):
    sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
    plt.show()


def explain_data(train):
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='click', y='tweekday', data=train, palette='viridis').set_title('offerid by click')
    plt.show()
    sns.boxplot(x='click', y='thour', data=train, palette='viridis').set_title('offerid by click')
    plt.show()


def check_distribution(train):
    sns.set_style('darkgrid')
    plt.figure(figsize=(10, 5))
    sns.countplot(train['click'], alpha=.80, palette=['grey', 'orange'])
    plt.title('click vs not clicked')
    plt.ylabel('# examples')
    plt.show()


def numeric_corr(train):
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(221)
    ax1.hist(train['thour'], bins=20, alpha=.50, edgecolor='black', color='teal')
    ax1.set_xlabel('thour')
    ax1.set_ylabel('# examples')
    ax1.set_title('hour click')

    ax2 = fig.add_subplot(223)
    ax2.hist(train['tweekday'], bins=20, alpha=.50, edgecolor='black', color='teal')
    ax2.set_xlabel('tweekday')
    ax2.set_ylabel('# examples')
    ax2.set_title('Week day click')

    plt.show()


def numeric_corr_click_notClick(dataset):
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(221)
    ax1.hist(dataset[dataset['click'] == 0].thour, bins=25, label='Did Not click', alpha=.50, edgecolor='black',
             color='grey')
    ax1.hist(dataset[dataset['click'] == 1].thour, bins=25, label='clicked', alpha=.50, edgecolor='black',
             color='orange')
    ax1.set_title('Hours click/not click')
    ax1.legend(loc='upper right')

    ax2 = fig.add_subplot(223)
    ax2.hist(dataset[dataset['click'] == 0].tweekday, bins=25, label='Not clicked', alpha=.50, edgecolor='black',
             color='grey')
    ax2.hist(dataset[dataset['click'] == 1].tweekday, bins=25, label='clicked', alpha=.50, edgecolor='black',
             color='orange')
    ax2.set_title('weekday clcik/not click')
    ax2.legend(loc='upper right')

    ax3 = fig.add_subplot(122)
    ax3.scatter(x=dataset[dataset['click'] == 0].thour, y=dataset[dataset['click'] == 0].tweekday,
                alpha=.50, edgecolor='black', c='grey', s=75, label='Notclicked')
    ax3.scatter(x=dataset[dataset['click'] == 1].thour, y=dataset[dataset['click'] == 1].tweekday,
                alpha=.50, edgecolor='black', c='orange', s=75, label='clicked')
    ax3.set_xlabel('thour')
    ax3.set_ylabel('tweekday')
    ax3.set_title('thour vs twekday')
    ax3.legend()
    plt.show()


def categorical_corr(dataset):
    sns.set_style('darkgrid')
    f, axes = plt.subplots(3, 2, figsize=(15, 15))
    sns.countplot(data=dataset, x='offerid', palette='viridis', ax=axes[0, 0])
    axes[0, 0].set_xlabel('offerid')
    axes[0, 0].set_ylabel('# examples')
    axes[0, 0].set_title('offerid')

    sns.countplot(data=dataset, x='category', palette='viridis', ax=axes[0, 1])
    axes[0, 1].set_xlabel('category')
    axes[0, 1].set_ylabel('# examples')
    axes[0, 1].set_title('category')

    sns.countplot(data=dataset, x='merchant', palette='viridis', ax=axes[1, 0])
    axes[1, 0].set_xlabel('merchant')
    axes[1, 0].set_ylabel('# examples')
    axes[1, 0].set_title('merchant')

    sns.countplot(data=dataset, x='countrycode', palette='viridis', ax=axes[1, 1])
    axes[1, 1].set_xlabel('countrycode')
    axes[1, 1].set_ylabel('# examples')
    axes[1, 1].set_title('countrycode')

    sns.countplot(data=dataset, x='browserid', palette='viridis', ax=axes[2, 0])
    axes[2, 0].set_xlabel('browserid')
    axes[2, 0].set_ylabel('# examples')
    axes[2, 0].set_title('browserid')
    plt.show()


def categorical_corr_click_no_click(dataset):
    sns.set_style('darkgrid')
    f, axes = plt.subplots(3, 2, figsize=(20, 15))
    countrycode = dataset.groupby(['countrycode', 'click']).countrycode.count().unstack()
    p1 = countrycode.plot(kind='bar', stacked=True,
                     title='countrycode - click vs not click',
                     color=['grey', 'orange'], alpha=.70, ax=axes[0, 0])
    p1.set_xlabel('countrycode')
    p1.set_ylabel('examples')
    p1.legend(['Not click', 'click'])
    plt.show()


def numeric_scatter(dataset):
    data = pd.concat([dataset['click'], dataset['tweekday']], axis=1)
    data.plot.scatter(x='tweekday', y='click', ylim=(0, 3));
    plt.show()


def category_scatter(dataset):
    data = pd.concat([dataset['click'], dataset['browserid']], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x='browserid', y="click", data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.show()


def corr_heatmap(dataset):
    corrmat = dataset.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    plt.show()


def corr_heatmap_click(dataset):
    corrmat = dataset.corr()
    k = 9  # number of variables for heatmap
    cols = corrmat.nlargest(k, 'click')['click'].index
    cm = np.corrcoef(dataset[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()

#drop greater than 15% missing
def get_missing_percent(dataset):
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent = (dataset.isnull().sum() / dataset.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print missing_data


if __name__ == '__main__':
    #train = get_train_data_raw()
    train = get_train_data_processed()
    print train.info()
    #check_distribution(train)
    #numeric_corr_click_notClick(train)
    #categorical_corr_click_no_click(train)
    #numeric_scatter(train)
    corr_heatmap_click(train)
    #get_missing_percent(train)