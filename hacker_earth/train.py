import xgboost as xg
from click_prediction import get_train_data_processed
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics.classification import confusion_matrix
from sklearn.metrics import roc_auc_score

def train():
    train, click = get_train_data_processed()
    X_train, X_valid, Y_train, Y_valid = train_test_split(train, click, test_size=0.2, random_state=2017)
    clf = xg.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, Y_train)
    preds = cross_val_predict(clf, X_train, Y_train, cv=3)
    print(confusion_matrix(Y_train, preds))
    print(roc_auc_score(Y_train, preds))

    predicted = clf.predict(X_valid)
    print(confusion_matrix(y_true=Y_valid, y_pred=predicted))
    print(roc_auc_score(Y_valid, predicted))

if __name__ == '__main__':
    train()

