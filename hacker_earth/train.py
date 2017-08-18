import xgboost as xg
from click_prediction import get_train_data_processed
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics.classification import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier, RandomForestClassifier

def train():
    train, click = get_train_data_processed()
    X_train, X_valid, Y_train, Y_valid = train_test_split(train, click, test_size=0.2, random_state=2017)
    print "started train"
    estimators = []
    model1 = xg.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05, silent=False,
                           min_child_weight=1, scale_pos_weight=1)
    estimators.append(('xgb', model1))

    model2 = ExtraTreesClassifier(n_estimators=300)
    estimators.append(('ext', model2))

    model3 = RandomForestClassifier(n_estimators=300)
    estimators.append(('rndf', model3))

    clf = VotingClassifier(estimators)
    clf.fit(X_train, Y_train)
    print "started cross validation"
    #preds = cross_val_predict(clf, X_train, Y_train, cv=3)
    #print(confusion_matrix(Y_train, preds))
    #print(roc_auc_score(Y_train, preds))
    print "started predict"
    predicted = clf.predict(X_valid)
    print(confusion_matrix(y_true=Y_valid, y_pred=predicted))
    print(roc_auc_score(Y_valid, predicted))

if __name__ == '__main__':
    train()

