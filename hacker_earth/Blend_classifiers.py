import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from click_prediction import get_train_data_processed, get_test_data_processed
from xgboost import XGBClassifier

np.random.seed(0)
n_folds = 10
verbose = True
shuffle = False

if __name__ == '__main__':
    X, y = get_train_data_processed();
    print X.shape, y.shape
    X_submission = get_test_data_processed()
    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]
    skf = list(StratifiedKFold(n_splits=n_folds).split(X, y))

    clfs = [XGBClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=300),
            RandomForestClassifier(n_estimators=300, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=300, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=300, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=300, n_jobs=-1, criterion='entropy')]

    print "Creating train and test sets for blending."

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print
    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print "Saving Results."
    tmp = np.vstack([range(1, len(y_submission) + 1), y_submission]).T
    np.savetxt(fname='submission.csv', X=tmp, fmt='%d,%0.9f',
               header='clickId,prob', comments='')