from sklearn.model_selection import StratifiedKFold
import numpy as np
X = np.ones(10)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
skf = StratifiedKFold(n_splits=3)
for i,(train, test) in enumerate(list(skf.split(X, y))):
   print("%s %s" % (y[train], y[test]))

