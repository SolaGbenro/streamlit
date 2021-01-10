import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

# save model
pickle.dump(clf, open('iris_rf_clf.pkl', 'wb'))
