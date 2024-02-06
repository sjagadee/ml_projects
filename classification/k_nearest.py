import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)

df.drop(['id'], axis=1, inplace=True)


X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[9,2,5,6,1,9,3,5,4]])
example_measures = example_measures.reshape(len(example_measures),-1)

predict = clf.predict(example_measures)
print(predict)
