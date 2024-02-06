import pandas as pd
import numpy as np
import nasdaqdatalink, datetime, math
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle

plt.style.use('ggplot')

df = nasdaqdatalink.get("WIKI/GOOG")

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# we will do high - low percent
df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] ) * 100
df['PCT_change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] ) * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna('-99999', inplace=True)

# we are going to forecast 1%  of the days out from current date
# for 1% = 0.01
# for 10 % = 0.1
forecast_out = int(math.ceil(0.1*len(df)))
# print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)


# define features and labels - https://1des.com/blog/posts/labels-features-key-of-machine-learning
# features = X (uppercase) - this represent the measurable characters or attributes that help us with predictions
# labels = y (lowercase) - this represent the possible outcome or predictions we want to make

X = np.array(df.drop(['label'], axis=1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# we can use n_jobs for number of threads (-1 uses max threads in my laptop)
clf = LinearRegression(n_jobs=-1)
# train the data on these training data set
clf.fit(X_train, y_train)

# pickle is to store trained data so, we can use it later
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

# test the data with testing data set
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)


print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
oneday = 86400
next_unix = last_unix + oneday

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += oneday
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    

# print(df.tail(40))

df['Adj. Close'].plot()
df['Forecast'].plot()

plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


