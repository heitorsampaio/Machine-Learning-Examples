import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import pickle

data = pd.read_csv('Beginner/data/student-mat.csv', sep=';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, Y, test_size=0.1)

best = 0

# Pick the highest acc
for _ in range(30):
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best:
        best = acc
        with open('Beginner/data/studentLmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)

pickle_in = open('Beginner/data/studentLmodel.pickle', 'rb')
linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(f'Predicted Grade: {predictions[x]}, Original Grade: {y_test[x]}')
