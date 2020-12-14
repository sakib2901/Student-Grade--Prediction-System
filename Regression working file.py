import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
data['romantic'] = data['romantic'].map({'yes': 1, 'no': 0})
data['schoolsup'] = data['schoolsup'].map({'yes': 1, 'no': 0})

data = data[["G1", "G2", "G3", "studytime", "romantic", "failures", "famrel", "schoolsup", "goout", "health", "absences"]]
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

best_socore = 0
for i in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    if acc > best_socore:
        best_socore = acc
print("Accuracy: ", best_socore)
print("Coefficient: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


style.use("ggplot")
plt.figure(figsize=(15, 10))
plt.scatter(data['G1'], data['G3'], c='r',  marker='+', label='G1')
plt.scatter(data['studytime'], data['G3'], c='g', marker='o', label='studytime')
plt.scatter(data['romantic'], data['G3'], c='b', marker='x', label='romantic')
plt.scatter(data['failures'], data['G3'], c='y', marker='v', label='failures')
plt.scatter(data['famrel'], data['G3'], c='c', marker='s', label='famrel')
plt.scatter(data['schoolsup'], data['G3'], c='m', marker='d', label='schoolsup')
plt.scatter(data['goout'], data['G3'], c='k', marker='>', label='goout')
plt.scatter(data['health'], data['G3'], c='indigo', marker='<', label='health')
plt.scatter(data['absences'], data['G3'], c='navy', marker='.', label='absences')
plt.ylabel('G3')
plt.legend(loc='lower right')
plt.show()


