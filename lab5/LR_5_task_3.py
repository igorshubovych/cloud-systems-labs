import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from utilities import visualize_classifier
import statistics

input_file = "data_random_forests.txt"
data = np.loadtxt(input_file, delimiter=",")
X, y = data[:, :-1], data[:, -1]

# Розбиття даних на три класи на підставі міток
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])
class_2 = np.array(X[y == 2])

# Розбиття даних на начальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)

# Визначення сітки значень параметрів
parameter_grid = [
    {"n_estimators": [100], "max_depth": [2, 4, 7, 12, 16]},
    {"max_depth": [4], "n_estimators": [25, 50, 100, 250]},
]

metrics = ["precision_weighted", "recall_weighted"]

for metric in metrics:
    print("\n##### Searching optimal parameters for ", metric)
    classifier = GridSearchCV(
        ExtraTreesClassifier(random_state=0), parameter_grid, cv=5, scoring=metric
    )
    classifier.fit(X_train, y_train)

print("\nGrid scores for the parameter grid:")
for params in classifier.cv_results_.keys():
    # avg_score = statistics.mean(classifier.cv_results_[param])
    value = classifier.cv_results_[params]
    avg_score = 0
    try:
        avg_score = statistics.mean(value)
    except TypeError:
        avg_score = 0
    print("MEAN(", params, ") = ", avg_score)
    print("\nBest parameters:", classifier.best_params_)

y_pred = classifier.predict(X_test)
print("\nPerformance report: \n")
print(classification_report(y_test, y_pred))
