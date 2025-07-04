import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = read_csv("iris.csv", names=names)

print(dataset.shape)
print(dataset.head(20))
# Стастичні зведення методом describe
print(dataset.describe())
print(dataset.groupby("class").size())

# Діаграма розмаху
dataset.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

# Гістограма розподілу атрибутів датасета
dataset.hist()
plt.show()

# Матриця діаграм розсіювання
scatter_matrix(dataset)
plt.show()

# Розділення датасету на навчальну та контрольну вибірки

array = dataset.values
X = array[:, 0:4]  # Вибір перших 4-х стовпців
y = array[:, 4]  # Вибір 5-го стовпця
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1
)

# Завантажуємо алгоритми
models = []
models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(gamma="auto")))

# Оцінюємо модель на кожній ітерації
results = []
names = []
for name, model in models:
    model.fit(X_train, Y_train)
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

# Порівняння алгоритмів
plt.boxplot(results, labels=names)
plt.title("Порівняння алгоритмів")
plt.show()

# Створюємо прогноз на контрольній вибірці
model = SVC(gamma="auto")
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Оцінюємо прогноз
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Отримання прогнозу (застосування моделі для передбачення)
X_new = np.array([[5, 2.9, 1, 0.2]])
print("форма масиву X_new: {}".format(X_new.shape))

knn_model = next((row[1] for row in models if row[0] == "KNN"), None)
prediction = knn_model.predict(X_new)
iris_class = prediction[0].replace("Iris-", "")
print("Прогноз: {}".format(prediction))
print("Спрогнозована мітка: {}".format(iris_class))
