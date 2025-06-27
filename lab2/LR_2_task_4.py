import numpy as np
import pandas as pd

from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import time
from tabulate import tabulate

df = pd.read_csv("income_data.csv", header=None, skipinitialspace=True)

# Перетворюємо категоріальні дані в числові
df_encoded = df.copy()
label_encoders = {}
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object" or df_encoded[col].dtype.name == "category":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

# Розділяємо дані на ознаки та мітки
X = df_encoded.iloc[:, :-1]
y = df_encoded.iloc[:, -1]  # Останній стовпець - це мітка класу

# Train on 20% of data (NOTE: test_size=0.2 to keep only 80% for training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LR": LogisticRegression(solver="liblinear"),
    "LDA": LinearDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(),
    "CART": DecisionTreeClassifier(),
    "NB": GaussianNB(),
    "SVM": SVC(gamma="auto"),
}

model_results = {}
for model_name, model in models.items():
    # Train the model and measure time
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()

    # Predict and measure time
    start_pred = time.time()
    y_pred = model.predict(X_test)
    end_pred = time.time()

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    model_results[model_name] = {
        "accuracy": accuracy,
        "train_time": end_train - start_train,
        "predict_time": end_pred - start_pred,
    }

model_results_output = []
for model_name in model_results.keys():
    model_results_output.append(
        [
            model_name,
            model_results[model_name]["accuracy"],
            model_results[model_name]["train_time"],
            model_results[model_name]["predict_time"],
        ]
    )

print(
    tabulate(
        model_results_output,
        headers=["Model", "Accuracy", "Train Time (sec)", "Predict Time (sec)"],
        tablefmt="orgtbl",
    )
)
