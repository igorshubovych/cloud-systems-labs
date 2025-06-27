import pandas as pd
import numpy as np
import time
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report


# Читаємо дані з CSV файлу
df = pd.read_csv("income_data.csv", header=None, skipinitialspace=True)

# Кодуємо категоріальні ознаки (стовпці)
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

# Розділяємо дані на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ініціюємо різні моделі SVM
models = {
    "LinearSVC": LinearSVC(max_iter=10000),
    "SVC (poly kernel)": SVC(kernel="poly"),
    "SVC (rbf kernel)": SVC(kernel="rbf"),  # default kernel (Gaussian RBF)
    "SVC (sigmoid kernel)": SVC(kernel="sigmoid"),
}

# Навчаємо кожну модель та вимірюємо час тренування
model_results = {}
for model_name, model in models.items():
    # Train the model and measure time
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()

    model_results[model_name] = {"train_time": end_train - start_train}


# Тестуємо кожну модель та оцінюємо їх продуктивність
for model_name, model in models.items():
    # Predict and measure time
    start_pred = time.time()
    y_pred = model.predict(X_test)
    end_pred = time.time()

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    rpt = classification_report(y_test, y_pred, output_dict=True)

    model_results[model_name]["accuracy"] = accuracy
    model_results[model_name]["predict_time"] = end_pred - start_pred
    model_results[model_name]["macro avg precision"] = rpt["macro avg"]["precision"]
    model_results[model_name]["macro avg recall"] = rpt["macro avg"]["recall"]
    model_results[model_name]["macro avg f1-score"] = rpt["macro avg"]["f1-score"]

# Форматуємо результати для виводу
model_results_output = []
for model_name in model_results.keys():
    result = model_results[model_name]
    model_results_output.append(
        [
            model_name,
            result["accuracy"],
            result["macro avg precision"],
            result["macro avg recall"],
            result["macro avg f1-score"],
            result["train_time"],
            result["predict_time"],
        ]
    )

# Виводимо результати в табличному форматі
print(
    tabulate(
        model_results_output,
        headers=[
            "Model",
            "Accuracy",
            "Macro Avg Precision",
            "Macro Avg Recall",
            "Macro Avg F1-score",
            "Train Time (sec)",
            "Predict Time (sec)",
        ],
        tablefmt="orgtbl",
    )
)
