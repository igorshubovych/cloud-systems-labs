import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Читаємо CSV файл
df = pd.read_csv("income_data.csv", header=None, skipinitialspace=True)

count_class1 = len(df[df[:-1] == "<=50K"])
count_class2 = len(df[df[:-1] == ">50K"])
print(f"Кількість рядків з класом <=50K: {count_class1}")
print(f"Кількість рядків з класом >50K: {count_class2}")

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

# Тренуємо модель LinearSVC
model = LinearSVC(max_iter=10000)
model.fit(X_train, y_train)

# Оцінюємо точність моделі за допомогою тестового набору
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі LinearSVC: {accuracy:.2f}\n")
print("Звіт по класифікації\n", classification_report(y_test, y_pred))


# Сирі вхідні дані (з прикладу)
input_data = [
    "37",
    "Private",
    "215646",
    "HS-grad",
    "9",
    "Never-married",
    "Handlers-cleaners",
    "Not-in-family",
    "White",
    "Male",
    "0",
    "0",
    "40",
    "United-States",
]

# Перетворюємо вхідні дані в DataFrame
input_df = pd.DataFrame([input_data], columns=X.columns)

# Перетворюємо категоріальні ознаки вхідних даних
for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

# Передбачаємо клас для вхідних даних
prediction = model.predict(input_df)
predicted_class = label_encoders[df.columns[-1]].inverse_transform(prediction)

print("Вхідні дані:", input_data)
print("Передбачений клас для вхідних даних:", predicted_class[0])
