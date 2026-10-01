import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("loan_data.csv")

print("\nDataset columns:")
print(df.columns.tolist())

# Remove missing rows
df = df.dropna()

# =========================
# INPUT AND TARGET
# =========================

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# =========================
# FEATURES
# =========================

categorical_features = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area"
]

numeric_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

# =========================
# PREPROCESSING
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

# =========================
# MODEL
# =========================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]
)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================
# TRAIN
# =========================

model.fit(X_train, y_train)

# =========================
# TEST
# =========================

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\n==============================")
print("MODEL TRAINED SUCCESSFULLY")
print("==============================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, prediction))

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "loan_model.pkl")

print("\nloan_model.pkl created successfully!")