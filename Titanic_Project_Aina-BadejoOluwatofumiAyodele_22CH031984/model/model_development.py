import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load Dataset
df = pd.read_csv("train.csv")

# Feature Selection
selected_features = ["Pclass", "Sex", "Age", "SibSp", "Embarked", "Survived"]
df = df[selected_features]

# Handle Missing Values
df["Age"].fillna(df["Age"].median(), inplace=True)
# Embarked may have missing
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Encode Categorical Variables
le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])

le_embarked = LabelEncoder()
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

# Split Features and Target
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Optional: scale numeric features
scaler = StandardScaler()
numeric_cols = ["Age", "SibSp"]
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Train Model

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save Model and Encoders
with open("titanic_survival_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("le_sex.pkl", "wb") as f:
    pickle.dump(le_sex, f)

with open("le_embarked.pkl", "wb") as f:
    pickle.dump(le_embarked, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nModel, encoders, and scaler saved successfully!")
