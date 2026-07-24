import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("diabetes (2).csv")

print(df.head())
print(df.info())

# Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Handle Missing Values
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred))

# Save Model
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")

print("Model Saved Successfully!")
