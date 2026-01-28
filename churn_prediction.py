import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

# 1) Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# 2) Drop customerID (not useful)
df.drop("customerID", axis=1, inplace=True)

# 3) Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 4) Convert target column Churn into 0/1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 5) Convert categorical columns to numeric using One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# 6) Split into features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 7) Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.preprocessing import StandardScaler

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
 n

# 8) Train Logistic Regression
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# 9) Prediction
y_pred = model.predict(X_test)

# 10) Evaluation
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# 11) Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Churn Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
