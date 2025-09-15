import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
import joblib

# -----------------------------
# Step 1: Generate synthetic dataset
# -----------------------------
np.random.seed(42)
n_samples = 5000

data = pd.DataFrame({
    "CustomerId": np.arange(1, n_samples + 1),
    "CreditScore": np.random.randint(300, 850, size=n_samples),
    "Age": np.random.randint(18, 70, size=n_samples),
    "Tenure": np.random.randint(0, 10, size=n_samples),
    "Balance": np.random.uniform(0, 250000, size=n_samples),
    "NumOfProducts": np.random.randint(1, 4, size=n_samples),
    "HasCrCard": np.random.randint(0, 2, size=n_samples),
    "IsActiveMember": np.random.randint(0, 2, size=n_samples),
    "EstimatedSalary": np.random.uniform(10000, 150000, size=n_samples)
})

# Define churn based on risk factors
data["Churn"] = (
    (data["CreditScore"] < 500).astype(int) +
    (data["Age"] > 50).astype(int) +
    (data["IsActiveMember"] == 0).astype(int) +
    (data["Balance"] > 100000).astype(int)
)

# Make it binary: churn if >= 2 risk factors
data["Churn"] = (data["Churn"] >= 2).astype(int)

print("Sample data:")
print(data.head())

# -----------------------------
# Step 2: Train LightGBM model
# -----------------------------
X = data.drop(columns=["CustomerId", "Churn"])
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

print("\nTraining model...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    num_boost_round=200,
    # early_stopping_rounds=20
)

# -----------------------------
# Step 3: Evaluate model
# -----------------------------
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nEvaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# -----------------------------
# Step 4: Save model
# -----------------------------
joblib.dump(model, "bank_churn_model.pkl")
print("\n✅ Model saved as bank_churn_model.pkl")
