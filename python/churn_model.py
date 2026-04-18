# ================================
# Customer Churn AI Project
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ================================
# 1. Load Data
# ================================
print("Loading data...")

df = pd.read_csv("../data/customer_churn_clean.csv")

print("Data Loaded Successfully")
print(df.head())


# ================================
# 2. Data Cleaning
# ================================
print("\nCleaning data...")

# Remove missing values
df.dropna(inplace=True)

# Fix TotalCharges column
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Convert target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Remove ID column safely
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Drop rows after conversion
df.dropna(inplace=True)

print("Cleaning completed")


# ================================
# 3. Encoding
# ================================
print("\nEncoding categorical variables...")

df = pd.get_dummies(df, drop_first=True)

# Ensure all numeric
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# Convert everything to float for SHAP compatibility
df = df.astype(np.float64)

# Check for non-numeric columns
non_numeric = df.select_dtypes(include=['object']).columns
print("Remaining non-numeric columns:", non_numeric)


# ================================
# 4. Train/Test Split
# ================================
X = df.drop('Churn', axis=1).astype(np.float64)
y = df['Churn'].astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ================================
# 5. Train Model
# ================================
print("\nTraining model...")

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)


# ================================
# 6. Feature Importance
# ================================
print("\nGenerating feature importance plot...")

importance = model.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(X.columns, importance)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("Feature importance saved as feature_importance.png")


# ================================
# 7. SHAP Explainability
# ================================
print("\nRunning SHAP analysis (this may take a few seconds)...")

X_sample = X.sample(200, random_state=42)

# Disable strict additivity checking for TreeExplainer probability outputs.
explainer = shap.Explainer(model, X, feature_names=X.columns)
shap_values = explainer(X_sample, check_additivity=False)[..., 1]

shap.plots.beeswarm(shap_values, show=False)

plt.savefig("shap_summary.png", bbox_inches='tight')
plt.close()

print("SHAP summary saved as shap_summary.png")


# ================================
# 8. Predictions
# ================================
print("\nGenerating predictions...")

df['churn_probability'] = model.predict_proba(X)[:, 1]


# ================================
# 9. Save Output
# ================================
df.to_csv("../data/churn_predictions.csv", index=False)

print("\n✅ Predictions saved as churn_predictions.csv")
print("Project completed successfully!")