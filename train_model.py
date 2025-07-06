# train_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Load data
data = pd.read_csv('customer_churn.csv')  # Adjust path as needed

# Label encoding for categorical features
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    if col != 'customerID':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# Drop missing values or fill them (depends on dataset)
data = data.dropna()

# Feature-target split
X = data.drop(columns=['customerID', 'Churn'])
y = data['Churn']

# Feature selection with RFE
rfe_selector = RFE(RandomForestClassifier(), n_features_to_select=15)
rfe_selector = rfe_selector.fit(X, y)
selected_features = X.columns[rfe_selector.support_]
X = X[selected_features]

# Balance data
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_resampled, y_resampled)

# Save model, encoders, and selected features
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
joblib.dump(selected_features.tolist(), "models/selected_features.pkl")

print("Model and encoders saved successfully.")

