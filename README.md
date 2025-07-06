# Customer-Churn-Prediction
Customer churn is a critical problem in the telecom industry, where retaining customers is often more cost-effective than acquiring new ones. This project aims to predict whether a customer is likely to churn using machine learning based on their usage behavior, account information, and demographic data.

We trained a model using the XGBoost classifier, combined with SMOTE-ENN to handle class imbalance. The system is deployed via a Streamlit web interface, and a Jupyter Notebook is provided for manual testing with custom inputs.

⚙️ **Technologies Used**

    Python

    Pandas, NumPy for data manipulation

    Scikit-learn for preprocessing and feature selection

    XGBoost for model training

    SMOTE-ENN (imblearn) for handling imbalanced data

    Streamlit for deployment

    Jupyter Notebook for manual input and testing

    Joblib for model serialization

📦 **Features**

    Data preprocessing with label encoding

    Feature selection using Recursive Feature Elimination (RFE)

    Class imbalance handled using SMOTE-ENN

    Model training with XGBoost

    Trained model saved with encoders and selected features

    Streamlit app with:

        Customer ID lookup

        Manual input for unseen users

        Churn prediction with confidence probability

    Manual prediction via Jupyter Notebook for testing scenarios

📈 **Model Performance**

The model achieved the following on a held-out test set:

    Accuracy: 95.4%

    Precision: 95%

    Recall: 97%

    F1-Score: 96%

This indicates strong performance and low false-negative rate, which is crucial in churn prediction.

✅ **Use Cases**

    Telecom companies can identify at-risk customers and take proactive retention actions.

    Marketing and customer support teams can prioritize engagement.

    Academic use for understanding machine learning with imbalanced data.



## Features

- 📊 Model training with feature selection and class balancing
- 🧠 XGBoost classifier trained on telecom dataset
- 🌐 Streamlit-based UI for real-time prediction
- 🧪 Jupyter Notebook for manual input testing
- ✅ Probability-based prediction output
- 💾 Includes saved model and encoders

## Project Structure

```
customer-churn-prediction/
│
├── models/
│   ├── churn_model.pkl
│   ├── label_encoders.pkl
│   └── selected_features.pkl
│
├── data/
│   └── customer_churn.csv
│
├── app.py                  # Streamlit web application
├── train_model.py          # Script for model training and saving
├── create.ipynb            # Jupyter notebook for testing and manual input
├── churn_manual_input_test.txt
├── churn_full_code_with_output.txt
├── README.md               # Project overview
└── requirements.txt        # Python dependencies
```

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py


## Files
- `train_model.py` – model training and saving
- `app.py` – Streamlit UI
- `create.ipynb` – manual input and testing
- `models/` – saved model and encoders
- `data/` – raw data



