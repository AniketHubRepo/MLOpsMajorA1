import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os

def load_data(test_size=0.2, random_state=42):
    data = fetch_california_housing()
    return train_test_split(data.data, data.target, test_size=test_size, random_state=random_state)

def save_model(model, path="artifacts/model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path="artifacts/model.joblib"):
    return joblib.load(path)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)
