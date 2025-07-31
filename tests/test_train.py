import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import os
from sklearn.linear_model import LinearRegression
from src.train import train_model
from src.utils import load_model, load_data

def test_load_data():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train is not None and X_test is not None, "Features data not loaded"
    assert y_train is not None and y_test is not None, "Target data not loaded"
    assert len(X_train) > 0 and len(X_test) > 0, "Feature datasets are empty"
    assert len(y_train) > 0 and len(y_test) > 0, "Target datasets are empty"

def test_training():
    # Run the training function
    r2, mse = train_model()
    assert os.path.exists("artifacts/model.joblib"), "Model file not saved"

    model = load_model("artifacts/model.joblib")
    assert isinstance(model, LinearRegression), "Model is not LinearRegression"
    assert hasattr(model, "coef_"), "Model is not trained properly"
    assert r2 > 0.5, f"R² score too low: {r2}"
