import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
from src.utils import load_model, load_data

def main():
    model = load_model("artifacts/model.joblib")
    print("Trained Model loaded successfully.")

    _, X_test, _, _ = load_data()

    preds = model.predict(X_test[:5])
    print("Few Sample predictions:", preds)

if __name__ == "__main__":
    main()
