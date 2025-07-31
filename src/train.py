import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_data, save_model, evaluate_model
from sklearn.linear_model import LinearRegression

def train_model():
    X_train, X_test, y_train, y_test = load_data()
    model = LinearRegression()
    model.fit(X_train, y_train)

    r2, mse = evaluate_model(model, X_test, y_test)
    print(f"R2 Score: {r2}, MSE: {mse}")

    save_model(model)
    return r2, mse

if __name__ == "__main__":
    train_model()
