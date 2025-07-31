import os
import joblib
import numpy as np
from sklearn.metrics import r2_score
from src.utils import load_model, load_data

def quantize_model():
    model = load_model("artifacts/model.joblib")
    print("Loaded trained model from artifacts/model.joblib")

    coef = model.coef_
    intercept = model.intercept_

    os.makedirs("artifacts", exist_ok=True)

    joblib.dump({"coef": coef, "intercept": intercept}, "artifacts/unquant_params.joblib")
    print("Saved unquantized parameters at artifacts/unquant_params.joblib")

    quant_coef = np.clip(coef * 100, 0, 255).astype(np.uint8)
    quant_intercept = np.clip(intercept * 100, 0, 255).astype(np.uint8)

    joblib.dump({"coef": quant_coef, "intercept": quant_intercept}, "artifacts/quant_params.joblib")
    print("Saved quantized parameters at artifacts/quant_params.joblib")

    dequant_coef = quant_coef.astype(np.float32) / 100
    dequant_intercept = quant_intercept.astype(np.float32) / 100

    _, X_test, _, y_test = load_data()

    original_pred = model.predict(X_test)
    quant_pred = np.dot(X_test, dequant_coef) + dequant_intercept

    r2_original = r2_score(y_test, original_pred)
    r2_quantized = r2_score(y_test, quant_pred)

    print("\n Sample predictions with de-quantized weights:", quant_pred[:5])
    print(f"Original Model R² Score: {r2_original:.4f}")
    print(f"Quantized Model R² Score: {r2_quantized:.4f}")

if __name__ == "__main__":
    quantize_model()
