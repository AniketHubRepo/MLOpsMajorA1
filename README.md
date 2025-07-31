# MLOpsMajorA1
MLOps Major Assignment or Exam


## Project Overview
This project implements a complete MLOps pipeline for the **California Housing dataset** using **Linear Regression**. The pipeline includes:

- Model Training and Evaluation
- Docker Containerization for Reproducibility
- CI/CD Automation with GitHub Actions
- Model Optimization through **Manual Quantization**

---

## Objectives
- Train a **Linear Regression** model using `scikit-learn` on the California Housing dataset.
- Save the trained model as `model.joblib`.
- Containerize the application using **Docker** for consistent execution across environments.
- Configure **GitHub Actions** workflows for:
  - Unit Testing
  - Model Training & Quantization
  - Docker Build and Verification
- Perform **manual quantization** to reduce model size while comparing accuracy.

---

## Directory Structure
```
.
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── src/
│   ├── predict.py
│   ├── quantize.py
│   ├── train.py
│   └── utils.py
│
├── tests/
│   └── test_train.py
│
├── artifacts/
│   ├── model.joblib
│   ├── unquant_params.joblib
│   └── quant_params.joblib
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Setup Instructions

### 1. Create conda Environment
```bash
conda create -n mlops5 python=3.10 -y
conda activate mlops5
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training
```bash
python -m src.train
```

### 4. Run Quantization
```bash
python -m src.quantize
```

### 5. Run Prediction for Verification
```bash
python -m src.predict
```

---

## Docker Instructions

### Build Docker Image
```bash
docker build -t mlops-pipeline .
```

### Run Docker Container
```bash
docker run --rm mlops-pipeline
```

**Expected Output:**
```
Trained Model loaded successfully.
Few Sample predictions: [0.71912284 1.76401657 2.70965883 2.83892593 2.60465725]
```

---

## GitHub Actions Workflow
- **File:** `.github/workflows/ci.yml`
- **Jobs:**
  1. **test_suite** → Runs `pytest` for data loading, model training validation.
  2. **train_and_quantize** → Trains model and quantizes it.
  3. **build_and_test_container** → Builds Docker image and verifies `predict.py` inside container.

Uses **artifact v4** for uploading and downloading files.

---

## Branching Strategy
- **main:** All codes are moved here

---

## Results- Model Comparision Table
| Metric        | Original Model | Quantized Model |
|--------------|---------------|-----------------|
| **R² Score** | 0.5758        | 0.0954         |
| **Model Size** | ~68 KB      | ~0.31 KB       |

**Conclusion:**
Quantization significantly reduces model size but impacts accuracy when using 8-bit integers.

---

## Author
**Name:** Aniket Srivastava  
**Roll No:** G24AI1077  
