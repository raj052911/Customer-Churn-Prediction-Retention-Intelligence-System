# Customer Churn Prediction & Retention Intelligence System

## 1. Project Overview

The Customer Churn Prediction & Retention Intelligence System is a machine learning-based application designed to predict whether a customer is likely to leave a telecommunications service.

The system analyzes customer information, identifies churn patterns, predicts churn probability, classifies customers into risk categories, and provides rule-based retention recommendations.

The main objective is to help businesses identify customers who are at risk of leaving and take proactive steps to improve customer retention.

## 2. Project Objectives

- Analyze customer data to identify churn patterns.
- Perform data cleaning and preprocessing.
- Handle missing values and categorical variables.
- Perform feature engineering to improve model performance.
- Train multiple machine learning classification models.
- Tune model hyperparameters using cross-validation.
- Evaluate models using classification metrics.
- Select the best-performing model.
- Predict customer churn probability.
- Classify customers into Low, Medium, and High risk categories.
- Generate personalized, rule-based retention recommendations.
- Deploy the prediction system using FastAPI.

## 3. Technologies Used

### Programming Language
- Python

### Data Analysis
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn

### Model Storage
- Joblib

### API Development
- FastAPI
- Uvicorn
- Pydantic

### Testing
- Pytest

### Development Tools
- Visual Studio Code
- Git
- GitHub

## 4. Machine Learning Models

The project uses three classification algorithms:

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

Hyperparameter tuning is performed using RandomizedSearchCV with stratified cross-validation.

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Average Precision

The best-performing model is selected based on evaluation results.

## 5. Dataset

**Dataset:** IBM Telco Customer Churn Dataset

**Dataset file:** `telco_churn.csv`

**Dataset dimensions:** 7,043 rows and 21 columns.

The dataset contains customer information such as:

- Customer ID
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Monthly Charges
- Total Charges
- Churn Status

The target variable is `Churn`.

- Yes: Customer has left the service.
- No: Customer has continued the service.

Place the original dataset inside:

`data/raw/telco_churn.csv`

The original dataset is kept unchanged.

## 6. Project Structure

```text
Customer Churn Prediction & Retention Intelligence System/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── dependencies.py
│
├── artifacts/
│   └── Saved machine learning models
│
├── data/
│   ├── raw/
│   │   └── telco_churn.csv
│   └── processed/
│
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   ├── classification_report.json
│   └── feature_importance.csv
│
├── tests/
│   └── test_api.py
│
├── src/
│   ├── analysis/
│   │   ├── exploratory_analysis.py
│   │   ├── statistical_analysis.py
│   │   ├── model_explainability.py
│   │   └── final_evaluation.py
│   │
│   ├── data/
│   │   ├── load_data.py
│   │   └── validate_data.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   ├── tune.py
│   │   ├── compare_models.py
│   │   └── select_best_model.py
│   │
│   ├── preprocessing/
│   │   └── preprocessing.py
│   │
│   ├── retention/
│   │   └── recommendations.py
│   │
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── requirements.txt
├── .gitignore
├── README.md
└── run.py
```

## 7. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

### Step 2: Navigate to the Project Directory

```bash
cd Customer-Churn-Prediction-Retention-Intelligence-System
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Add the Dataset

Place `telco_churn.csv` inside:

```text
data/raw/telco_churn.csv
```

## 8. Model Training and Execution

Run the following commands from the project root directory.

### Step 1: Train the Machine Learning Models

```bash
python -m src.models.train
```

### Step 2: Perform Hyperparameter Tuning

```bash
python -m src.models.tune
```

### Step 3: Compare the Tuned Models

```bash
python -m src.models.compare_models
```

### Step 4: Select the Best Model

```bash
python -m src.models.select_best_model
```

The selected model is saved as:

```text
artifacts/best_model.joblib
```

### Step 5: Generate Model Explainability Results

```bash
python -m src.analysis.model_explainability
```

### Step 6: Perform Final Model Evaluation

```bash
python -m src.analysis.final_evaluation
```

## 9. Running the FastAPI Application

Start the API server:

```bash
uvicorn api.main:app --reload
```

The application runs locally at:

`http://127.0.0.1:8000`

### API Documentation

Open the interactive Swagger documentation:

`http://127.0.0.1:8000/docs`

### Available API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Checks API and model status |
| POST | `/predict` | Predicts customer churn and provides retention recommendations |

## 10. Prediction Output

The prediction endpoint returns:

- **Churn Prediction:** Yes or No.
- **Churn Probability:** Probability of customer churn.
- **Prediction:** Binary prediction (0 or 1).
- **Risk Category:** Low, Medium, or High.
- **Classification Threshold:** Prediction threshold.
- **Retention Recommendations:** Suggested actions to improve customer retention.

### Risk Classification

| Churn Probability | Risk Category |
|---|---|
| Below 0.30 | Low |
| 0.30 to below 0.60 | Medium |
| 0.60 and above | High |

## 11. Testing

Run automated tests using:

```bash
python -m pytest tests/ -v
```

The tests verify:

- API home endpoint.
- Model health endpoint.
- Customer churn prediction.
- Invalid customer input handling.
- Missing field validation.
- Retention recommendation generation.

## 12. Generated Reports and Visualizations

The project generates analytical reports and visualizations, including:

- Customer churn distribution.
- Customer tenure analysis.
- Monthly charges analysis.
- Contract analysis.
- Correlation heatmap.
- Model comparison report.
- Feature importance report.
- Confusion matrix.
- ROC curve.
- Precision-Recall curve.
- Classification report.

Generated reports are stored inside the `reports/` directory.

## 13. Future Enhancements

- Develop a web-based dashboard for customer churn monitoring.
- Add customer segmentation.
- Improve retention recommendations using business feedback.
- Introduce model monitoring and periodic retraining.
- Add historical churn trend analysis.

## 14. Author

**Project:** Customer Churn Prediction & Retention Intelligence System

**Domain:** Machine Learning, Data Science and API Development

**Language:** Python

---

*This project demonstrates an end-to-end machine learning workflow, from customer data preprocessing and model training to churn prediction, retention recommendations, and API deployment.*