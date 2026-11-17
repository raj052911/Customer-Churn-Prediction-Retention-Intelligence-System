# Customer Churn Prediction & Retention Intelligence System

## Project Overview

A machine learning-based system that predicts customer churn in the telecommunications industry. It estimates churn probability, classifies customers by risk level, and generates rule-based retention recommendations.

## Objectives

- Predict whether a customer is likely to leave.
- Analyze customer churn patterns.
- Perform data preprocessing and feature engineering.
- Train and compare machine learning models.
- Identify high-risk customers.
- Generate retention recommendations.
- Deploy predictions through a REST API.

## Technologies Used

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Storage:** Joblib
- **API:** FastAPI, Uvicorn, Pydantic
- **Testing:** Pytest
- **Version Control:** Git, GitHub

## Dataset

**IBM Telco Customer Churn Dataset**

- Records: 7,043
- Features: 21 columns
- Target variable: `Churn`
- File: `data/raw/telco_churn.csv`

## Machine Learning Models

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

Hyperparameter tuning is performed using RandomizedSearchCV and stratified cross-validation.

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC and Average Precision.

## Key Features

- Data cleaning and preprocessing
- Feature engineering
- Exploratory data analysis
- Machine learning model training and tuning
- Model comparison and selection
- Customer churn probability prediction
- Low, Medium and High risk classification
- Rule-based retention recommendations
- FastAPI deployment

## Installation

Clone the repository:

```bash
git clone https://github.com/raj052911/Customer-Churn-Prediction-Retention-Intelligence-System.git
```

Navigate to the project directory:

```bash
cd Customer-Churn-Prediction-Retention-Intelligence-System
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure the dataset is available at:

```text
data/raw/telco_churn.csv
```

## Run the Application

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

Application: `http://127.0.0.1:8000`

API Documentation: `http://127.0.0.1:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Model health check |
| POST | `/predict` | Customer churn prediction and retention recommendations |

## Testing

Run automated tests:

```bash
python -m pytest tests/ -v
```

## Project Structure

```text
api/             FastAPI application
artifacts/       Trained machine learning models
data/            Dataset files
reports/         Evaluation reports and visualizations
src/             Source code
tests/           Automated tests
requirements.txt Project dependencies
README.md        Project documentation
```

## Author

**Project:** Customer Churn Prediction & Retention Intelligence System  
**Domain:** Machine Learning and Data Science  
**Programming Language:** Python