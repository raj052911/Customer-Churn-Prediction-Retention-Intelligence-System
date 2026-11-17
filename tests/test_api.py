from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


# --------------------------------------------------
# SAMPLE CUSTOMER DATA
# --------------------------------------------------

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 1026.00
}


# --------------------------------------------------
# TEST 1: HOME ENDPOINT
# --------------------------------------------------

def test_home_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "Running"

    assert data["message"] == "Customer Churn Prediction API"


# --------------------------------------------------
# TEST 2: HEALTH ENDPOINT
# --------------------------------------------------

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["model_loaded"] is True


# --------------------------------------------------
# TEST 3: CUSTOMER PREDICTION
# --------------------------------------------------

def test_customer_prediction():

    response = client.post(
        "/predict",
        json=SAMPLE_CUSTOMER
    )

    assert response.status_code == 200

    data = response.json()

    # Check churn prediction
    assert data["churn_prediction"] in ["Yes", "No"]

    # Check churn probability
    assert 0 <= data["churn_probability"] <= 1

    # Check binary prediction
    assert data["prediction"] in [0, 1]

    # Check risk category
    assert data["risk_category"] in [
        "Low",
        "Medium",
        "High"
    ]

    # Check classification threshold
    assert 0 <= data["classification_threshold"] <= 1

    # Check retention recommendations
    assert isinstance(
        data["retention_recommendations"],
        list
    )

    assert len(
        data["retention_recommendations"]
    ) > 0


# --------------------------------------------------
# TEST 4: INVALID CUSTOMER INPUT
# --------------------------------------------------

def test_invalid_customer_input():

    invalid_customer = SAMPLE_CUSTOMER.copy()

    invalid_customer["gender"] = "Unknown"

    response = client.post(
        "/predict",
        json=invalid_customer
    )

    assert response.status_code == 422


# --------------------------------------------------
# TEST 5: MISSING CUSTOMER FIELD
# --------------------------------------------------

def test_missing_customer_field():

    incomplete_customer = SAMPLE_CUSTOMER.copy()

    del incomplete_customer["tenure"]

    response = client.post(
        "/predict",
        json=incomplete_customer
    )

    assert response.status_code == 422


# --------------------------------------------------
# TEST 6: RETENTION RECOMMENDATION VALIDATION
# --------------------------------------------------

def test_retention_recommendations():

    response = client.post(
        "/predict",
        json=SAMPLE_CUSTOMER
    )

    assert response.status_code == 200

    data = response.json()

    recommendations = data["retention_recommendations"]

    assert isinstance(recommendations, list)

    assert all(
        isinstance(item, str)
        for item in recommendations
    )

    assert len(recommendations) > 0