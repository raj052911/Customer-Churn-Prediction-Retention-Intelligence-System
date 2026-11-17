from fastapi import (
    FastAPI,
    Depends,
    HTTPException
)

from api.schemas import (
    CustomerData,
    PredictionResponse
)

from api.dependencies import get_prediction_model

from src.models.predict import predict_customer_churn

from src.retention.recommendations import (
    generate_retention_recommendation
)

from src.utils.logger import get_logger


logger = get_logger(__name__)


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",

    description=(
        "REST API for predicting customer churn "
        "and generating customer retention recommendations."
    ),

    version="1.0.0"
)


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API",
        "status": "Running",
        "documentation": "/docs"
    }


# --------------------------------------------------
# HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check(
    model=Depends(get_prediction_model)
):

    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# CUSTOMER CHURN PREDICTION ENDPOINT
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_churn(
    customer: CustomerData,
    model=Depends(get_prediction_model)
):

    try:

        logger.info("Received customer prediction request.")

        # Convert validated customer input into dictionary
        customer_data = customer.model_dump()

        # Generate churn prediction
        result = predict_customer_churn(
            customer_data,
            model=model
        )

        # Generate retention recommendations
        recommendation_result = generate_retention_recommendation(
            churn_probability=result["churn_probability"],
            customer_data=customer_data
        )

        # Add retention recommendations to prediction result
        result["retention_recommendations"] = (
            recommendation_result["retention_recommendations"]
        )

        logger.info(
            "Prediction and retention recommendations completed."
        )

        return result

    except (ValueError, TypeError) as error:

        logger.warning(
            "Invalid prediction input: %s",
            error
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:

        logger.exception(
            "Unexpected prediction error."
        )

        raise HTTPException(
            status_code=500,
            detail="An internal prediction error occurred."
        ) from error