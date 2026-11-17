from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


# --------------------------------------------------
# CUSTOMER INPUT SCHEMA
# --------------------------------------------------

class CustomerData(BaseModel):

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]

    SeniorCitizen: Literal[0, 1]

    Partner: Literal["Yes", "No"]

    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0, le=72)

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service"
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No"
    ]

    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(ge=0)

    TotalCharges: float | None = Field(
        default=None,
        ge=0
    )


# --------------------------------------------------
# PREDICTION RESPONSE SCHEMA
# --------------------------------------------------

class PredictionResponse(BaseModel):

    churn_prediction: Literal["Yes", "No"]

    churn_probability: float = Field(
        ge=0,
        le=1
    )

    prediction: Literal[0, 1]

    risk_category: Literal[
        "Low",
        "Medium",
        "High"
    ]

    classification_threshold: float

    retention_recommendations: list[str]