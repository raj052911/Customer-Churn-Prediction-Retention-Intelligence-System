from src.utils.logger import get_logger


logger = get_logger(__name__)


# --------------------------------------------------
# RETENTION RECOMMENDATION ENGINE
# --------------------------------------------------

def generate_retention_recommendation(
    churn_probability,
    customer_data
):

    logger.info("Generating retention recommendation.")

    contract = customer_data.get("Contract")

    tenure = customer_data.get("tenure")

    monthly_charges = customer_data.get("MonthlyCharges")

    recommendations = []

    # HIGH RISK
    if churn_probability >= 0.60:

        risk_category = "High"

        recommendations.append(
            "Prioritize this customer for retention outreach."
        )

        recommendations.append(
            "Offer a personalized retention incentive."
        )

        if contract == "Month-to-month":

            recommendations.append(
                "Encourage migration to a longer-term contract."
            )

        if monthly_charges >= 80:

            recommendations.append(
                "Review pricing and offer a suitable plan."
            )

    # MEDIUM RISK
    elif churn_probability >= 0.30:

        risk_category = "Medium"

        recommendations.append(
            "Schedule a customer satisfaction follow-up."
        )

        recommendations.append(
            "Offer relevant service improvements or discounts."
        )

        if tenure < 12:

            recommendations.append(
                "Provide onboarding assistance and early engagement."
            )

    # LOW RISK
    else:

        risk_category = "Low"

        recommendations.append(
            "Maintain regular customer engagement."
        )

        recommendations.append(
            "Offer loyalty benefits and service updates."
        )

    result = {

        "risk_category": risk_category,

        "churn_probability": round(
            float(churn_probability),
            4
        ),

        "retention_recommendations": recommendations

    }

    logger.info("Retention recommendations generated.")

    return result