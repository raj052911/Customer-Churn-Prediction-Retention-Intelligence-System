from src.models.predict import load_model


# --------------------------------------------------
# MODEL DEPENDENCY
# --------------------------------------------------

def get_prediction_model():

    model = load_model()

    return model