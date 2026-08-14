from fastapi import APIRouter
from app.schema import modelInput
from app.model import load_model
import pandas as pd

router = APIRouter()


@router.get('/')
def homepage():
    return{
        'message':'API is running'
    }

@router.post('/predict')
def predict(data : modelInput):

    model = load_model()

    df = pd.DataFrame([data.dict()])

    prediction = model.predict(df)
    if prediction[0] == 0:
         message = "The accident is predicted to have low severity."
    elif prediction[0] == 1:
        message = "The accident is predicted to have moderate severity."
    elif prediction[0] == 2:
        message = "The accident is predicted to have high severity."
    elif prediction[0] == 3:
        message = "The accident is predicted to have very high severity."
    else:
        message = "Prediction unavailable."

    return {
    "result": message
    }
