import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

#Load the saved model
model = joblib.load('model/model.jlib')

# Instantiate our FastAPI
app = FastAPI()


class PredictionInput(BaseModel):
    ENGINESIZE : float
    CYLINDERS : float
    FUELCONSUMPTION_COMB : float


@app.post('/predict')
def predict(input_data: PredictionInput):
    # prepare input data as a numpy array
    data = np.array([[
        input_data.ENGINESIZE,
        input_data.CYLINDERS,
        input_data.FUELCONSUMPTION_COMB
    ]])

    # Make predictions

    prediction = model.predict(data)


    #returning the prediction
    return {
        'prediction' : prediction[0]
    }


#run the app with uvicorn
if __name__ == "__master__":
    uvicorn.run(app, host="0.0.0.0.", port=8000)
