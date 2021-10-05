from typing import Optional
from fastapi import FastAPI
from models import Request

from keras.models import load_model
import numpy as np
import pickle
from math import ceil

input_encoder = pickle.load(open('artifacts/input_encoder.pkl', 'rb'))
input_scalar = pickle.load(open('artifacts/input_scalar.pkl', 'rb'))
output_scalar = pickle.load(open('artifacts/output_scalar.pkl', 'rb'))
model = load_model('artifacts/model.h5')

app = FastAPI()

@app.post('/predict')
def predict(request: Request, q:Optional[str]=None):
    try:
        #Retrieve inputs
        bus_stop_code = np.array([[request.bus_stop_code]]) #4652946491
        day = np.array([[request.day]]) #4
        time = np.array([[request.time]]) #100

        #Perform transformation
        bus_stop_code_encoded = input_encoder.transform(bus_stop_code).reshape(1, -1)
        time_scaled = input_scalar.transform(time)

        #Perform prediction
        input = np.concatenate((bus_stop_code_encoded, day, time_scaled), axis=1)
        result = model.predict(input)

        #Inverse model scaled output
        descaled_result = output_scalar.inverse_transform(result)

        minute = 1

        #Round up prediction to nearest minute
        if int(descaled_result[0][0]) > 60:
            minute = ceil(int(descaled_result[0][0]) / 60)

        message = {
            'message' : 'Success',
            'result' : minute
        }

        return str(message)
    except Exception as e:
        print(e)

        message = {
            'message' : "Failure, please check if your inputs are valid"
        }

        return str(message)