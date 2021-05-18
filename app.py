import uvicorn
from fastapi import FastAPI
from banknotes import Banknote
import numpy as np
import pandas as pd
import pickle

app = FastAPI()
classifier = pickle.load(open('classifier.pkl', 'rb'))

@app.get('/')
def index():
    return {'Hi!!! This project is made by YASH PATIL'}
#expose the prediction functionality from the json data and display with a confidence(proability) value
@app.post('/predict')
def predict_banknote(data:Banknote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if (prediction[0]>0.5):
        prediction = 'Fake Note'
    else:
        prediction = 'Its a Bank Note'

    return {
        'prediction' : prediction
    } 
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn app:app --reload