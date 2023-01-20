import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from typing import List, Tuple




app = FastAPI()
model = joblib.load("model.pkl")






class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float



mnist_model = joblib.load("mnist_model.pkl")

@app.post("/mnist")
async def predict_mnist(img: UploadFile = File(...)):
    # Open the image file
    img = Image.open(img.file)
    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert the image to grayscale
    img = img.convert("L")
    # Convert the image to a numpy array
    img = np.array(img)
    # Flatten the array for the model
    img = img.flatten()
    # Reshape the array to (1, 784)
    img = img.reshape(1, -1)
    # Make a prediction using the MNIST model
    result = mnist_model.predict(img)
    # Return the predicted number as an integer
    return {"number": int(result[0])}



# Cargar los modelos previamente entrenados
svm_model = joblib.load("svm_iris_model.joblib")
dbscan_model = joblib.load("dbscan_moons_model.joblib")

@app.post("/svm/predict")
async def svm_predict(data:IrisInput):
    # Hacer predicciones utilizando el modelo SVM
    prediction = svm_model.predict(data)
    return {"prediction": prediction}

@app.post("/dbscan/predict")
async def dbscan_predict(data: List[Tuple[float,float]]):
    # Hacer predicciones utilizando el modelo DBSCAN
    prediction = dbscan_model.fit_predict(data)
    return {"prediction": prediction}



@app.post("/dbscan_2_2/predict")
async def dbscan_predict(data: List[Tuple[float,float]]):
    # Hacer predicciones utilizando el modelo DBSCAN
    prediction = dbscan_model.fit_predict(data)
    return {"prediction": prediction}





@app.post("/dbscan_3_2/predict")
async def dbscan_predict(data: List[Tuple[float,float]]):
    # Hacer predicciones utilizando el modelo DBSCAN
    prediction = dbscan_model.fit_predict(data)
    return {"prediction": prediction}



@app.post("/predict")
async def predict(iris_input: IrisInput):
    result = model.predict([[iris_input.sepal_length, iris_input.sepal_width,
                            iris_input.petal_length, iris_input.petal_width]])
    return {"class": result.tolist()[0]}
