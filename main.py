import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from PIL import ImageDraw
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
    img = Image.open(img.file)
    img = img.resize((28, 28))
    img = img.convert("L")
    img = np.array(img)
    img = img.flatten()
    img = img.reshape(1, -1)
    result = mnist_model.predict(img)
    return {"number": int(result[0])}



# Cargar los modelos previamente entrenados
svm_model = joblib.load("svm_iris_model.joblib")
dbscan_model = joblib.load("dbscan_moons_model.joblib")

@app.post("/svm/predict")
async def svm_predict(data:IrisInput):
    # Hacer predicciones utilizando el modelo SVM
    prediction = svm_model.predict(data)
    return {"prediction": prediction}

#@app.post("/dbscan/predict")
#async def dbscan_predict(data: List[Tuple[float,float]]):
#    # Hacer predicciones utilizando el modelo DBSCAN
#    prediction = dbscan_model.fit_predict(data)
 #   return {"prediction": prediction}






@app.post("/predict")
async def predict(iris_input: IrisInput):
    result = model.predict([[iris_input.sepal_length, iris_input.sepal_width,
                            iris_input.petal_length, iris_input.petal_width]])
    return {"class": result.tolist()[0]}
