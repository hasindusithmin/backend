from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
from starlette.responses import FileResponse

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#set model
MODEL = tf.keras.models.load_model("keras_model.h5")

#predicted class
CLASS_NAMES = ['Aegypti','Albopictus','Other']

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=33, sigmaY=33)
    divide = cv2.divide(gray, blur, scale=255)
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    test_img=cv2.resize(thresh,(224,224))
    test_img=test_img/255
    test_img=test_img.reshape(1,224,224,1)   
    predictions = MODEL.predict(test_img)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence1 = np.argmax(predictions[0])
    return {"predicted_class":predicted_class,"confidence":float(confidence1)}
    
    
    
    
