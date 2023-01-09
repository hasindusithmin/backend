import io
from fastapi import FastAPI,File,UploadFile   
from fastapi.responses import StreamingResponse
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from fastapi.middleware.cors import CORSMiddleware
import keras_preprocessing

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the save model
model = tf.keras.models.load_model('keras_model.h5')


@app.post("/")
async def upload_file(file:UploadFile):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert('RGB')
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # expand tha array with another demention
    test_image = np.expand_dims(image_array, axis=0)

    label = ['Aegypti','Albopictus','Other']
    prediction = model.predict(test_image)
    return list(zip(label,prediction.tolist()[0]))