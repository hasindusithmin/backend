import io
import os
import time
import uuid0
import shutil
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from fastapi import FastAPI, UploadFile, File, status,BackgroundTasks


app = FastAPI()
# Load the model
model = load_model('keras_model.h5')

def delete_image(name):
    time.sleep(30)
    os.remove(name)

@app.post("/upload")
async def upload_image(backgroundTasks:BackgroundTasks,file: UploadFile = File(...)):
    ext = file.content_type.split('/')[1]
    name = uuid0.generate().base62
    with open(f"image/{name}.{ext}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    content = await file.read()
    # image = Image.open(io.BytesIO(content)).convert('RGB')
    image = Image.open(f'image/{name}.{ext}').convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # # run the inference
    label = ['Aegypti','Albopictus','Other']
    prediction = model.predict(data)
    backgroundTasks.add_task(delete_image,name=f'image/{name}.{ext}')
    return list(zip(label,prediction.tolist()[0]))
