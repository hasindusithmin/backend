import shutil
import uuid0
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from fastapi import FastAPI, UploadFile, File, status


app = FastAPI()
# Load the model
model = load_model('keras_model.h5')

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    ext = file.content_type.split('/')[1]
    name = uuid0.generate().base62
    with open(f"image/{name}.{ext}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

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
    label = ['Class1','Class2']
    prediction = model.predict(data)
    return list(zip(label,prediction.tolist()[0]))
