
from fastapi import FastAPI, UploadFile, File


app = FastAPI()

@app.post("/upload")
def upload_img(file: UploadFile = File(...)):
    name = file.filename
    f = file.file
    res = drive.put(name, f)
    return res