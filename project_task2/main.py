from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ml.predict import prediction

import os 
print(os.getcwd())

app = FastAPI()
INDICES = "./ml/model/classes_indices.pkl"
MODEL = "./ml/model/fruit_classification.h5"

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def upload_file(img : UploadFile, request: Request):
    label = prediction(img_path=img.file, model_path=MODEL, indices_path=INDICES)
    return  templates.TemplateResponse("result.html", {"request":request, "result": label})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000, host="0.0.0.0")
