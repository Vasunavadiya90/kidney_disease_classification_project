from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse


from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier.pipeline.prediction import PredictionPipeline


from fastapi.responses import HTMLResponse
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()

        image = data["image"]

        image_path = "inputImage.jpg"

        decodeImage(image, image_path)

        classifier = PredictionPipeline(image_path)

        result = classifier.predict()

        return result

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }




@app.get("/health")
async def health():
    return {"status": "healthy"}