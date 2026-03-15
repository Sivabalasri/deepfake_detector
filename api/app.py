from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from src.inference import predict_image

app = FastAPI()

# Serve static files (CSS/JS if needed later)
app.mount("/static", StaticFiles(directory="web"), name="static")

# Root route -> open website
@app.get("/")
def read_root():
    return FileResponse(Path("web/index.html"))

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return predict_image(file)