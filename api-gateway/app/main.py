from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.routes import router


app = FastAPI(
    title="Trend Prediction API Gateway",
    version="1.0.0"
)

app.include_router(router)


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_PATH = BASE_DIR / "static" / "index.html"


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(FRONTEND_PATH)
