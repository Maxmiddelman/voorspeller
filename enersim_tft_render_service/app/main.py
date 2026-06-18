from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.services.tft_forecast_service import train_tft_for_site, forecast_tft_for_site

app = FastAPI(title="EnerSim TFT Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SiteRequest(BaseModel):
    site_id: str

@app.get("/")
def root():
    return {"status": "online", "service": "enersim-tft"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train(req: SiteRequest):
    try:
        return train_tft_for_site(req.site_id)
    except Exception as e:
        return JSONResponse(status_code=500, content={"site_id": req.site_id, "error": str(e)})

@app.post("/forecast")
def forecast(req: SiteRequest):
    try:
        result = forecast_tft_for_site(req.site_id)
        if "error" in result:
            return JSONResponse(status_code=422, content=result)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"site_id": req.site_id, "error": str(e)})
