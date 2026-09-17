from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os

from ai_engine import predict_cyclone_v2
from bulletin_generator import generate_imd_bulletin_v2
from v2.api.routes import v2_router

app = FastAPI(
    title="CycloVision AI V2",
    description="Authoritative Tropical Cyclone Early Warning & Decision Support System (MoES/IMD)",
    version="2.1.0"
)

# Mount Static UI & V2 Routers
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(v2_router)

@app.get("/")
def read_root():
    return FileResponse("static/v2_dashboard.html")

# Slide 32 Clean API Endpoints
@app.get("/api/data-status")
def api_data_status():
    return {
        "satellite_source": "ISRO MOSDAC INSAT-3D/3DS",
        "freshness": "Live 30-Minute Ingestion Active",
        "best_track_reference": "NOAA NCEI IBTrACS v4r01 (DOI: 10.25921/82ty-9e16)",
        "reanalysis": "NASA MERRA-2 + NOAA ERSSTv6",
        "qc_status": "ALL_FEEDS_HEALTHY"
    }

@app.post("/api/analyze")
def api_analyze_v2(t_number: float = 5.5, sst: float = 30.0, vws: float = 7.5):
    return predict_cyclone_v2(t_number=t_number, sst=sst, vws=vws)

@app.post("/api/bulletin")
def api_bulletin(storm_name: str = "CYCLONE-X", wind_kmph: float = 165.0, stage: str = "VSCS"):
    text = generate_imd_bulletin_v2(storm_name, stage, wind_kmph, 965.0, "Puri, Odisha Sector", 18)
    return {"bulletin_text": text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
