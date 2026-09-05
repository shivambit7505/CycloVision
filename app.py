import io
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

# Load environment variables from .env if present
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k] = v

from data_simulator import SatelliteDataSimulator
from ai_engine import CycloVisionAIEngine
from similarity_engine import CycloneSimilarityEngine
from bulletin_generator import IMDBulletinDispatcher
from external_service import ExternalIntelligenceService

app = FastAPI(title="CycloVision AI API", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

simulator = SatelliteDataSimulator()
ai_engine = CycloVisionAIEngine()
sim_engine = CycloneSimilarityEngine()
ext_service = ExternalIntelligenceService()

app.mount("/static", StaticFiles(directory="static"), name="static")

class CycloneAnalysisRequest(BaseModel):
    basin: str = "Bay of Bengal"
    simulated_intensity: float = 4.5
    center_lat: float = 16.2
    center_lon: float = 84.6

@app.get("/")
def root():
    return {"status": "CycloVision AI is running. Access dashboard at /static/index.html"}

@app.post("/api/analyze")
def analyze_cyclone(req: CycloneAnalysisRequest):
    try:
        env = simulator.get_environmental_parameters(basin=req.basin)
        current_wind = float(req.simulated_intensity * 17.5 + 10.0)
        current_press = float(1010.0 - (req.simulated_intensity * 9.2))
        
        stage_name, stage_code, alert_level = ai_engine.classify_imd_stage(current_wind)
        ri_data = ai_engine.calculate_rapid_intensification(
            sst=env["sst_celsius"],
            ohc=env["ocean_heat_content_kj_cm2"],
            vws=env["vertical_wind_shear_knots"],
            current_wind=current_wind
        )
        traj_data = ai_engine.predict_trajectory_and_landfall(
            curr_lat=req.center_lat,
            curr_lon=req.center_lon,
            env_params=env,
            current_wind=current_wind
        )
        analogs = sim_engine.find_top_analogs(
            current_wind=current_wind,
            current_pressure=current_press,
            current_basin=req.basin
        )

        # Generate Gemini / LLM Meteorological Insights
        gemini_explanation = ext_service.generate_gemini_cyclone_explanation({
            "basin": req.basin,
            "dvorak_t_number": round(req.simulated_intensity, 1),
            "estimated_wind_knots": round(current_wind, 1),
            "central_pressure_hpa": round(current_press, 1),
            "env": env,
            "ri_probability": ri_data["ri_probability"]
        })

        mosdac_status = ext_service.verify_mosdac_feed_status()
        
        return {
            "storm_metadata": {
                "system_name": "BOB-02 / DEEP CYCLONIC SYSTEM",
                "basin": req.basin,
                "current_center": {"lat": req.center_lat, "lon": req.center_lon},
                "dvorak_t_number": round(req.simulated_intensity, 1),
                "estimated_wind_knots": round(current_wind, 1),
                "estimated_wind_kmph": round(current_wind * 1.852, 1),
                "central_pressure_hpa": round(current_press, 1),
                "pressure_deficit_hpa": round(1010.0 - current_press, 1),
                "imd_stage": stage_name,
                "imd_stage_code": stage_code,
                "alert_level": alert_level
            },
            "rapid_intensification": ri_data,
            "trajectory_forecast": traj_data["forecast_track"],
            "landfall_risk": traj_data["landfall_estimate"],
            "top_historical_analogs": analogs,
            "gemini_ai_reasoning": gemini_explanation,
            "mosdac_integration": mosdac_status,
            "mapbox_token": ext_service.mapbox_token
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-satellite-image")
async def upload_satellite_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
        detection_res = ai_engine.process_uploaded_image(pil_img)
        
        detected_t = detection_res["estimated_t_number"]
        current_wind = float(detected_t * 17.5 + 10.0)
        stage_name, stage_code, alert_level = ai_engine.classify_imd_stage(current_wind)
        
        env = simulator.get_environmental_parameters(basin="Bay of Bengal")
        ri_data = ai_engine.calculate_rapid_intensification(
            sst=env["sst_celsius"],
            ohc=env["ocean_heat_content_kj_cm2"],
            vws=env["vertical_wind_shear_knots"],
            current_wind=current_wind
        )
        traj_data = ai_engine.predict_trajectory_and_landfall(
            curr_lat=17.2,
            curr_lon=85.4,
            env_params=env,
            current_wind=current_wind
        )

        return {
            "detection": detection_res,
            "filename": file.filename,
            "dvorak_t_number": detected_t,
            "estimated_wind_knots": round(current_wind, 1),
            "estimated_wind_kmph": round(current_wind * 1.852, 1),
            "imd_stage": stage_name,
            "alert_level": alert_level,
            "rapid_intensification": ri_data,
            "landfall_risk": traj_data["landfall_estimate"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/api/bulletin")
def get_bulletin(req: CycloneAnalysisRequest):
    data = analyze_cyclone(req)
    bulletin = IMDBulletinDispatcher.generate_bulletin({
        "bulletin_no": "CV-01",
        "current_lat": data["storm_metadata"]["current_center"]["lat"],
        "current_lon": data["storm_metadata"]["current_center"]["lon"],
        "basin": data["storm_metadata"]["basin"],
        "dvorak_t_number": data["storm_metadata"]["dvorak_t_number"],
        "wind_knots": data["storm_metadata"]["estimated_wind_knots"],
        "pressure_hpa": data["storm_metadata"]["central_pressure_hpa"],
        "imd_category": data["storm_metadata"]["imd_stage"],
        "alert_level": data["storm_metadata"]["alert_level"],
        "ri_risk": data["rapid_intensification"],
        "landfall": data["landfall_risk"]
    })
    return {"bulletin_text": bulletin}
