from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import json
import shutil
from pathlib import Path
import uvicorn
import traceback
from dotenv import load_dotenv
load_dotenv()
from brand_fluidity_analyzer import BrandFluidityAnalyzer
from llm_analyzer import LLMAnalyzer

app = FastAPI(title="Brand Fluidity Analyzer API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
(UPLOAD_DIR / "texts").mkdir(exist_ok=True)
(UPLOAD_DIR / "images").mkdir(exist_ok=True)

analyzer = BrandFluidityAnalyzer()
llm_analyzer = LLMAnalyzer()

@app.post("/api/upload")
async def upload_files(
    brand_name: str = Form(...),
    texts: List[UploadFile] = File([]),
    images: List[UploadFile] = File([]),
    years: Optional[str] = Form(None)
):
    """
    Upload brand materials (texts and images) for analysis.
    years should be a JSON string mapping filenames to years.
    """
    try:
        # Parse year mapping
        year_mapping = json.loads(years) if years else {}
        
        # Create brand directory
        brand_dir = UPLOAD_DIR / brand_name.replace(" ", "_")
        brand_dir.mkdir(exist_ok=True)
        text_dir = brand_dir / "texts"
        image_dir = brand_dir / "images"
        text_dir.mkdir(exist_ok=True)
        image_dir.mkdir(exist_ok=True)
        
        # Save text files
        text_files = []
        for text_file in texts:
            file_path = text_dir / text_file.filename
            with open(file_path, "wb") as f:
                shutil.copyfileobj(text_file.file, f)
            text_files.append({
                "filename": text_file.filename,
                "path": str(file_path),
                "year": year_mapping.get(text_file.filename, None)
            })
        
        # Save image files
        image_files = []
        for image_file in images:
            file_path = image_dir / image_file.filename
            with open(file_path, "wb") as f:
                shutil.copyfileobj(image_file.file, f)
            image_files.append({
                "filename": image_file.filename,
                "path": str(file_path),
                "year": year_mapping.get(image_file.filename, None)
            })
        
        return JSONResponse({
            "status": "success",
            "brand_name": brand_name,
            "texts_uploaded": len(text_files),
            "images_uploaded": len(image_files),
            "text_files": text_files,
            "image_files": image_files
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_brand(
    brand_name: str = Form(...),
    include_llm_insights: bool = Form(True)
):
    """
    Analyze brand fluidity metrics for uploaded materials.
    """
    try:
        print("="*70)
        print(f"Starting analysis for: {brand_name}")
        print(f"Include LLM insights: {include_llm_insights}")
        print("="*70)
        
        brand_dir = UPLOAD_DIR / brand_name.replace(" ", "_")
        if not brand_dir.exists():
            raise HTTPException(status_code=404, detail="Brand not found")
        
        text_dir = brand_dir / "texts"
        image_dir = brand_dir / "images"
        
        # Count files
        text_files = list(text_dir.glob("*")) if text_dir.exists() else []
        image_files = list(image_dir.glob("*")) if image_dir.exists() else []
        
        print(f"Found {len(text_files)} text files in {text_dir}")
        print(f"Found {len(image_files)} image files in {image_dir}")
        
        # Run analysis
        print("\nStarting Brand Fluidity Analysis...")
        results = analyzer.analyze_brand(
            text_dir=str(text_dir),
            image_dir=str(image_dir),
            brand_name=brand_name
        )
        print("Brand Fluidity Analysis completed successfully!")
        
        # Add LLM insights if requested
        if include_llm_insights:
            print("\nGenerating LLM insights...")
            llm_insights = await llm_analyzer.generate_insights(results)
            results["llm_insights"] = llm_insights
            print("LLM insights generated successfully!")
        
        print("="*70)
        print("Analysis completed successfully!")
        print("="*70)
        
        return JSONResponse(results)
    
    except Exception as e:
        print("="*70)
        print("ERROR OCCURRED DURING ANALYSIS:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("="*70)
        print("FULL TRACEBACK:")
        traceback.print_exc()
        print("="*70)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/brands")
async def list_brands():
    """List all analyzed brands."""
    brands = []
    for brand_dir in UPLOAD_DIR.iterdir():
        if brand_dir.is_dir():
            text_count = len(list((brand_dir / "texts").glob("*"))) if (brand_dir / "texts").exists() else 0
            image_count = len(list((brand_dir / "images").glob("*"))) if (brand_dir / "images").exists() else 0
            brands.append({
                "name": brand_dir.name.replace("_", " "),
                "text_count": text_count,
                "image_count": image_count
            })
    return {"brands": brands}

@app.delete("/api/brands/{brand_name}")
async def delete_brand(brand_name: str):
    """Delete a brand and its uploaded files."""
    try:
        brand_dir = UPLOAD_DIR / brand_name.replace(" ", "_")
        if brand_dir.exists():
            shutil.rmtree(brand_dir)
            return {"status": "success", "message": f"Brand '{brand_name}' deleted"}
        else:
            raise HTTPException(status_code=404, detail="Brand not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Brand Fluidity Analyzer API", "version": "1.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
