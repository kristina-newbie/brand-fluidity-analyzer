from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
import json
import shutil
from pathlib import Path
from collections import defaultdict
import uvicorn
import traceback
import numpy as np
from dotenv import load_dotenv

from brand_fluidity_analyzer_v2 import BrandLiquidityAnalyzer
from utils import save_files, extract_year_from_filename

load_dotenv()

from llm_analyzer import LLMAnalyzer

app = FastAPI(title="Brand Liquidity Analyzer API", version="5.0")

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

analyzer = BrandLiquidityAnalyzer()
llm_analyzer = LLMAnalyzer()

def load_texts_by_period(text_dir: Path) -> Dict[str, List[str]]:
    """
    Load text files and organize by period (year).
    
    Args:
        text_dir: Directory containing text files
        
    Returns:
        Dict mapping year strings to lists of text content
    """
    texts_by_period = defaultdict(list)
    
    if not text_dir.exists():
        return dict(texts_by_period)
    
    for text_file in text_dir.glob("*"):
        if text_file.is_file() and text_file.suffix.lower() in ['.txt', '.md', '.doc', '.docx']:
            try:
                # Try to read text content
                content = text_file.read_text(encoding='utf-8', errors='ignore')
                
                # Extract year from filename
                year = extract_year_from_filename(text_file.name)
                
                if year:
                    texts_by_period[year].append(content)
                else:
                    # Default to "unknown" period if no year found
                    texts_by_period["unknown"].append(content)
                    
            except Exception as e:
                print(f"Error reading {text_file}: {e}")
                continue
    
    # Remove "unknown" period if we have dated content
    if "unknown" in texts_by_period and len(texts_by_period) > 1:
        del texts_by_period["unknown"]
    
    return dict(texts_by_period)


def load_images_by_period(image_dir: Path) -> Dict[str, List[str]]:
    """
    Load image paths and organize by period (year).
    
    Args:
        image_dir: Directory containing image files
        
    Returns:
        Dict mapping year strings to lists of image paths
    """
    images_by_period = defaultdict(list)
    
    if not image_dir.exists():
        return dict(images_by_period)
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
    
    for image_file in image_dir.glob("*"):
        if image_file.is_file() and image_file.suffix.lower() in valid_extensions:
            # Extract year from filename
            year = extract_year_from_filename(image_file.name)
            
            if year:
                images_by_period[year].append(str(image_file))
            else:
                # Default to "unknown" period if no year found
                images_by_period["unknown"].append(str(image_file))
    
    # Remove "unknown" period if we have dated content
    if "unknown" in images_by_period and len(images_by_period) > 1:
        del images_by_period["unknown"]
    
    return dict(images_by_period)


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {
            (int(key) if isinstance(key, np.integer) else 
             float(key) if isinstance(key, np.floating) else 
             str(key) if not isinstance(key, (str, int, float, bool, type(None))) else key): 
            convert_numpy_types(value) 
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


@app.post("/api/upload")
async def upload_files(
    brand_name: str = Form(...),
    texts: List[UploadFile] = File([]),
    images: List[UploadFile] = File([]),
    years: Optional[str] = Form(None)
):
    """
    Upload brand materials (texts and images) for analysis.
    
    Files should include year in filename (e.g., 'brand_copy_2019.txt', 'campaign_2021.jpg')
    for automatic period detection. Alternatively, provide years as JSON mapping.
    """
    try:
        # Parse year mapping if provided
        year_mapping = json.loads(years) if years else {}
        
        # Create brand directory
        brand_dir = UPLOAD_DIR / brand_name.replace(" ", "_")
        brand_dir.mkdir(exist_ok=True)
        text_dir = brand_dir / "texts"
        image_dir = brand_dir / "images"
        text_dir.mkdir(exist_ok=True)
        image_dir.mkdir(exist_ok=True)

        text_files = save_files(texts, text_dir, year_mapping)
        image_files = save_files(images, image_dir, year_mapping)
        
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
    Analyze brand liquidity metrics for uploaded materials.
    
    Returns comprehensive liquidity analysis including:
    - Core metrics: TLI, VLI, CMCS, BLI
    - Second-order metrics: Drift volatility, coherence trajectory, drift ratio, etc.
    - Optional LLM-generated strategic insights
    """
    try:
        print("="*70)
        print(f"Starting analysis for: {brand_name}")
        print(f"Include LLM insights: {include_llm_insights}")
        print("="*70)
        
        brand_dir = UPLOAD_DIR / brand_name.replace(" ", "_")
        if not brand_dir.exists():
            raise HTTPException(status_code=404, detail="Brand not found. Please upload files first.")
        
        text_dir = brand_dir / "texts"
        image_dir = brand_dir / "images"
        
        # Load and organize data by period
        print("\nLoading and organizing data by period...")
        texts_by_period = load_texts_by_period(text_dir)
        images_by_period = load_images_by_period(image_dir)
        
        # Report what was found
        text_count = sum(len(texts) for texts in texts_by_period.values())
        image_count = sum(len(imgs) for imgs in images_by_period.values())
        
        print(f"Found {text_count} texts across {len(texts_by_period)} periods: {sorted(texts_by_period.keys())}")
        print(f"Found {image_count} images across {len(images_by_period)} periods: {sorted(images_by_period.keys())}")
        
        if text_count == 0 and image_count == 0:
            raise HTTPException(
                status_code=400, 
                detail="No valid text or image files found. Ensure files have year in filename (e.g., 'content_2019.txt')."
            )
        
        # Run analysis with v5 analyzer
        print("\nStarting Brand Liquidity Analysis v5.0...")
        results = analyzer.analyze_brand(
            texts_by_period=texts_by_period,
            images_by_period=images_by_period,
            brand_name=brand_name
        )
        print("Brand Liquidity Analysis completed successfully!")
        
        # Add LLM insights if requested
        if include_llm_insights:
            print("\nGenerating LLM insights...")
            llm_insights = await llm_analyzer.generate_insights(results)
            results["llm_insights"] = llm_insights
            print("LLM insights generated successfully!")
        
        print("="*70)
        print("Analysis completed successfully!")
        print("="*70)
        
        # Convert numpy types for JSON serialization
        print("Converting numpy types to Python native types...")
        results = convert_numpy_types(results)
        print("Conversion completed!")
        
        return JSONResponse(results)
    
    except HTTPException:
        raise
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
    """List all analyzed brands with file counts and detected periods."""
    brands = []
    for brand_dir in UPLOAD_DIR.iterdir():
        if brand_dir.is_dir():
            text_dir = brand_dir / "texts"
            image_dir = brand_dir / "images"
            
            texts_by_period = load_texts_by_period(text_dir)
            images_by_period = load_images_by_period(image_dir)
            
            text_count = sum(len(texts) for texts in texts_by_period.values())
            image_count = sum(len(imgs) for imgs in images_by_period.values())
            
            all_periods = sorted(set(texts_by_period.keys()) | set(images_by_period.keys()))
            
            brands.append({
                "name": brand_dir.name.replace("_", " "),
                "text_count": text_count,
                "image_count": image_count,
                "periods": all_periods,
                "n_periods": len(all_periods)
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
    return {
        "message": "Brand Liquidity Analyzer API",
        "version": "5.0",
        "features": [
            "Core metrics: TLI, VLI, CMCS, BLI",
            "Second-order metrics: Drift volatility, coherence trajectory, drift ratio",
            "Directional consistency and topic-semantic decoupling",
            "LLM-powered strategic insights"
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
