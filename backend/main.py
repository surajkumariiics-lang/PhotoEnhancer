"""
FastAPI backend for AI image quality enhancement.
Provides REST API endpoint for conservative image enhancement.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging
from typing import Optional

from enhancer import get_enhancer
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Image Quality Enhancer",
    description="Conservative image enhancement API using Real-ESRGAN",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "device": config.device,
        "model": config.model_name,
        "version": "1.0.0"
    }


@app.post("/enhance")
async def enhance_image(
    file: UploadFile = File(..., description="Image file to enhance"),
    strength: Optional[float] = Form(0.7, description="Enhancement strength (0.0-1.0)"),
    format: Optional[str] = Form("AUTO", description="Output format: AUTO, JPEG, PNG, WEBP")
):
    """
    Enhance image quality using conservative AI enhancement.
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        strength: Enhancement strength from 0.0 (minimal) to 1.0 (full)
        format: Output format (AUTO, JPEG, PNG, WEBP)
    
    Returns:
        Enhanced image in specified format
    
    Raises:
        HTTPException: If image processing fails
    """
    logger.info(f"Received enhancement request: {file.filename}, strength={strength}, format={format}")
    
    # Validate strength parameter
    if not 0.0 <= strength <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Strength must be between 0.0 and 1.0"
        )
    
    # Validate format parameter
    valid_formats = ["AUTO", "JPEG", "PNG", "WEBP"]
    if format.upper() not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Format must be one of: {', '.join(valid_formats)}"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != "RGB":
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert("RGB")
        
        logger.info(f"Input image size: {image.size}")
        
        # Enhance image with format selection
        enhancer = get_enhancer()
        compressed_data, media_type = enhancer.enhance_with_compression(image, strength=strength, output_format=format.upper())
        
        # Create response buffer
        output_buffer = io.BytesIO(compressed_data)
        
        # Log compression results
        original_size = len(image_data) / 1024
        compressed_size = len(compressed_data) / 1024
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
        
        logger.info(f"Compression: {original_size:.1f}KB -> {compressed_size:.1f}KB (ratio: {compression_ratio:.1f}x)")
        logger.info("Ultra-HD enhancement with compression successful")
        
        # Return compressed enhanced image
        return StreamingResponse(
            output_buffer,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=enhanced_{file.filename}",
                "X-Original-Size": str(int(original_size)),
                "X-Compressed-Size": str(int(compressed_size)),
                "X-Compression-Ratio": f"{compression_ratio:.1f}x"
            }
        )
        
    except Exception as e:
        logger.error(f"Enhancement failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Image enhancement failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Detailed health check with system information."""
    import torch
    
    return {
        "status": "healthy",
        "device": config.device,
        "cuda_available": torch.cuda.is_available(),
        "model": config.model_name,
        "model_scale": config.model_scale,
        "face_enhance": config.face_enhance
    }


if __name__ == "__main__":
    import uvicorn
    import os
    
    logger.info("Starting AI Image Quality Enhancer API")
    logger.info(f"Device: {config.device}")
    logger.info(f"Model: {config.model_name}")
    
    # Use PORT environment variable for Render deployment
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
