"""
Simple working image enhancer using basic upscaling.
Fallback when Real-ESRGAN has issues.
"""

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class SimpleEnhancer:
    """Simple image enhancer using traditional methods."""
    
    def __init__(self):
        logger.info("Initializing Simple Image Enhancer")
    
    def enhance(self, image: Image.Image, strength: float = 0.7) -> Image.Image:
        """
        Enhance image using traditional upscaling and sharpening.
        
        Args:
            image: Input PIL Image
            strength: Enhancement strength (0.0-1.0)
        """
        if strength <= 0.0:
            return image
            
        logger.info(f"Starting simple enhancement with strength: {strength}")
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # 1. Upscale using high-quality interpolation
        height, width = img_array.shape[:2]
        new_height = int(height * 2)  # 2x upscaling
        new_width = int(width * 2)
        
        # Use INTER_CUBIC for high quality upscaling
        upscaled = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 2. Apply denoising if strength is high enough
        if strength > 0.3:
            upscaled = cv2.bilateralFilter(upscaled, 5, 50, 50)
        
        # 3. Apply sharpening based on strength
        if strength > 0.5:
            # Create sharpening kernel
            kernel = np.array([[-0.1, -0.1, -0.1],
                              [-0.1,  1.8, -0.1],
                              [-0.1, -0.1, -0.1]])
            
            # Apply sharpening
            sharpened = cv2.filter2D(upscaled, -1, kernel)
            
            # Blend based on strength
            blend_factor = (strength - 0.5) * 2  # Scale to 0-1 range
            upscaled = cv2.addWeighted(upscaled, 1.0 - blend_factor, sharpened, blend_factor, 0)
        
        # 4. Enhance contrast slightly
        if strength > 0.7:
            # Convert to LAB for better contrast control
            lab = cv2.cvtColor(upscaled, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            
            # Apply mild contrast enhancement
            enhanced_l = cv2.convertScaleAbs(l_channel, alpha=1.0 + strength * 0.2, beta=0)
            lab[:, :, 0] = enhanced_l
            
            # Convert back to RGB
            upscaled = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Ensure values are in valid range
        upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)
        
        logger.info(f"Simple enhancement complete: {image.size} -> {upscaled.shape[1]}x{upscaled.shape[0]}")
        return Image.fromarray(upscaled)
    
    def enhance_with_compression(self, image: Image.Image, strength: float = 0.7, output_format: str = "AUTO") -> tuple[bytes, str]:
        """
        Enhance and compress image.
        
        Args:
            image: Input PIL Image
            strength: Enhancement strength
            output_format: Output format
            
        Returns:
            Tuple of (compressed_bytes, media_type)
        """
        # Enhance the image
        enhanced = self.enhance(image, strength)
        
        # Simple compression
        import io
        buffer = io.BytesIO()
        
        if output_format.upper() == "PNG":
            enhanced.save(buffer, format="PNG", optimize=True)
            media_type = "image/png"
        else:
            # Default to JPEG with good quality
            enhanced.save(buffer, format="JPEG", quality=85, optimize=True)
            media_type = "image/jpeg"
        
        buffer.seek(0)
        return buffer.getvalue(), media_type

# Global simple enhancer instance
_simple_enhancer = None

def get_simple_enhancer():
    """Get or create the global simple enhancer instance."""
    global _simple_enhancer
    if _simple_enhancer is None:
        _simple_enhancer = SimpleEnhancer()
    return _simple_enhancer