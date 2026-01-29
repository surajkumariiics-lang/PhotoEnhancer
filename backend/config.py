"""
Configuration for the image enhancement pipeline.
Conservative settings to preserve realism and avoid hallucination.
"""

import torch
from dataclasses import dataclass


@dataclass
class EnhancementConfig:
    """Configuration for conservative image enhancement."""
    
    # Model settings - Optimized for Ultra-HD quality
    model_name: str = "RealESRGAN_x4plus"
    model_scale: int = 4  # Full 4x upscaling for Ultra-HD output
    tile_size: int = 640  # Larger tiles for maximum quality (was 512)
    tile_padding: int = 40 # Increased padding for seamless tile blending
    
    # Denoising settings - mild to preserve texture
    denoise_strength: float = 10.0  # OpenCV NLMeans h parameter (lower = less aggressive)
    denoise_template_window: int = 7
    denoise_search_window: int = 21
    
    # Sharpening settings - subtle enhancement only
    sharpen_radius: float = 1.0  # Small radius for subtle sharpening
    sharpen_amount: float = 0.5  # Mild sharpening strength
    sharpen_threshold: int = 0
    
    # Face enhancement - DISABLED to preserve original appearance
    face_enhance: bool = False
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = False  # Use FP32 for better quality on CPU
    
    # Output settings - Optimized for Ultra-HD quality with smart compression
    output_format: str = "AUTO"  # Automatically select best format
    output_quality: int = 92  # High quality with good compression balance
    
    # Advanced processing settings
    enable_advanced_postprocessing: bool = True
    enable_edge_enhancement: bool = True
    enable_color_enhancement: bool = True
    enable_smart_compression: bool = True
    
    # Compression settings - Ultra-aggressive targeting 100KB
    max_file_size_kb: int = 100  # Target 100KB file size
    compression_quality_threshold: int = 60  # Minimum quality for ultra compression
    enable_adaptive_resizing: bool = True  # Allow resizing to hit target size
    
    def get_enhancement_params(self, strength: float = 0.7) -> dict:
        """
        Get enhancement parameters scaled by strength factor.
        
        Args:
            strength: Enhancement strength from 0.0 (minimal) to 1.0 (full)
        
        Returns:
            Dictionary of scaled parameters
        """
        strength = max(0.0, min(1.0, strength))  # Clamp to valid range
        
        return {
            "denoise_strength": self.denoise_strength * strength,
            "sharpen_amount": self.sharpen_amount * strength,
            "upscale_factor": 1.0 + (self.model_scale - 1.0) * strength,  # 1x to 2x based on strength
        }


# Global configuration instance
config = EnhancementConfig()
