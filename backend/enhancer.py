"""
Real-ESRGAN image enhancement pipeline using native PyTorch implementation.
Restores full AI capability without flaky dependencies.
"""

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
import logging
import os
import requests
import io
from typing import Optional

from config import config
from architecture import RRDBNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageEnhancer:
    """
    Production-ready AI image enhancement using Real-ESRGAN.
    Uses direct PyTorch implementation to avoid dependency issues.
    """
    
    def __init__(self):
        """Initialize the enhancement pipeline."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        logger.info(f"Initializing AI ImageEnhancer on device: {self.device}")
        
    def _download_weights(self, url: str, path: str):
        """Download model weights if not present."""
        if os.path.exists(path):
            return
            
        logger.info(f"Downloading model weights to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        response = requests.get(url, stream=True)
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Download complete.")

    def _load_model(self):
        """Load the Real-ESRGAN model."""
        if self.model is not None:
            return

        # Model settings from config
        model_path = f"weights/{config.model_name}.pth"
        
        # Use correct download URL based on model
        if config.model_name == "RealESRGAN_x4plus":
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            model_scale = 4
        elif config.model_name == "RealESRGAN_x2plus":
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            model_scale = 2
        else:
            url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{config.model_name}.pth"
            model_scale = 4
        
        self._download_weights(url, model_path)
        
        # Initialize model architecture - use standard 3-channel architecture
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=model_scale)
        
        # Load weights
        loadnet = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)
            
        model.eval()
        model = model.to(self.device)
        self.model = model
        logger.info("Real-ESRGAN model loaded successfully")

    def _process_tile(self, img_tensor: torch.Tensor, tile_size: int = 400, tile_pad: int = 10) -> torch.Tensor:
        """Process image in tiles to save memory."""
        # Use config defaults if not specified
        if tile_size is None: tile_size = config.tile_size
        if tile_pad is None: tile_pad = config.tile_padding
        batch, channel, height, width = img_tensor.shape
        output_height = height * 4
        output_width = width * 4
        output_shape = (batch, channel, output_height, output_width)

        # Start with black image
        output = img_tensor.new_zeros(output_shape)
        tiles_x = np.ceil(width / tile_size).astype(int)
        tiles_y = np.ceil(height / tile_size).astype(int)

        for y in range(tiles_y):
            for x in range(tiles_x):
                # Calculate tile coordinates
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                
                # Input tile area with padding
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)
                
                # Input tile area with padding for processing
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # Crop input tile
                input_tile = img_tensor[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # Run inference
                with torch.no_grad():
                    try:
                        output_tile = self.model(input_tile)
                    except RuntimeError as error:
                        logger.error(f"Error processing tile: {error}")
                        # Fallback: simple interpolation for this tile
                        output_tile = F.interpolate(input_tile, scale_factor=4, mode='bicubic')

                # Calculate output coordinates
                output_start_x = input_start_x * 4
                output_end_x = input_end_x * 4
                output_start_y = input_start_y * 4
                output_end_y = input_end_y * 4

                # Calculate relative coordinates in the output tile
                output_start_x_tile = (input_start_x - input_start_x_pad) * 4
                output_end_x_tile = output_start_x_tile + (input_end_x - input_start_x) * 4
                output_start_y_tile = (input_start_y - input_start_y_pad) * 4
                output_end_y_tile = output_start_y_tile + (input_end_y - input_start_y) * 4
                
                # Place output tile
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                    output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

        return output

    def _apply_advanced_post_processing(self, image: Image.Image, strength: float) -> Image.Image:
        """
        Apply advanced post-processing for ultra-HD quality.
        
        Args:
            image: Enhanced image from AI model
            strength: Enhancement strength for controlling post-processing intensity
        """
        # Convert to numpy for advanced processing
        img_array = np.array(image).astype(np.float32)
        
        # 1. Advanced sharpening using unsharp mask
        if strength > 0.3:
            # Create Gaussian blur for unsharp mask
            blurred = cv2.GaussianBlur(img_array, (0, 0), 1.0)
            sharpened = cv2.addWeighted(img_array, 1.0 + strength * 0.5, blurred, -strength * 0.5, 0)
            img_array = np.clip(sharpened, 0, 255)
        
        # 2. Adaptive contrast enhancement
        if strength > 0.4:
            # Convert to LAB color space for better contrast control
            lab = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0 * strength, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel.astype(np.uint8)).astype(np.float32)
            
            # Blend original and enhanced L channel
            l_blended = cv2.addWeighted(l_channel, 1.0 - strength * 0.3, l_enhanced, strength * 0.3, 0)
            lab[:, :, 0] = np.clip(l_blended, 0, 255).astype(np.uint8)
            
            # Convert back to RGB
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
        
        # 3. Edge enhancement for ultra-sharp details
        if strength > 0.5:
            # Detect edges using Sobel operator
            gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            
            # Normalize edges
            edges = edges / edges.max() if edges.max() > 0 else edges
            
            # Apply edge enhancement to each channel
            for c in range(3):
                channel = img_array[:, :, c]
                enhanced_channel = channel + edges * strength * 10.0
                img_array[:, :, c] = np.clip(enhanced_channel, 0, 255)
        
        # 4. Color vibrance enhancement (subtle)
        if strength > 0.6:
            # Convert to HSV for color enhancement
            hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Enhance saturation selectively (avoid over-saturation)
            saturation = hsv[:, :, 1]
            enhanced_sat = saturation * (1.0 + strength * 0.2)
            hsv[:, :, 1] = np.clip(enhanced_sat, 0, 255)
            
            # Convert back to RGB
            img_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def _preprocess_for_quality(self, image: Image.Image) -> Image.Image:
        """
        Advanced preprocessing to optimize input for best AI enhancement results.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed image optimized for AI enhancement
        """
        img_array = np.array(image).astype(np.float32)
        
        # 1. Gentle noise reduction (preserves details)
        if img_array.shape[0] * img_array.shape[1] > 100000:  # Only for larger images
            denoised = cv2.bilateralFilter(img_array.astype(np.uint8), 5, 50, 50)
            img_array = denoised.astype(np.float32)
        
        # 2. Gamma correction for better dynamic range
        gamma = 1.1  # Slight gamma adjustment
        img_array = np.power(img_array / 255.0, gamma) * 255.0
        
        # 3. Ensure optimal bit depth
        img_array = np.clip(img_array, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    def enhance(self, image: Image.Image, strength: float = 0.7) -> Image.Image:
        """
        Ultra-HD image enhancement using advanced AI pipeline with post-processing.
        
        Args:
            image: Input PIL Image
            strength: Enhancement strength (0.0 = original, 1.0 = maximum ultra-HD enhancement)
        """
        if strength <= 0.0:
            return image
            
        logger.info(f"Starting Ultra-HD enhancement with strength: {strength}")
        self._load_model()
        
        # Advanced preprocessing for optimal AI input
        preprocessed_image = self._preprocess_for_quality(image)
        
        # Pre-process for AI model
        img_np = np.array(preprocessed_image)
        img_np = img_np.astype(np.float32) / 255.
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=2)
        
        # HWC to CHW and RGB to BGR (model expects BGR)
        img_tensor = torch.from_numpy(np.transpose(img_np[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # AI Enhancement with optimized tiling
        try:
            with torch.no_grad():
                # Use larger tiles for better quality when possible
                optimal_tile_size = min(config.tile_size, 640) if strength > 0.8 else config.tile_size
                output_tensor = self._process_tile(img_tensor, tile_size=optimal_tile_size, tile_pad=config.tile_padding)
        except Exception as e:
            logger.error(f"AI inference failed: {e}")
            return image

        # Post-process AI output with higher precision
        output = output_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # BGR to RGB, CHW to HWC
        
        # Use higher precision for better quality
        output = (output * 255.0).astype(np.float32)  # Keep as float32 longer
        
        # Create initial enhanced image
        enhanced_image = Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))
        
        # Apply advanced post-processing for ultra-HD quality
        ultra_hd_image = self._apply_advanced_post_processing(enhanced_image, strength)
        
        # Smart blending with original if strength < 1.0
        if strength < 1.0:
            # Upscale original to match enhanced size using highest quality resampling
            original_upscaled = image.resize(ultra_hd_image.size, Image.LANCZOS)
            
            # Advanced blending that preserves fine details
            ultra_hd_image = Image.blend(original_upscaled, ultra_hd_image, strength)
        
        # Final quality optimization
        final_image = self._final_quality_pass(ultra_hd_image, strength)
        
        # Log the enhancement results
        logger.info(f"Ultra-HD Enhancement complete: {image.size} -> {final_image.size} (4x upscale)")
        logger.info(f"Applied advanced post-processing with strength: {strength}")
        
        return final_image

    def _optimize_for_compression(self, image: Image.Image) -> Image.Image:
        """
        Optimize image for maximum compression efficiency without quality loss.
        
        Args:
            image: Enhanced ultra-HD image
            
        Returns:
            Compression-optimized image with same visual quality
        """
        img_array = np.array(image).astype(np.float32)
        
        # 1. Intelligent noise reduction for better compression
        # Remove compression artifacts and noise that hurt compression ratios
        denoised = cv2.bilateralFilter(img_array.astype(np.uint8), 3, 20, 20)
        img_array = denoised.astype(np.float32)
        
        # 2. Quantization optimization - reduce unnecessary color variations
        # This removes imperceptible color differences that hurt compression
        quantization_factor = 2  # Very subtle quantization
        img_array = np.round(img_array / quantization_factor) * quantization_factor
        
        # 3. Edge-preserving smoothing in flat areas
        # Smooth areas that don't need fine detail for better compression
        gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        # Create mask for non-edge areas
        smooth_mask = (edges_dilated == 0).astype(np.float32)
        smooth_mask = cv2.GaussianBlur(smooth_mask, (5, 5), 1.0)
        
        # Apply gentle smoothing only to non-edge areas
        smoothed = cv2.GaussianBlur(img_array.astype(np.uint8), (3, 3), 0.5).astype(np.float32)
        
        # Blend original and smoothed based on edge mask
        for c in range(3):
            img_array[:, :, c] = (img_array[:, :, c] * (1 - smooth_mask * 0.3) + 
                                 smoothed[:, :, c] * smooth_mask * 0.3)
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def _smart_format_selection(self, image: Image.Image) -> tuple[str, dict]:
        """
        Intelligently select the best format and settings for optimal compression.
        
        Args:
            image: Enhanced image
            
        Returns:
            Tuple of (format, save_params) for optimal compression
        """
        img_array = np.array(image)
        
        # Analyze image characteristics
        height, width = img_array.shape[:2]
        total_pixels = height * width
        
        # Calculate image complexity metrics
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels
        
        # Calculate color diversity
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        color_diversity = unique_colors / total_pixels
        
        # Smart format selection based on image characteristics
        if edge_density > 0.1 or color_diversity > 0.3:
            # High detail image - use optimized PNG
            return "PNG", {
                "optimize": True,
                "compress_level": 6,  # Good compression without quality loss
            }
        else:
            # Lower detail image - use high-quality JPEG
            return "JPEG", {
                "quality": 95,
                "optimize": True,
                "progressive": True,
                "subsampling": 0,  # No chroma subsampling for max quality
            }

    def _apply_advanced_compression(self, image: Image.Image) -> tuple[bytes, str]:
        """
        Apply advanced compression techniques for minimal file size with maximum quality.
        
        Args:
            image: Enhanced ultra-HD image
            
        Returns:
            Tuple of (compressed_bytes, media_type)
        """
        # Step 1: Optimize image for compression
        optimized_image = self._optimize_for_compression(image)
        
        # Step 2: Smart format selection
        format_name, save_params = self._smart_format_selection(optimized_image)
        
        # Step 3: Try multiple compression strategies and pick the best
        best_size = float('inf')
        best_data = None
        best_media_type = None
        
        compression_strategies = [
            # Strategy 1: Optimized PNG
            ("PNG", {
                "optimize": True,
                "compress_level": 6,
            }, "image/png"),
            
            # Strategy 2: High-quality JPEG
            ("JPEG", {
                "quality": 92,
                "optimize": True,
                "progressive": True,
                "subsampling": 0,
            }, "image/jpeg"),
            
            # Strategy 3: Ultra-compressed JPEG (if size is still too big)
            ("JPEG", {
                "quality": 88,
                "optimize": True,
                "progressive": True,
                "subsampling": 0,
            }, "image/jpeg"),
        ]
        
        for fmt, params, media_type in compression_strategies:
            buffer = io.BytesIO()
            try:
                optimized_image.save(buffer, format=fmt, **params)
                size = buffer.tell()
                
                if size < best_size:
                    best_size = size
                    buffer.seek(0)
                    best_data = buffer.getvalue()
                    best_media_type = media_type
                    
                # If we achieve good compression (< 400KB), use it
                if size < 400 * 1024:
                    break
                    
            except Exception as e:
                logger.warning(f"Compression strategy {fmt} failed: {e}")
                continue
        
    def _get_format_strategies(self, requested_format: str) -> list:
        """
        Get compression strategies based on requested format.
        
        Args:
            requested_format: Requested output format (AUTO, JPEG, PNG, WEBP)
            
        Returns:
            List of compression strategies for the format
        """
        if requested_format == "JPEG":
            return [
                ("JPEG", {"quality": 75, "optimize": True, "progressive": True, "subsampling": 1}, "image/jpeg"),
                ("JPEG", {"quality": 70, "optimize": True, "progressive": True, "subsampling": 1}, "image/jpeg"),
                ("JPEG", {"quality": 65, "optimize": True, "progressive": True, "subsampling": 1}, "image/jpeg"),
                ("JPEG", {"quality": 60, "optimize": True, "progressive": True, "subsampling": 1}, "image/jpeg"),
            ]
        elif requested_format == "PNG":
            return [
                ("PNG", {"optimize": True, "compress_level": 6}, "image/png"),
                ("PNG", {"optimize": True, "compress_level": 9}, "image/png"),
            ]
        elif requested_format == "WEBP":
            return [
                ("WEBP", {"quality": 80, "method": 6, "lossless": False}, "image/webp"),
                ("WEBP", {"quality": 75, "method": 6, "lossless": False}, "image/webp"),
                ("WEBP", {"quality": 70, "method": 6, "lossless": False}, "image/webp"),
                ("WEBP", {"quality": 65, "method": 6, "lossless": False}, "image/webp"),
            ]
        else:  # AUTO
            return [
                # Try WebP first (best compression)
                ("WEBP", {"quality": 80, "method": 6, "lossless": False}, "image/webp"),
                ("WEBP", {"quality": 75, "method": 6, "lossless": False}, "image/webp"),
                # Then JPEG
                ("JPEG", {"quality": 75, "optimize": True, "progressive": True, "subsampling": 1}, "image/jpeg"),
                ("JPEG", {"quality": 70, "optimize": True, "progressive": True, "subsampling": 1}, "image/jpeg"),
                ("JPEG", {"quality": 65, "optimize": True, "progressive": True, "subsampling": 1}, "image/jpeg"),
                # PNG as fallback
                ("PNG", {"optimize": True, "compress_level": 9}, "image/png"),
            ]

    def _ultra_aggressive_compression_with_format(self, image: Image.Image, requested_format: str = "AUTO") -> tuple[bytes, str]:
        """
        Ultra-aggressive compression with format selection targeting ~100KB.
        
        Args:
            image: Enhanced ultra-HD image
            requested_format: Requested output format (AUTO, JPEG, PNG, WEBP)
            
        Returns:
            Tuple of (compressed_bytes, media_type) targeting 100KB
        """
        # Step 1: Advanced pre-compression optimization (same as before)
        img_array = np.array(image).astype(np.float32)
        
        # More aggressive noise reduction for compression
        denoised = cv2.bilateralFilter(img_array.astype(np.uint8), 5, 40, 40)
        img_array = denoised.astype(np.float32)
        
        # Aggressive quantization while preserving edges
        gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
        
        # Create edge mask
        edge_mask = (edges_dilated > 0).astype(np.float32)
        smooth_mask = 1.0 - edge_mask
        
        # Apply different quantization levels
        edge_quantization = 1  # Preserve edges
        smooth_quantization = 4  # Aggressive quantization in smooth areas
        
        for c in range(3):
            channel = img_array[:, :, c]
            # Edge areas - minimal quantization
            edge_quantized = np.round(channel / edge_quantization) * edge_quantization
            # Smooth areas - aggressive quantization
            smooth_quantized = np.round(channel / smooth_quantization) * smooth_quantization
            # Combine based on mask
            img_array[:, :, c] = edge_quantized * edge_mask + smooth_quantized * smooth_mask
        
        optimized_image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        
        # Step 2: Get format-specific compression strategies
        target_size_kb = 100
        compression_strategies = self._get_format_strategies(requested_format)
        
        best_data = None
        best_size = float('inf')
        best_media_type = None
        
        for fmt, params, media_type in compression_strategies:
            buffer = io.BytesIO()
            try:
                optimized_image.save(buffer, format=fmt, **params)
                size = buffer.tell()
                
                # If we hit our target size, use this strategy
                if size <= target_size_kb * 1024:
                    buffer.seek(0)
                    logger.info(f"Target achieved: {size / 1024:.1f} KB with {fmt} format")
                    return buffer.getvalue(), media_type
                
                # Keep track of best option
                if size < best_size:
                    best_size = size
                    buffer.seek(0)
                    best_data = buffer.getvalue()
                    best_media_type = media_type
                    
            except Exception as e:
                logger.warning(f"Compression strategy {fmt} failed: {e}")
                continue
        
        # If we couldn't hit 100KB, try adaptive resizing
        if best_size > target_size_kb * 1024:
            logger.info(f"Applying adaptive resizing to reach {target_size_kb}KB target")
            return self._adaptive_resize_compression_with_format(optimized_image, target_size_kb, requested_format)
        
        logger.info(f"Ultra-aggressive compression achieved: {best_size / 1024:.1f} KB")
        return best_data, best_media_type

    def _adaptive_resize_compression_with_format(self, image: Image.Image, target_kb: int, requested_format: str) -> tuple[bytes, str]:
        """
        Adaptively resize image to hit target file size with format selection.
        
        Args:
            image: Optimized image
            target_kb: Target file size in KB
            requested_format: Requested output format
            
        Returns:
            Tuple of (compressed_bytes, media_type)
        """
        original_size = image.size
        resize_factors = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        
        # Get the best format strategy for this format
        strategies = self._get_format_strategies(requested_format)
        best_strategy = strategies[0] if strategies else ("JPEG", {"quality": 75, "optimize": True}, "image/jpeg")
        fmt, params, media_type = best_strategy
        
        for factor in resize_factors:
            new_width = int(original_size[0] * factor)
            new_height = int(original_size[1] * factor)
            
            # Resize with high-quality resampling
            resized = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Try compression
            buffer = io.BytesIO()
            try:
                resized.save(buffer, format=fmt, **params)
                size = buffer.tell()
                
                if size <= target_kb * 1024:
                    buffer.seek(0)
                    logger.info(f"Adaptive resize successful: {new_width}x{new_height}, {size / 1024:.1f} KB ({fmt})")
                    return buffer.getvalue(), media_type
            except Exception as e:
                logger.warning(f"Adaptive resize with {fmt} failed: {e}")
                continue
        
        # Fallback - use smallest size achieved
        final_factor = resize_factors[-1]
        new_width = int(original_size[0] * final_factor)
        new_height = int(original_size[1] * final_factor)
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        buffer = io.BytesIO()
        try:
            resized.save(buffer, format=fmt, **params)
            buffer.seek(0)
            logger.info(f"Fallback compression: {new_width}x{new_height}, {buffer.tell() / 1024:.1f} KB ({fmt})")
            return buffer.getvalue(), media_type
        except Exception as e:
            # Ultimate fallback to JPEG
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=70, optimize=True)
            buffer.seek(0)
            logger.info(f"Ultimate fallback to JPEG: {buffer.tell() / 1024:.1f} KB")
            return buffer.getvalue(), "image/jpeg"
        """
        Ultra-aggressive compression targeting ~100KB file size while maintaining visual quality.
        
        Args:
            image: Enhanced ultra-HD image
            
        Returns:
            Tuple of (compressed_bytes, media_type) targeting 100KB
        """
        # Step 1: Advanced pre-compression optimization
        img_array = np.array(image).astype(np.float32)
        
        # More aggressive noise reduction for compression
        denoised = cv2.bilateralFilter(img_array.astype(np.uint8), 5, 40, 40)
        img_array = denoised.astype(np.float32)
        
        # Aggressive quantization while preserving edges
        gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
        
        # Create edge mask
        edge_mask = (edges_dilated > 0).astype(np.float32)
        smooth_mask = 1.0 - edge_mask
        
        # Apply different quantization levels
        edge_quantization = 1  # Preserve edges
        smooth_quantization = 4  # Aggressive quantization in smooth areas
        
        for c in range(3):
            channel = img_array[:, :, c]
            # Edge areas - minimal quantization
            edge_quantized = np.round(channel / edge_quantization) * edge_quantization
            # Smooth areas - aggressive quantization
            smooth_quantized = np.round(channel / smooth_quantization) * smooth_quantization
            # Combine based on mask
            img_array[:, :, c] = edge_quantized * edge_mask + smooth_quantized * smooth_mask
        
        optimized_image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        
        # Step 2: Ultra-aggressive compression strategies targeting 100KB
        target_size_kb = 100
        compression_strategies = [
            # Strategy 1: Ultra-compressed JPEG with progressive
            ("JPEG", {
                "quality": 75,
                "optimize": True,
                "progressive": True,
                "subsampling": 1,  # Allow some chroma subsampling for size
            }),
            
            # Strategy 2: More aggressive JPEG
            ("JPEG", {
                "quality": 70,
                "optimize": True,
                "progressive": True,
                "subsampling": 1,
            }),
            
            # Strategy 3: Very aggressive JPEG
            ("JPEG", {
                "quality": 65,
                "optimize": True,
                "progressive": True,
                "subsampling": 1,
            }),
            
            # Strategy 4: Extreme compression
            ("JPEG", {
                "quality": 60,
                "optimize": True,
                "progressive": True,
                "subsampling": 1,
            }),
        ]
        
        best_data = None
        best_size = float('inf')
        
        for fmt, params in compression_strategies:
            buffer = io.BytesIO()
            try:
                optimized_image.save(buffer, format=fmt, **params)
                size = buffer.tell()
                
                # If we hit our target size, use this strategy
                if size <= target_size_kb * 1024:
                    buffer.seek(0)
                    logger.info(f"Target achieved: {size / 1024:.1f} KB with quality {params['quality']}")
                    return buffer.getvalue(), "image/jpeg"
                
                # Keep track of best option
                if size < best_size:
                    best_size = size
                    buffer.seek(0)
                    best_data = buffer.getvalue()
                    
            except Exception as e:
                logger.warning(f"Compression strategy failed: {e}")
                continue
        
        # If we couldn't hit 100KB, try adaptive resizing
        if best_size > target_size_kb * 1024:
            logger.info(f"Applying adaptive resizing to reach {target_size_kb}KB target")
            return self._adaptive_resize_compression(optimized_image, target_size_kb)
        
        logger.info(f"Ultra-aggressive compression achieved: {best_size / 1024:.1f} KB")
        return best_data, "image/jpeg"

    def _adaptive_resize_compression(self, image: Image.Image, target_kb: int) -> tuple[bytes, str]:
        """
        Adaptively resize image to hit target file size while maintaining quality.
        
        Args:
            image: Optimized image
            target_kb: Target file size in KB
            
        Returns:
            Tuple of (compressed_bytes, media_type)
        """
        original_size = image.size
        resize_factors = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        
        for factor in resize_factors:
            new_width = int(original_size[0] * factor)
            new_height = int(original_size[1] * factor)
            
            # Resize with high-quality resampling
            resized = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Try compression
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=75, optimize=True, progressive=True)
            size = buffer.tell()
            
            if size <= target_kb * 1024:
                buffer.seek(0)
                logger.info(f"Adaptive resize successful: {new_width}x{new_height}, {size / 1024:.1f} KB")
                return buffer.getvalue(), "image/jpeg"
        
        # Fallback - use smallest size achieved
        final_factor = resize_factors[-1]
        new_width = int(original_size[0] * final_factor)
        new_height = int(original_size[1] * final_factor)
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=70, optimize=True, progressive=True)
        buffer.seek(0)
        
        logger.info(f"Fallback compression: {new_width}x{new_height}, {buffer.tell() / 1024:.1f} KB")
        return buffer.getvalue(), "image/jpeg"
    def _final_quality_pass(self, image: Image.Image, strength: float) -> Image.Image:
        """
        Final quality optimization pass for ultra-HD output.
        
        Args:
            image: Enhanced image
            strength: Enhancement strength
            
        Returns:
            Final optimized ultra-HD image
        """
        if strength < 0.7:
            return image
            
        img_array = np.array(image).astype(np.float32)
        
        # Final sharpening pass for ultra-crisp details
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1,  1.8, -0.1],
                          [-0.1, -0.1, -0.1]])
        
        # Apply to each channel
        for c in range(3):
            channel = img_array[:, :, c]
            sharpened = cv2.filter2D(channel, -1, kernel)
            # Blend with original channel
            blend_factor = (strength - 0.7) * 0.5  # Scale for final pass
            img_array[:, :, c] = cv2.addWeighted(channel, 1.0 - blend_factor, sharpened, blend_factor, 0)
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def enhance_with_compression(self, image: Image.Image, strength: float = 0.7, output_format: str = "AUTO") -> tuple[bytes, str]:
        """
        Ultra-HD enhancement with format selection and ultra-aggressive compression targeting ~100KB.
        
        Args:
            image: Input PIL Image
            strength: Enhancement strength (0.0 = original, 1.0 = maximum ultra-HD enhancement)
            output_format: Output format (AUTO, JPEG, PNG, WEBP)
            
        Returns:
            Tuple of (compressed_image_bytes, media_type) targeting 100KB
        """
        # Perform ultra-HD enhancement
        enhanced_image = self.enhance(image, strength)
        
        # Apply ultra-aggressive compression with format selection
        compressed_data, media_type = self._ultra_aggressive_compression_with_format(enhanced_image, output_format)
        
        return compressed_data, media_type


# Global enhancer instance
_enhancer = None

def get_enhancer():
    """Get or create the global enhancer instance."""
    global _enhancer
    if _enhancer is None:
        _enhancer = ImageEnhancer()
    return _enhancer
