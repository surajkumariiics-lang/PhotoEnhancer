
import os
import sys
import logging
import numpy as np
from PIL import Image
import torch
import cv2

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

from enhancer import ImageEnhancer
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reproduce")

def create_grid_image(size=(200, 200), grid_size=50):
    img = Image.new('RGB', size, color='white')
    pixels = img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            if (x // grid_size) % 2 == (y // grid_size) % 2:
                pixels[x, y] = (0, 0, 0)
    return img

def test_enhancement():
    print("Testing ImageEnhancer...")
    enhancer = ImageEnhancer()
    
    # Create a test image with a grid pattern to check for tile artifacts
    input_size = (800, 800) # Large enough to force tiling (tile_size is 400 or 512)
    img = create_grid_image(input_size, grid_size=50)
    img.save("test_input.png")
    
    print(f"Input image saved: test_input.png {img.size}")
    
    # Run enhancement
    try:
        # Force a specific strength to see full effect
        output = enhancer.enhance(img, strength=1.0)
        output.save("test_output.png")
        print(f"Output image saved: test_output.png {output.size}")
        
        # Check if output is actually upscaled
        if output.size[0] == img.size[0] * 4:
            print("SUCCESS: Output is 4x input size.")
        else:
            print(f"WARNING: Output size {output.size} is not 4x input size {img.size}.")
            
    except Exception as e:
        print(f"ERROR during enhancement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhancement()
