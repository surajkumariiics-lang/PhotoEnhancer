
import os
import sys
import logging
import argparse
from PIL import Image

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

from enhancer import ImageEnhancer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_enhancement")

def main():
    parser = argparse.ArgumentParser(description="Enhance an image using Real-ESRGAN.")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("--output", help="Path to save output image (optional, defaults to enhanced_<input_name>)")
    parser.add_argument("--strength", type=float, default=0.7, help="Enhancement strength (0.0 - 1.0)")
    
    args = parser.parse_args()
    
    input_path = args.input_image
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return
        
    if args.output:
        output_path = args.output
    else:
        filename, ext = os.path.splitext(input_path)
        output_path = f"{filename}_enhanced{ext}"
        
    print(f"Loading '{input_path}'...")
    try:
        image = Image.open(input_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    print("Initializing enhancer...")
    enhancer = ImageEnhancer()
    
    print(f"Enhancing image with strength {args.strength}...")
    try:
        enhanced_image = enhancer.enhance(image, strength=args.strength)
        
        print(f"Saving result to '{output_path}'...")
        enhanced_image.save(output_path)
        print("Done! Image enhanced successfully.")
        
    except Exception as e:
        print(f"Error during enhancement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
