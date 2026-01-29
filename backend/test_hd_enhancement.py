#!/usr/bin/env python3
"""
Test script to verify Ultra-HD enhancement with format selection and 100KB target compression.
"""

import requests
import os
from PIL import Image
import io
import time

def test_format_selection():
    """Test the Ultra-HD enhancement with different output formats."""
    
    # Check if test image exists
    test_image_path = "test_input.png"
    if not os.path.exists(test_image_path):
        print(f"Test image {test_image_path} not found. Please add a test image.")
        return
    
    # Test different formats
    formats = ["AUTO", "WEBP", "JPEG", "PNG"]
    strength = 1.0
    url = "http://localhost:8000/enhance"
    
    original_image = Image.open(test_image_path)
    original_file_size = os.path.getsize(test_image_path) / 1024
    
    print(f"📸 Original image: {original_image.size}")
    print(f"📦 Original file size: {original_file_size:.1f} KB")
    print(f"🎯 Target: ~100KB with format selection")
    print("\n" + "="*80)
    
    results = []
    
    for fmt in formats:
        print(f"\n🚀 Testing {fmt} format (strength: {strength})")
        
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            data = {'strength': str(strength), 'format': fmt}
            
            start_time = time.time()
            response = requests.post(url, files=files, data=data)
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                # Determine file extension based on content type
                content_type = response.headers.get('content-type', 'image/jpeg')
                if 'webp' in content_type:
                    ext = 'webp'
                elif 'png' in content_type:
                    ext = 'png'
                else:
                    ext = 'jpg'
                
                # Save the enhanced image
                output_filename = f"test_output_{fmt.lower()}_{int(strength*100)}.{ext}"
                enhanced_image = Image.open(io.BytesIO(response.content))
                
                # Save to file
                with open(output_filename, 'wb') as f:
                    f.write(response.content)
                
                file_size_kb = len(response.content) / 1024
                compression_ratio = original_file_size / file_size_kb if file_size_kb > 0 else 1
                
                print(f"✅ {fmt} format successful!")
                print(f"   📐 Enhanced size: {enhanced_image.size}")
                print(f"   📈 Upscale factor: {enhanced_image.size[0] / original_image.size[0]:.1f}x")
                print(f"   📦 File size: {file_size_kb:.1f} KB")
                print(f"   🗜️  Compression ratio: {compression_ratio:.1f}x smaller")
                print(f"   ⚡ Processing time: {processing_time:.2f} seconds")
                print(f"   💾 Saved as: {output_filename}")
                print(f"   🎨 Content-Type: {content_type}")
                
                # Quality metrics
                pixel_count = enhanced_image.size[0] * enhanced_image.size[1]
                megapixels = pixel_count / 1_000_000
                kb_per_megapixel = file_size_kb / megapixels
                
                print(f"   🎯 Resolution: {megapixels:.1f} megapixels")
                print(f"   📊 Efficiency: {kb_per_megapixel:.1f} KB/MP")
                
                # Target achievement rating
                if file_size_kb <= 100:
                    rating = "🎯 TARGET HIT!"
                elif file_size_kb <= 120:
                    rating = "🏆 EXCELLENT"
                elif file_size_kb <= 150:
                    rating = "🥇 VERY GOOD"
                else:
                    rating = "🥈 GOOD"
                
                print(f"   {rating}")
                
                # Store results for comparison
                results.append({
                    'format': fmt,
                    'size_kb': file_size_kb,
                    'content_type': content_type,
                    'filename': output_filename,
                    'compression_ratio': compression_ratio
                })
                
            else:
                print(f"❌ {fmt} format failed: {response.status_code}")
                print(f"   Error: {response.text}")
    
    # Format comparison
    if results:
        print("\n" + "="*80)
        print("📊 FORMAT COMPARISON RESULTS:")
        print("-" * 80)
        
        # Sort by file size
        results.sort(key=lambda x: x['size_kb'])
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['format']:6} | {result['size_kb']:6.1f} KB | {result['compression_ratio']:4.1f}x | {result['content_type']}")
        
        print("-" * 80)
        best_format = results[0]
        print(f"🏆 BEST COMPRESSION: {best_format['format']} ({best_format['size_kb']:.1f} KB)")
        
        # Recommendations
        print("\n💡 FORMAT RECOMMENDATIONS:")
        print("• WebP: Best compression, modern browsers")
        print("• JPEG: Universal compatibility, good compression")
        print("• PNG: Lossless quality, larger files")
        print("• AUTO: Automatically selects best option")
    
    print("\n" + "="*80)
    print("🎯 Format Selection Test Complete!")
    print("\n🔧 Features Tested:")
    print("• Multiple output formats (AUTO, WebP, JPEG, PNG)")
    print("• Format-specific compression optimization")
    print("• 100KB target for all formats")
    print("• Content-Type header detection")
    print("• Compression efficiency comparison")

if __name__ == "__main__":
    test_format_selection()