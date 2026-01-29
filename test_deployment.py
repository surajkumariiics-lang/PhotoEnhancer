#!/usr/bin/env python3
"""
Test script to verify Render deployment is working.
Run this after deployment to check if your API is responding.
"""

import requests
import sys

def test_api(base_url):
    """Test the deployed API endpoints."""
    print(f"🧪 Testing API at: {base_url}")
    
    try:
        # Test health endpoint
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("   ✅ Health check passed")
            data = response.json()
            print(f"   📊 Status: {data.get('status')}")
            print(f"   🖥️  Device: {data.get('device')}")
            print(f"   🤖 Model: {data.get('model')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
            
        # Test root endpoint
        print("2. Testing root endpoint...")
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("   ✅ Root endpoint working")
            data = response.json()
            print(f"   📊 Status: {data.get('status')}")
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
            return False
            
        print("\n🎉 All tests passed! Your API is ready.")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    else:
        api_url = input("Enter your Render API URL (e.g., https://your-app.onrender.com): ").strip()
    
    if not api_url.startswith('http'):
        api_url = f"https://{api_url}"
    
    success = test_api(api_url)
    sys.exit(0 if success else 1)