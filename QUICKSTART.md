# Quick Start Guide

## Installation

### Backend Setup

1. Open a terminal in the `backend` folder
2. Run the setup script:
   ```bash
   setup.bat
   ```
   This will:
   - Create a Python virtual environment
   - Install all required dependencies
   - Create the weights directory

### Frontend Setup

The frontend dependencies are already installed. If needed, you can reinstall them:
```bash
cd frontend
npm install
```

## Running the Application

### Option 1: Using Helper Scripts (Recommended)

1. **Start Backend** (Terminal 1):
   ```bash
   cd backend
   start_backend.bat
   ```
   Wait for "Uvicorn running on http://0.0.0.0:8000"

2. **Start Frontend** (Terminal 2):
   ```bash
   start_frontend.bat
   ```
   Browser will open automatically at http://localhost:5173

### Option 2: Manual Start

**Backend**:
```bash
cd backend
venv\Scripts\activate
python main.py
```

**Frontend**:
```bash
cd frontend
npm run dev
```

## First Run Notes

⚠️ **Model Download**: On first run, Real-ESRGAN will automatically download the model weights (~67MB). This is a one-time download.

⚠️ **GPU vs CPU**: The application will automatically detect if you have a CUDA-capable GPU. If not, it will use CPU mode (slower but functional).

## Usage

1. **Upload an Image**
   - Drag and drop an image onto the upload zone
   - Or click to browse and select

2. **Adjust Enhancement Strength**
   - Use the slider (0-100%)
   - 30% = Subtle
   - 70% = Moderate (default)
   - 100% = Strong

3. **Enhance**
   - Click "Enhance Image"
   - Wait for processing (2-30 seconds depending on GPU/CPU)

4. **Download**
   - Compare before/after
   - Click "Download" to save

## Troubleshooting

### Backend won't start
- Make sure you ran `setup.bat` first
- Check that Python 3.8+ is installed
- Try deleting `venv` folder and running `setup.bat` again

### Frontend won't start
- Run `npm install` in the frontend folder
- Make sure Node.js is installed

### Enhancement is slow
- This is normal on CPU (10-30 seconds)
- GPU processing is much faster (2-5 seconds)
- Consider using a smaller enhancement strength for faster processing

### Model download fails
- Manual download: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
- Place in `backend/weights/` folder

## Project Structure

```
PhotoEnhancer/
├── backend/           # Python FastAPI backend
├── frontend/          # React frontend
├── README.md          # Full documentation
└── start_*.bat        # Helper scripts
```

For detailed documentation, see [README.md](file:///c:/Users/pop/Desktop/PhotoEnhancer/README.md)
