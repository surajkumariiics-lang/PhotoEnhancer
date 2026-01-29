# Render Deployment Guide

## 🚀 Quick Deploy to Render

### Step 1: Push to GitHub
1. Create a new GitHub repository
2. Push your code:
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### Step 2: Deploy Backend (API)
1. Go to [render.com](https://render.com) and sign up/login
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `ai-image-enhancer-api`
   - **Environment**: `Python 3`
   - **Build Command**: `cd backend && pip install -r requirements.txt`
   - **Start Command**: `cd backend && python main.py`
   - **Plan**: `Free`

### Step 3: Deploy Frontend
1. Click "New +" → "Static Site"
2. Connect the same GitHub repository
3. Configure:
   - **Name**: `ai-image-enhancer-frontend`
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Publish Directory**: `frontend/dist`
   - **Plan**: `Free`

### Step 4: Update Frontend API URL
1. In Render dashboard, go to your frontend service
2. Go to "Environment" tab
3. Add environment variable:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://your-backend-url.onrender.com` (copy from backend service)

### Step 5: Custom Domain (Optional)
1. In your frontend service → Settings → Custom Domains
2. Add your subdomain: `enhance.yourdomain.com`
3. Update your DNS records as instructed

## 🔧 Your URLs After Deployment:
- **Backend API**: `https://ai-image-enhancer-api.onrender.com`
- **Frontend**: `https://ai-image-enhancer-frontend.onrender.com`
- **Custom Domain**: `https://enhance.yourdomain.com`

## ⚠️ Free Tier Limitations:
- Apps sleep after 15 minutes of inactivity
- 512MB RAM limit (may cause issues with large images)
- 30-second request timeout
- CPU-only processing (slower enhancement)

## 🎯 Testing Your Deployment:
1. Visit your frontend URL
2. Upload a small image (< 1MB recommended for free tier)
3. Test enhancement with low strength (30-50%)
4. Check if download works

## 🐛 Troubleshooting:
- **Backend won't start**: Check logs in Render dashboard
- **Frontend can't connect**: Verify VITE_API_URL is correct
- **Enhancement fails**: Try smaller images or lower strength
- **Timeout errors**: Normal on free tier with large images