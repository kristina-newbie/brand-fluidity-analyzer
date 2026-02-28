# Quick Start Guide - Brand Fluidity Analyzer

Get your Brand Fluidity Analyzer running in 10 minutes!

## 🚀 Fastest Path: Backend + Lovable Frontend

### Step 1: Backend Setup (5 minutes)

**Supported Python:** 3.10, 3.11, or 3.12 (3.14 is not yet supported by some dependencies, e.g. gensim).

```bash
# From project root
chmod +x setup.sh
./setup.sh

# Configure API key (edit backend/.env)
# Add your ANTHROPIC_API_KEY

# Start server (with venv activated)
cd backend && python main.py
```

Or manual setup from project root: `pip install -r backend/requirements-dev.txt`, then create `backend/.env` from `backend/.env.example`.

✅ Backend running at: `http://localhost:8000`

### Step 2: Frontend with Lovable (5 minutes)

1. **Go to [Lovable.dev](https://lovable.dev)** and sign in

2. **Create New Project**:
   - Click "New Project"
   - Name: "Brand Fluidity Analyzer"

3. **Add Component**:
   - Open the main app file (usually `src/App.tsx`)
   - Copy **entire content** from `frontend/BrandFluidityAnalyzer.jsx`
   - Paste into Lovable editor

4. **Install Dependencies**:
   - In Lovable, go to "Dependencies"
   - Add: `lucide-react`

5. **Set Environment**:
   - Go to Settings → Environment Variables
   - Add: `REACT_APP_API_URL` = `http://localhost:8000`

6. **Preview & Deploy**:
   - Click "Preview" to test
   - Click "Deploy" when ready
   - Get your live URL!

### Step 3: Test with Sample Data (2 minutes)

```bash
# Generate test data
cd backend
python generate_sample_data.py

# Sample data created in sample_data/
```

Now upload the files from `sample_data/texts/` and `sample_data/images/` in your app!

---

## 🐳 Alternative: Docker (One Command)

```bash
# Set API key
export ANTHROPIC_API_KEY=your_key_here

# Start everything
docker-compose up -d

# Access
# Backend: http://localhost:8000
# Frontend: Deploy separately or add to docker-compose
```

---

## 📝 Your First Analysis

1. **Open the app** (Lovable URL or localhost:3000)

2. **Upload Tab**:
   - Enter brand name: "Sample Brand"
   - Drag & drop text files (or use sample data)
   - Drag & drop images
   - Optionally assign years

3. **Analyze Tab**:
   - Click "Analyze Brand Fluidity"
   - Wait 30-60 seconds

4. **Results Tab**:
   - View BFI score and category
   - Read AI insights
   - Check individual metrics

---

## 🎯 Expected Output

```
Brand Fluidity Index: 0.523
Category: Semi-fluid
Interpretation: Balanced brand identity - recognizable yet adaptive

Components:
├─ Textual Fluidity (TFI): 0.451
├─ Visual Fluidity (VFI): 0.678
└─ Cross-Modal Coherence (CMCS): 0.756

AI Insights: "This brand demonstrates..."
```

---

## 🔧 Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt --break-system-packages
```

### "NLTK data not found"
```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

### "CORS error"
- Check backend is running: `curl http://localhost:8000`
- Verify `REACT_APP_API_URL` matches backend address

### "Analysis too slow"
- Reduce number of files
- First run downloads models (~500MB)
- Subsequent runs are faster

### "No AI insights"
- Check `ANTHROPIC_API_KEY` in `.env`
- Get key from: https://console.anthropic.com/

---

## 📚 Next Steps

1. **Read the docs**:
   - `README.md` - Full documentation
   - `DEPLOYMENT.md` - Production deployment
   - `PROJECT_STRUCTURE.md` - Technical details

2. **Customize**:
   - Adjust metric weights in `brand_fluidity_analyzer.py`
   - Modify UI in `BrandFluidityAnalyzer.jsx`
   - Add new metrics

3. **Deploy to production**:
   - Backend: Railway, Render, AWS
   - Frontend: Lovable (already deployed!)

---

## 📞 Need Help?

- **Setup issues**: Run `python test_setup.py`
- **API issues**: Check logs in backend terminal
- **Frontend issues**: Check browser console (F12)

---

## 🎉 You're Ready!

Your Brand Fluidity Analyzer is now running. Start analyzing brand evolution!

**Pro Tips**:
- Use consistent file naming (include years)
- Upload 5-20 files for best results
- Compare brands over time
- Use AI insights for strategic decisions

---

**Built with ❤️ for brand researchers and marketers**
