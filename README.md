# Brand Fluidity Analyzer

A comprehensive application for analyzing how brand identities evolve across text and visual content over time, using advanced NLP, computer vision, and AI-powered insights.

## 📊 What It Does

The Brand Fluidity Analyzer calculates four core metrics based on academic research in brand semiotics:

1. **Textual Fluidity Index (TFI)** - Measures semantic drift, tone diversity, and topic turnover in brand messaging
2. **Visual Fluidity Index (VFI)** - Analyzes visual evolution through color entropy, motif diversity, and visual drift
3. **Cross-Modal Coherence Score (CMCS)** - Evaluates alignment between text and visual content
4. **Brand Fluidity Index (BFI)** - Composite score with AI-generated strategic insights

### Fluidity Categories

- **0.0-0.3 (Solid)**: Consistent identity, stable brand cues
- **0.3-0.6 (Semi-fluid)**: Balanced - recognizable yet adaptive
- **0.6-1.0 (Highly Fluid)**: Dynamic, context-responsive identity

## 🏗️ Architecture

```
┌─────────────────────┐
│   React Frontend    │  ← Drag & drop interface
│   (Lovable-ready)   │
└──────────┬──────────┘
           │ HTTP/REST
┌──────────▼──────────┐
│   FastAPI Backend   │  ← File upload & analysis
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │   Analysis  │
    ├─────────────┤
    │ • CLIP      │  ← Image/text embeddings
    │ • SBERT     │  ← Semantic analysis
    │ • NLTK      │  ← Sentiment analysis
    │ • Gensim    │  ← Topic modeling
    │ • OpenCV    │  ← Visual analysis
    │ • Claude AI │  ← Strategic insights
    └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 14+ (for frontend)
- Anthropic API key (for AI insights)
- ~2GB free disk space (for ML models)

### Backend Setup

1. **Install Python dependencies:**

```bash
cd backend
pip install -r requirements.txt --break-system-packages
```

Required packages include:
- `fastapi` - API framework
- `uvicorn` - ASGI server
- `torch` - PyTorch for deep learning
- `clip` - OpenAI's CLIP model
- `sentence-transformers` - Sentence-BERT
- `nltk` - Natural language toolkit
- `gensim` - Topic modeling
- `opencv-python` - Computer vision
- `anthropic` - Claude API client
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities

2. **Download NLTK data:**

```python
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

3. **Set up environment variables:**

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

Or create a `.env` file with:
```
ANTHROPIC_API_KEY=your_api_key_here
```

4. **Run the backend:**

```bash
python main.py
```

The API will be available at `http://localhost:8000`

**First run**: Models (CLIP, Sentence-BERT) will download automatically (~400MB). This may take a few minutes.

### Frontend Setup

#### Option 1: Using Lovable (Recommended)

1. Go to [Lovable.dev](https://lovable.dev)
2. Create a new project
3. Copy the contents of `frontend/BrandFluidityAnalyzer.jsx`
4. Paste it as your main component
5. Install required dependencies:
   - `lucide-react` (for icons)
6. Set environment variable: `REACT_APP_API_URL=http://localhost:8000`
7. Deploy or run locally

#### Option 2: Manual React Setup

```bash
npx create-react-app brand-fluidity-frontend
cd brand-fluidity-frontend
npm install lucide-react
```

Copy `BrandFluidityAnalyzer.jsx` to `src/App.js`, then:

```bash
npm start
```

## 📖 Usage Guide

### 1. Upload Tab

1. **Enter Brand Name**: Type the brand you're analyzing (e.g., "Nike", "Apple")

2. **Upload Text Files**:
   - Drag and drop `.txt` or `.md` files
   - Examples: brand manifestos, ad copy, social media posts, website text
   - Optionally assign years to track evolution

3. **Upload Images**:
   - Drag and drop images (`.jpg`, `.png`, etc.)
   - Examples: logos, product photos, ad visuals, social media graphics
   - Optionally assign years

4. **Click "Upload Files"**

### 2. Analyze Tab

1. Review your uploaded files
2. Click "Analyze Brand Fluidity"
3. Wait for processing (typically 30-120 seconds depending on file count)

### 3. Results Tab

View comprehensive analysis including:

- **Overall BFI Score** and category
- **Individual metric breakdowns**
- **Component scores** (semantic drift, color entropy, etc.)
- **AI-Generated Strategic Insights** powered by Claude
- **Recommendations** for brand strategy

## 🎯 Use Cases

### Brand Managers
- Monitor brand consistency over time
- Identify when messaging drifts from core values
- Optimize the balance between stability and adaptation

### Marketing Agencies
- Benchmark clients against competitors
- Demonstrate ROI of brand consistency efforts
- Guide rebranding decisions with data

### Researchers
- Study brand evolution patterns
- Analyze cultural shifts through brand language
- Explore cross-modal communication strategies

### Example Analysis Workflow

```
Brand: Sustainable Fashion Startup
Files: 
  - 2020_mission.txt
  - 2021_campaign.txt
  - 2022_manifesto.txt
  - 2023_website.txt
  + 15 images from campaigns (2020-2023)

Results:
  TFI: 0.45 (Semi-fluid) - Adaptive messaging
  VFI: 0.62 (Fluid) - Evolving visual identity
  CMCS: 0.78 (Strong coherence)
  BFI: 0.52 (Semi-fluid)
  
Insight: Brand maintains semantic core while 
         experimenting with visual expression.
         Recommendation: Establish visual guidelines
         to match textual consistency.
```

## 🔧 API Endpoints

### POST `/api/upload`
Upload brand materials.

**Parameters:**
- `brand_name` (string): Brand identifier
- `texts` (files[]): Text files
- `images` (files[]): Image files
- `years` (JSON): Optional year mapping

**Response:**
```json
{
  "status": "success",
  "texts_uploaded": 5,
  "images_uploaded": 12
}
```

### POST `/api/analyze`
Run fluidity analysis.

**Parameters:**
- `brand_name` (string): Brand to analyze
- `include_llm_insights` (bool): Generate AI insights

**Response:**
```json
{
  "brand_name": "Nike",
  "textual_fluidity": {
    "TFI": 0.612,
    "semantic_drift": 0.543,
    "tone_diversity": 0.621,
    "topic_turnover": 0.672,
    "interpretation": "Semi-fluid..."
  },
  "visual_fluidity": {...},
  "cross_modal_coherence": {...},
  "brand_fluidity_index": {
    "BFI": 0.648,
    "category": "Semi-fluid"
  },
  "llm_insights": {
    "insights": "Strategic analysis..."
  }
}
```

### GET `/api/brands`
List all analyzed brands.

### DELETE `/api/brands/{brand_name}`
Delete a brand and its files.

## 📚 Technical Details

### Textual Fluidity Index (TFI)

**Components:**
1. **Semantic Drift**: Cosine distance between SBERT embeddings across time periods
2. **Tone Diversity**: Standard deviation of VADER sentiment scores
3. **Topic Turnover**: LDA-based topic distribution variance

**Formula:**
```
TFI = (1/3) × (SD + TD + TT)
```

### Visual Fluidity Index (VFI)

**Components:**
1. **Visual Drift**: K-means clustering variance of CLIP embeddings
2. **Color Entropy**: Shannon entropy of HSV color histograms
3. **Motif Diversity**: Embedding distribution standard deviation

**Formula:**
```
VFI = (1/3) × (VD + CE + MD)
```

### Cross-Modal Coherence Score (CMCS)

Measures text-image alignment using CLIP's joint embedding space:
```
CMCS = mean(cosine_similarity(text_embed, image_embed))
```

### Brand Fluidity Index (BFI)

Composite score:
```
BFI = 0.4×TFI + 0.4×VFI + 0.2×(1-CMCS)
```

## 🛠️ Customization

### Adjust Metric Weights

In `brand_fluidity_analyzer.py`, modify the `compute_brand_fluidity_index` function:

```python
def compute_brand_fluidity_index(self, TFI, VFI, CMCS, 
                                  alpha=0.5,  # ← Increase text weight
                                  beta=0.3,   # ← Decrease visual weight
                                  gamma=0.2):
    ...
```

### Change AI Model

In `llm_analyzer.py`, update the model:

```python
message = self.client.messages.create(
    model="claude-sonnet-4-5-20250929",  # ← Change model
    max_tokens=2000,
    ...
)
```

### Add New Metrics

Extend `BrandFluidityAnalyzer` class with custom analysis functions:

```python
def calculate_custom_metric(self, data):
    # Your metric calculation
    return score
```

## 🎨 Frontend Customization

The React component is fully customizable:

- **Colors**: Modify Tailwind classes (`bg-blue-600` → `bg-purple-600`)
- **Layout**: Adjust grid layouts (`md:grid-cols-2` → `md:grid-cols-3`)
- **Components**: Add charts using Recharts or Chart.js
- **Styling**: Works with any CSS framework (Tailwind, Material-UI, etc.)

## 🐛 Troubleshooting

### Installation Issues

#### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch torchvision --break-system-packages
```

#### "ModuleNotFoundError: No module named 'clip'"
```bash
pip install git+https://github.com/openai/CLIP.git
```

#### "NLTK data not found"
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
```

### Model & Network Issues

#### "CLIP model download fails"
The model downloads automatically on first run. Ensure:
- Active internet connection
- Sufficient disk space (~400MB for models)
- No firewall blocking GitHub/Hugging Face

If download fails, try:
```python
import clip
clip.load("ViT-B/32", device="cpu")  # Forces download
```

#### "Sentence-BERT model download fails"
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')  # Forces download
```

### Runtime Errors

#### "Analysis failed: Internal Server Error" or "500 Internal Server Error"

**Check backend logs** in the terminal where you ran `python main.py`. Common causes:

1. **Memory Error** (too many images):
   - Solution: Process in smaller batches (reduce image count)
   - Or: Increase system memory/swap

2. **CUDA/GPU Error**:
   ```python
   # In brand_fluidity_analyzer.py, force CPU usage:
   device = "cpu"  # Instead of "cuda"
   ```

3. **JSON Serialization Error** (`TypeError: keys must be str, int, float, bool or None, not numpy.int32`):
   - **Fixed in latest version** - ensure your `main.py` has the `convert_numpy_types()` function
   - Update with the latest code that includes numpy type conversion

#### "CORS errors in frontend"
Ensure:
- Backend is running on `http://localhost:8000`
- Frontend `API_URL` matches backend address
- CORS is enabled in `main.py` (already configured)

#### "Analysis takes too long"
Performance tips:
- **Reduce file count**: 20-50 files per analysis is optimal
- **Resize images**: Keep under 1MB each
- **Use GPU**: Install CUDA-enabled PyTorch for 3-5x speedup
- **Batch processing**: Already implemented for large image sets

### Data Issues

#### "Insufficient data" errors
- Need at least **2 text files** for TFI calculation
- Need at least **2 images** for VFI calculation
- Year detection requires year in filename (e.g., `brand_2023.txt`)

#### "Year not detected from filename"
Filename must contain 4-digit year:
```
✅ Good: nike_2020_manifesto.txt, campaign_2021.jpg
❌ Bad: manifesto.txt, campaign.jpg
```

Files without years default to 2024.

### API Issues

#### "Brand not found" (404)
- Ensure you clicked "Upload Files" before "Analyze"
- Check that files were successfully uploaded
- Verify brand name matches exactly (case-sensitive)

#### "Anthropic API Error"
- Check your `.env` file has valid `ANTHROPIC_API_KEY`
- Verify API key has sufficient credits
- Check [Anthropic status page](https://status.anthropic.com)

### Debug Mode

For detailed error messages, run backend with:
```bash
python3 main.py  # Already has detailed logging
```

Look for:
```
======================================================================
ERROR OCCURRED DURING ANALYSIS:
Error type: [Error Name]
Error message: [Details]
======================================================================
```

## 📝 File Naming Conventions

For automatic year detection, include year in filename:

```
✅ Good:
- nike_2020_manifesto.txt
- campaign_2021.jpg
- brand_guidelines_2023.txt

❌ Not automatically detected:
- manifesto.txt (defaults to 2024)
- campaign.jpg (defaults to 2024)
```

## 📂 Project Structure

```
brand-fluidity-analyzer/
├── backend/
│   ├── main.py                      # FastAPI server with error handling
│   ├── brand_fluidity_analyzer.py   # Core metrics calculation
│   ├── llm_analyzer.py              # Claude AI insights
│   ├── requirements.txt             # Python dependencies
│   ├── .env                         # Environment variables (create this)
│   └── uploads/                     # Uploaded files (auto-created)
│       └── {brand_name}/
│           ├── texts/
│           └── images/
├── frontend/
│   └── BrandFluidityAnalyzer.jsx    # React component
├── IJRM-Core_Metrics.docx           # Research documentation
└── README.md                        # This file
```

## 🔐 Security Notes

- The app stores uploaded files in `/backend/uploads` directory
- Only text content and image embeddings are sent to Claude API for insights
- Delete brands after analysis to manage storage
- In production, implement:
  - Authentication/authorization
  - Rate limiting
  - Input validation and sanitization
  - HTTPS/TLS
  - File size limits
  - Restricted CORS origins

## 📄 License

This project is based on academic research in brand semiotics and fluidity theory.

## 🙏 Acknowledgments

- **CLIP** by OpenAI - Vision-language model
- **Sentence-BERT** by UKP Lab - Semantic text embeddings
- **Claude** by Anthropic - Strategic insights generation
- **Brand fluidity metrics** based on liquid modernity theory (Bauman, 2000)

## 💡 Future Enhancements

- [ ] Multi-brand comparative analysis
- [ ] Time-series visualizations with charts
- [ ] Export reports as PDF with branded templates
- [ ] Real-time brand monitoring via APIs
- [ ] Social media API integrations (Twitter, Instagram)
- [ ] Custom metric builder UI
- [ ] Historical benchmark database
- [ ] GPU acceleration for faster processing
- [ ] Batch analysis for multiple brands
- [ ] Advanced filtering and search

## 📧 Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review backend logs in terminal for detailed errors
3. Refer to research documentation in `IJRM-Core_Metrics.docx`
4. Open an issue on GitHub with:
   - Error message from terminal
   - Steps to reproduce
   - File counts and sizes
   - Python version and OS

## 🔄 Recent Updates

**v1.1.0** (December 2024)
- ✅ Fixed JSON serialization error for numpy types
- ✅ Added comprehensive error handling and logging
- ✅ Improved memory management for large image sets
- ✅ Enhanced debugging output for troubleshooting
- ✅ Added numpy type conversion utility

---

**Built with ❤️ for brand researchers and marketing professionals**
