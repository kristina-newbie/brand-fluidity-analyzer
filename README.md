# Brand Liquidity Analyzer

A computational framework for measuring brand identity evolution over time, grounded in Bauman's (2000) liquid modernity theory. Uses Sentence-BERT, CLIP, NLTK, and Gensim to quantify how brands adapt their textual and visual identities across temporal windows.

Developed as part of academic research targeting publication in the *International Journal of Research in Marketing* (IJRM).

## What It Does

The Brand Liquidity Analyzer measures brand identity change through four core metrics:

1. **Textual Liquidity Index (TLI)** — Semantic drift in brand language over time (Sentence-BERT centroid distances)
2. **Visual Liquidity Index (VLI)** — Visual drift in brand imagery over time (CLIP centroid distances)
3. **Cross-Modal Coherence Score (CMCS)** — Alignment between textual and visual brand territories (centroid-to-centroid similarity in CLIP's joint latent space). Reported as a **separate axis**, not included in BLI.
4. **Brand Liquidity Index (BLI)** — Composite: `BLI = 0.5 × TLI + 0.5 × VLI`

Additionally, second-order diagnostic metrics capture *how* the brand evolves:

- **Drift Volatility** — Stability of evolution (sample std of drift scores)
- **Coherence Trajectory** — OLS slope of CMCS over time
- **Drift Ratio** — Semantic Drift / Visual Drift (raw values)
- **Directional Consistency** — Whether evolution follows a purposeful direction or oscillates
- **Topic-Semantic Decoupling Index (TSDI)** — Topic Turnover / Semantic Drift (cultural agility indicator)

### Key Methodological Distinctions

**Liquidity vs. Diversity.** Liquidity measures change *across* time periods, not variety *within* a period. A brand may have rich thematic content while maintaining perfect temporal stability.

**Liquidity vs. Coherence.** These are orthogonal dimensions. A brand can be highly liquid yet coherent (evolving in integrated ways across modalities), or solid yet fragmented (stable but misaligned across channels).

### Interpretation Categories

| BLI Range | Category | Interpretation |
|-----------|----------|----------------|
| 0.00–0.30 | **Solid** | Strong continuity across both dimensions |
| 0.30–0.60 | **Semi-solid** | Controlled evolution with stable core identity |
| 0.60–0.80 | **Semi-liquid** | Active repositioning, adaptive identity |
| 0.80–1.00 | **Liquid** | Fundamental brand transformation |

## Architecture

```
┌─────────────────────┐
│   React Frontend    │  ← Drag & drop interface (Lovable.dev)
└──────────┬──────────┘
           │ HTTP/REST
┌──────────▼──────────┐
│  FastAPI Backend     │  ← File upload, analysis, API v5.0
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │   Analysis  │
    ├─────────────┤
    │ • CLIP      │  ← Visual embeddings & cross-modal coherence
    │ • SBERT     │  ← Semantic text embeddings (all-MiniLM-L6-v2)
    │ • NLTK      │  ← Sentiment analysis (VADER)
    │ • Gensim    │  ← Topic modeling (LDA)
    │ • Claude AI │  ← Strategic insights generation
    └─────────────┘
```

## Quick Start

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

Required packages include: `fastapi`, `uvicorn`, `torch`, `clip`, `sentence-transformers`, `nltk`, `gensim`, `opencv-python`, `anthropic`, `numpy`, `pandas`, `scikit-learn`.

2. **Download NLTK data:**

```python
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

3. **Set up environment variables:**

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

4. **Run the backend:**

```bash
python main.py
```

The API will be available at `http://localhost:8000`. On first run, CLIP and Sentence-BERT models download automatically (~400MB).

### Frontend Setup

#### Option 1: Using Lovable (Recommended)

1. Go to [Lovable.dev](https://lovable.dev)
2. Create a new project
3. Copy `frontend/BrandFluidityAnalyzer.jsx` as the main component
4. Set environment variable: `REACT_APP_API_URL=http://localhost:8000`

#### Option 2: Manual React Setup

```bash
npx create-react-app brand-liquidity-frontend
cd brand-liquidity-frontend
npm install lucide-react
# Copy BrandFluidityAnalyzer.jsx to src/App.js
npm start
```

## Usage Guide

### 1. Upload Materials

Upload brand texts (`.txt`, `.md`) and images (`.jpg`, `.png`, etc.) organized by time period. Files must include the year in their filename for automatic period detection:

```
✅ nike_2019_manifesto.txt, campaign_2021.jpg
❌ manifesto.txt, campaign.jpg (no year → defaults to unknown)
```

### 2. Run Analysis

Click "Analyze Brand Liquidity". Processing typically takes 30–120 seconds depending on file count.

### 3. Interpret Results

Results include core metrics (TLI, VLI, CMCS, BLI), second-order pattern metrics, period-by-period evolution data, and AI-generated strategic insights powered by Claude.

## Technical Details

### Textual Liquidity Index (TLI)

```
TLI = Semantic Drift (normalized)
```

Semantic Drift is the average cosine distance between Sentence-BERT period centroids across consecutive time periods:

```
SD = (1/(n-1)) × Σ (1 - cos(μ_t, μ_{t+1}))
```

where μ_t is the centroid of all text embeddings in period t.

Normalization bounds: [0.05, 0.25]. Topic Turnover (LDA-based) and Tone Diversity (VADER) are computed as diagnostic descriptors but are **not** included in TLI.

### Visual Liquidity Index (VLI)

```
VLI = Visual Drift (normalized)
```

Visual Drift is the average cosine distance between CLIP image-embedding centroids across consecutive periods.

Normalization bounds: [0.02, 0.12]. Color Entropy and Motif Diversity are computed as descriptive metrics but are **not** included in VLI.

### Cross-Modal Coherence Score (CMCS)

Measures holistic alignment between textual and visual brand territories using centroid-to-centroid cosine similarity in CLIP's joint latent space:

```
CMCS = mean(cos(text_centroid_t, image_centroid_t))  across periods
```

Text and image corpora are treated as independent collections per period (not paired). CMCS is **deliberately excluded** from BLI — liquidity and coherence are orthogonal dimensions following Kapferer's (2012) Brand Identity Prism.

Normalization bounds: [0.20, 0.45], recalibrated for the centroid method (ecological correlation effect; Robinson, 1950).

### Brand Liquidity Index (BLI)

```
BLI = 0.5 × TLI + 0.5 × VLI
```

Equal weighting of textual and visual dimensions. Weights are configurable.

### Second-Order Metrics

These capture *how* the brand evolves, not just how much:

- **Drift Volatility**: Sample std (ddof=1) of period-to-period drift scores. Low = stable evolution; high = erratic changes.
- **Coherence Trajectory**: OLS slope of period CMCS over time. Positive = improving integration.
- **Drift Ratio**: Raw Semantic Drift / Raw Visual Drift. High-performing brands cluster around 3.0–3.5 ("visual fortress with narrative flexibility").
- **Directional Consistency**: Mean cosine similarity between consecutive drift vectors. Near +1 = purposeful evolution; near 0 = random drift.
- **TSDI**: Topic Turnover / Raw Semantic Drift. High = cultural agility (rotating themes while preserving semantic core).

## API Endpoints

### `POST /api/upload`

Upload brand materials.

**Parameters:** `brand_name` (string), `texts` (files[]), `images` (files[]), `years` (JSON, optional year mapping)

**Response:**
```json
{
  "status": "success",
  "brand_name": "Nike",
  "texts_uploaded": 5,
  "images_uploaded": 12,
  "text_files": [{"filename": "...", "year": "2019"}],
  "image_files": [{"filename": "...", "year": "2021"}]
}
```

### `POST /api/analyze`

Run liquidity analysis.

**Parameters:** `brand_name` (string), `include_llm_insights` (bool, default: true)

**Response:**
```json
{
  "brand_name": "Nike",
  "analyzer_version": "5.0",
  "periods_analyzed": ["2015", "2017", "2019", "2021", "2023", "2025"],
  "textual": {
    "tli": 0.45,
    "semantic_drift": 0.45,
    "semantic_drift_raw": 0.14,
    "topic_turnover": 0.67,
    "tone_diversity": 0.32,
    "period_similarities": {"2015->2017": 0.89, "2017->2019": 0.86}
  },
  "visual": {
    "vli": 0.28,
    "visual_drift": 0.28,
    "visual_drift_raw": 0.05,
    "color_entropy": 0.61,
    "motif_diversity": 0.43
  },
  "coherence": {
    "cmcs": 0.54,
    "cmcs_raw": 0.33,
    "temporal_coherence": 0.02,
    "period_coherences": {"2015": 0.31, "2017": 0.34}
  },
  "bli": {
    "score": 0.37,
    "category": "Semi-solid",
    "description": "Controlled evolution with stable core identity"
  },
  "second_order": {
    "semantic_drift_volatility": 0.03,
    "visual_drift_volatility": 0.01,
    "coherence_trajectory": 0.005,
    "drift_ratio": 2.8,
    "directional_consistency_text": 0.42,
    "directional_consistency_visual": 0.31,
    "topic_semantic_decoupling": 4.8
  },
  "llm_insights": {
    "insights": "Strategic analysis...",
    "model": "claude-sonnet-4-5"
  }
}
```

### `GET /api/brands`

List all analyzed brands with file counts and detected periods.

### `DELETE /api/brands/{brand_name}`

Delete a brand and its uploaded files.

## Project Structure

```
brand-liquidity-analyzer/
├── backend/
│   ├── main.py                      # FastAPI server (v5.0)
│   ├── brand_liquidity_analyzer.py  # Core metrics computation
│   ├── llm_analyzer.py              # Claude AI strategic insights
│   ├── requirements.txt             # Python dependencies
│   ├── .env                         # Environment variables (create this)
│   └── uploads/                     # Uploaded files (auto-created)
│       └── {brand_name}/
│           ├── texts/
│           └── images/
├── frontend/
│   └── BrandFluidityAnalyzer.jsx    # React component
├── IJRM-Core_Metrics.docx           # Research documentation
└── README.md
```

## Customization

### Adjust Metric Weights

In `brand_liquidity_analyzer.py`, modify the BLI computation:

```python
bli = self.compute_bli(tli, vli, weights=(0.6, 0.4))  # More text weight
```

### Adjust Normalization Bounds

```python
params = NormalizationParams(
    semantic_drift_min=0.05,
    semantic_drift_max=0.25,
    visual_drift_min=0.02,
    visual_drift_max=0.12,
    cmcs_min=0.20,
    cmcs_max=0.45,
)
analyzer = BrandLiquidityAnalyzer(norm_params=params)
```

### Change AI Model

In `llm_analyzer.py`:

```python
model="claude-sonnet-4-5-20250929"  # Change to preferred model
```

## Troubleshooting

### Installation Issues

**"ModuleNotFoundError: No module named 'clip'"**
```bash
pip install git+https://github.com/openai/CLIP.git
```

**"NLTK data not found"**
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
```

### Runtime Errors

**"Analysis failed: 500 Internal Server Error"** — Check backend terminal logs. Common causes: memory errors with large image sets (reduce batch size), CUDA/GPU issues (set `device = "cpu"`), or JSON serialization errors (already handled via `convert_numpy_types()`).

**"CORS errors in frontend"** — Ensure backend runs on `http://localhost:8000` and frontend `API_URL` matches. CORS is pre-configured in `main.py`.

**"Insufficient data"** — Need at least 2 text files for TLI and 2 images for VLI. Year must be in filename (4-digit: e.g., `brand_2023.txt`).

### Performance Tips

- 20–50 files per analysis is optimal
- Keep images under 1MB each
- Install CUDA-enabled PyTorch for 3–5× speedup on GPU

## Security Notes

- Uploaded files are stored in `/backend/uploads`
- Only text content and image embeddings are sent to Claude API
- In production: implement authentication, rate limiting, HTTPS, and restricted CORS origins

## Theoretical Foundations

This framework operationalizes concepts from:

- **Bauman (2000)** — Liquid modernity theory
- **Kapferer (2012)** — Brand Identity Prism (coherence as orthogonal to liquidity)
- **Reimers & Gurevych (2019)** — Sentence-BERT for semantic embeddings
- **Radford et al. (2021)** — CLIP for visual embeddings and cross-modal alignment
- **Hamilton et al. (2018)** — Diachronic semantic change measurement
- **Hutto & Gilbert (2014)** — VADER sentiment analysis
- **Blei et al. (2003)** — LDA topic modeling

## Version History

**v5.0** (February 2025)
- Renamed all terminology from "Fluidity" to "Liquidity" (aligned with Bauman's theory)
- TFI → TLI, VFI → VLI, BFI → BLI
- TLI = Semantic Drift only (Topic Turnover moved to diagnostic)
- VLI = Visual Drift only (Color Entropy, Motif Diversity moved to descriptive)
- BLI = 0.5 × TLI + 0.5 × VLI (CMCS excluded — orthogonal axis)
- CMCS rewritten: centroid-to-centroid cosine similarity in CLIP joint space
- Added second-order metrics: Drift Volatility, Coherence Trajectory, Drift Ratio, Directional Consistency, TSDI
- Normalization bounds recalibrated for centroid method
- Temporal Coherence uses sample std (ddof=1, Bessel's correction)

**v4.0** (December 2024)
- Separated TLI from Topic Turnover
- Separated VLI from Color Entropy / Motif Diversity
- Removed CMCS from BLI formula

**v1.1.0** (December 2024)
- Fixed JSON serialization for numpy types
- Added comprehensive error handling and logging

---

*Built for brand researchers and marketing professionals. Part of ongoing IJRM research on brand identity evolution in liquid modernity.*
