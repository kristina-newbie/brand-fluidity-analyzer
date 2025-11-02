# Brand Fluidity Analyzer - Project Structure

```
brand-fluidity-analyzer/
│
├── README.md                          # Main documentation
├── DEPLOYMENT.md                      # Deployment guide
├── docker-compose.yml                 # Docker Compose configuration
│
├── backend/                           # Python backend
│   ├── main.py                       # FastAPI application entry point
│   ├── brand_fluidity_analyzer.py    # Core metrics implementation
│   ├── llm_analyzer.py               # Claude AI integration
│   ├── requirements.txt              # Python dependencies
│   ├── Dockerfile                    # Docker configuration
│   ├── .env.example                  # Environment variables template
│   ├── setup.sh                      # Automated setup script
│   ├── test_setup.py                 # Setup verification script
│   ├── generate_sample_data.py       # Test data generator
│   │
│   └── uploads/                      # File upload directory (created on first run)
│       ├── texts/                    # Text files storage
│       └── images/                   # Image files storage
│
└── frontend/                          # React frontend
    └── BrandFluidityAnalyzer.jsx     # Main React component (Lovable-ready)
```

## File Descriptions

### Root Level

- **README.md**: Complete guide including usage, API docs, and technical details
- **DEPLOYMENT.md**: Production deployment instructions for various platforms
- **docker-compose.yml**: One-command deployment configuration

### Backend (`/backend`)

#### Core Application Files

- **main.py** (180 lines)
  - FastAPI server setup
  - REST API endpoints
  - File upload handling
  - CORS configuration

- **brand_fluidity_analyzer.py** (450 lines)
  - `BrandFluidityAnalyzer` class
  - Implements all 4 core metrics:
    * Textual Fluidity Index (TFI)
    * Visual Fluidity Index (VFI)
    * Cross-Modal Coherence Score (CMCS)
    * Brand Fluidity Index (BFI)
  - Uses: CLIP, SBERT, NLTK, Gensim, OpenCV

- **llm_analyzer.py** (100 lines)
  - `LLMAnalyzer` class
  - Claude API integration
  - Strategic insights generation
  - Prompt engineering for brand analysis

#### Configuration & Setup

- **requirements.txt**
  - All Python dependencies
  - Includes: fastapi, torch, clip, transformers, cv2, nltk, gensim, anthropic

- **Dockerfile**
  - Containerization configuration
  - Multi-stage build for optimization
  - NLTK data pre-download

- **.env.example**
  - Environment variables template
  - API key configuration

#### Utility Scripts

- **setup.sh** (Bash)
  - Automated backend setup
  - Dependency installation
  - NLTK data download
  - Directory creation

- **test_setup.py** (200 lines)
  - Verify installation
  - Test all imports
  - Check model loading
  - Validate API key

- **generate_sample_data.py** (150 lines)
  - Create test dataset
  - Generate synthetic texts (4 years)
  - Generate synthetic images (12 images)
  - Includes README for sample data

### Frontend (`/frontend`)

- **BrandFluidityAnalyzer.jsx** (600 lines)
  - Complete React component
  - Three-tab interface:
    * Upload: Drag-and-drop file upload
    * Analyze: Trigger analysis
    * Results: Metric visualization
  - Features:
    * File management (add/remove)
    * Year assignment
    * Progress indicators
    * Metric cards and charts
    * AI insights display
  - Styling: Tailwind CSS
  - Icons: Lucide React
  - **Fully Lovable-compatible!**

## Technology Stack

### Backend

| Category | Technology | Purpose |
|----------|-----------|---------|
| Framework | FastAPI | REST API server |
| ML/AI | PyTorch, CLIP | Image/text embeddings |
| NLP | SBERT, NLTK | Text analysis, sentiment |
| Topic Modeling | Gensim, LDA | Topic extraction |
| Computer Vision | OpenCV | Image processing |
| LLM | Anthropic Claude | Strategic insights |

### Frontend

| Category | Technology | Purpose |
|----------|-----------|---------|
| Framework | React | UI components |
| Styling | Tailwind CSS | Responsive design |
| Icons | Lucide React | UI icons |
| Builder | Lovable.dev | No-code deployment |

## Data Flow

```
User uploads files
    ↓
Frontend (React)
    ↓ HTTP POST /api/upload
Backend (FastAPI)
    ↓
Files saved to /uploads
    ↓
User triggers analysis
    ↓ HTTP POST /api/analyze
Brand Fluidity Analyzer
    ├→ Text Analysis (SBERT, NLTK, Gensim)
    ├→ Image Analysis (CLIP, OpenCV)
    ├→ Cross-Modal Analysis (CLIP)
    └→ Composite BFI Calculation
         ↓
    LLM Analyzer (Claude API)
         ↓
    Strategic Insights
         ↓
    JSON Response
         ↓
    Frontend Visualization
         ↓
    User sees results
```

## API Endpoints

### POST `/api/upload`
- **Purpose**: Upload brand materials
- **Input**: FormData with texts, images, brand_name, years
- **Output**: Upload confirmation with file counts

### POST `/api/analyze`
- **Purpose**: Analyze brand fluidity
- **Input**: FormData with brand_name, include_llm_insights
- **Output**: Complete analysis results with all metrics

### GET `/api/brands`
- **Purpose**: List all analyzed brands
- **Output**: Array of brands with file counts

### DELETE `/api/brands/{brand_name}`
- **Purpose**: Delete brand and files
- **Output**: Deletion confirmation

### GET `/`
- **Purpose**: API health check
- **Output**: Version and status

## Metrics Calculation

### 1. Textual Fluidity Index (TFI)

**Components:**
- Semantic Drift: `1 - cosine_similarity(embedding_t1, embedding_t2)`
- Tone Diversity: `std(sentiment_scores)`
- Topic Turnover: `std(topic_distributions)`

**Formula:** `TFI = (SD + TD + TT) / 3`

### 2. Visual Fluidity Index (VFI)

**Components:**
- Visual Drift: `mean(distance_to_cluster_center)`
- Color Entropy: `Shannon_entropy(HSV_histogram)`
- Motif Diversity: `std(embeddings) / mean(embeddings)`

**Formula:** `VFI = (VD + CE + MD) / 3`

### 3. Cross-Modal Coherence (CMCS)

**Calculation:** `mean(cosine_similarity(text_embed, image_embed))`

### 4. Brand Fluidity Index (BFI)

**Formula:** `BFI = 0.4*TFI + 0.4*VFI + 0.2*(1-CMCS)`

## Models Used

1. **CLIP (ViT-B/32)**: 400MB
   - Joint image-text embeddings
   - Cross-modal coherence

2. **Sentence-BERT (all-MiniLM-L6-v2)**: 80MB
   - Text embeddings
   - Semantic similarity

3. **VADER**: <1MB
   - Sentiment analysis
   - Built into NLTK

4. **LDA (Gensim)**: Runtime
   - Topic modeling
   - Dynamic creation

## Memory Requirements

- **Minimum**: 4GB RAM
- **Recommended**: 8GB RAM
- **With GPU**: 6GB VRAM + 8GB RAM
- **Disk Space**: 2GB (models + dependencies)

## Processing Time

| Files | Processing Time |
|-------|----------------|
| 5 texts + 10 images | 30-60 seconds |
| 20 texts + 50 images | 2-4 minutes |
| 50 texts + 100 images | 5-10 minutes |

*Times are for CPU processing. GPU is 3-5x faster.*

## Extensibility

### Adding New Metrics

1. Create method in `BrandFluidityAnalyzer`:
```python
def calculate_custom_metric(self, data):
    # Your implementation
    return score
```

2. Call in `analyze_brand()`:
```python
results["custom_metric"] = self.calculate_custom_metric(data)
```

3. Display in frontend:
```jsx
<MetricCard
  title="Custom Metric"
  value={results.custom_metric}
/>
```

### Adding New Data Sources

1. Add endpoint in `main.py`
2. Implement data fetching
3. Pass to analyzer
4. Update frontend UI

### Changing AI Models

**CLIP alternatives:**
- BLIP-2: Better captioning
- OpenCLIP: Larger models

**SBERT alternatives:**
- all-mpnet-base-v2: Higher quality
- multilingual-e5: Multi-language

**LLM alternatives:**
- Claude Opus: More sophisticated
- GPT-4: Alternative provider

## Best Practices

### File Organization
- Use year prefixes: `2023_campaign.txt`
- Consistent naming: `brand_year_type.ext`
- Group by campaign/period

### Data Quality
- Minimum 2 time periods for TFI
- Minimum 2 images for VFI
- Pair texts with images for CMCS

### Performance
- Batch similar file types
- Use GPU when available
- Cache analysis results

### Security
- API key authentication (production)
- Rate limiting (production)
- Input validation
- File type restrictions

## Future Enhancements

- [ ] Database integration (PostgreSQL)
- [ ] Batch processing queue (Celery)
- [ ] Real-time monitoring dashboard
- [ ] Multi-brand comparison view
- [ ] Time-series trend charts
- [ ] PDF report generation
- [ ] Social media API integration
- [ ] Webhook notifications

## License & Attribution

Based on brand fluidity research and liquid modernity theory.

**Model Credits:**
- CLIP: OpenAI
- Sentence-BERT: UKP Lab
- VADER: NLTK Project

---

**Questions?** See README.md or open an issue!
