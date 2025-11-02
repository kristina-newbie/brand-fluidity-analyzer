# Deployment Guide - Brand Fluidity Analyzer

This guide covers different deployment options for the Brand Fluidity Analyzer.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Frontend Deployment with Lovable](#frontend-deployment-with-lovable)
5. [Production Considerations](#production-considerations)

---

## Local Development

### Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

# Set up environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run the server
python main.py
```

Backend will be available at `http://localhost:8000`

### Frontend (Manual React Setup)

```bash
# Create React app
npx create-react-app brand-fluidity-frontend
cd brand-fluidity-frontend

# Install dependencies
npm install lucide-react

# Copy the component
cp ../frontend/BrandFluidityAnalyzer.jsx src/App.js

# Set API URL
echo "REACT_APP_API_URL=http://localhost:8000" > .env

# Start dev server
npm start
```

Frontend will be available at `http://localhost:3000`

---

## Docker Deployment

### Quick Start with Docker Compose

```bash
# Set your API key
export ANTHROPIC_API_KEY=your_key_here

# Build and start
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop
docker-compose down
```

### Manual Docker Build

```bash
cd backend

# Build image
docker build -t brand-fluidity-backend .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -e ANTHROPIC_API_KEY=your_key_here \
  --name brand-fluidity-backend \
  brand-fluidity-backend

# View logs
docker logs -f brand-fluidity-backend

# Stop container
docker stop brand-fluidity-backend
docker rm brand-fluidity-backend
```

---

## Cloud Deployment

### Option 1: Railway.app (Recommended for Backend)

1. **Create account** at [Railway.app](https://railway.app)

2. **Deploy from GitHub**:
   ```bash
   # Push your code to GitHub
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin your-repo-url
   git push -u origin main
   ```

3. **In Railway dashboard**:
   - New Project → Deploy from GitHub
   - Select your repository
   - Add environment variable: `ANTHROPIC_API_KEY`
   - Railway will auto-detect the Dockerfile
   - Get your deployment URL

4. **Update frontend** to use Railway URL:
   ```
   REACT_APP_API_URL=https://your-app.railway.app
   ```

### Option 2: Render.com

1. **Create account** at [Render.com](https://render.com)

2. **Create Web Service**:
   - New → Web Service
   - Connect GitHub repository
   - Environment: Docker
   - Add environment variable: `ANTHROPIC_API_KEY`
   - Deploy

3. **Configure**:
   - Instance Type: Starter ($7/month) or Free
   - Auto-Deploy: Yes

### Option 3: AWS ECS / Google Cloud Run

See `CLOUD_DEPLOYMENT_ADVANCED.md` for detailed instructions.

---

## Frontend Deployment with Lovable

### Method 1: Lovable (Recommended - No Code)

1. **Go to [Lovable.dev](https://lovable.dev)**

2. **Create new project**:
   - Click "Create New Project"
   - Name: "Brand Fluidity Analyzer"

3. **Add the component**:
   - Open `src/App.tsx` (or create it)
   - Copy entire contents of `frontend/BrandFluidityAnalyzer.jsx`
   - Paste into Lovable editor

4. **Install dependencies**:
   - Go to "Dependencies" tab
   - Add: `lucide-react`

5. **Configure environment**:
   - Go to "Settings" → "Environment Variables"
   - Add: `REACT_APP_API_URL` = `https://your-backend-url.com`

6. **Deploy**:
   - Click "Deploy" button
   - Get your live URL
   - Share with your team!

### Method 2: Vercel (Alternative)

```bash
cd brand-fluidity-frontend

# Install Vercel CLI
npm i -g vercel

# Set environment variable
echo "REACT_APP_API_URL=https://your-backend-url.com" > .env.production

# Deploy
vercel --prod
```

### Method 3: Netlify

```bash
cd brand-fluidity-frontend

# Build
npm run build

# Install Netlify CLI
npm i -g netlify-cli

# Deploy
netlify deploy --prod --dir=build
```

Set environment variable in Netlify dashboard:
- Site settings → Environment variables
- Add: `REACT_APP_API_URL`

---

## Production Considerations

### Security

1. **API Authentication**:
   ```python
   # In main.py, add API key authentication
   from fastapi import Security, HTTPException
   from fastapi.security import APIKeyHeader
   
   API_KEY = os.getenv("API_SECRET_KEY")
   api_key_header = APIKeyHeader(name="X-API-Key")
   
   async def verify_api_key(api_key: str = Security(api_key_header)):
       if api_key != API_KEY:
           raise HTTPException(status_code=403, detail="Invalid API Key")
   ```

2. **CORS Configuration**:
   ```python
   # Update CORS settings in main.py
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://your-frontend-domain.com"],  # Specific domain
       allow_credentials=True,
       allow_methods=["POST", "GET"],
       allow_headers=["*"],
   )
   ```

3. **Rate Limiting**:
   ```bash
   pip install slowapi
   ```
   
   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   app.add_exception_handler(429, _rate_limit_exceeded_handler)
   
   @app.post("/api/analyze")
   @limiter.limit("5/minute")  # 5 requests per minute
   async def analyze_brand(...):
       ...
   ```

### Performance

1. **Use GPU for faster inference**:
   ```python
   # In brand_fluidity_analyzer.py
   device = "cuda" if torch.cuda.is_available() else "cpu"
   self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
   ```

2. **Implement caching**:
   ```bash
   pip install redis
   ```
   
   ```python
   import redis
   r = redis.Redis(host='localhost', port=6379)
   
   # Cache analysis results
   cache_key = f"analysis:{brand_name}"
   if r.exists(cache_key):
       return json.loads(r.get(cache_key))
   ```

3. **Async processing** for large datasets:
   ```bash
   pip install celery
   ```

### Monitoring

1. **Add logging**:
   ```python
   import logging
   
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   logger = logging.getLogger(__name__)
   ```

2. **Health checks**:
   ```python
   @app.get("/health")
   async def health_check():
       return {
           "status": "healthy",
           "timestamp": datetime.now().isoformat()
       }
   ```

3. **Error tracking** with Sentry:
   ```bash
   pip install sentry-sdk
   ```
   
   ```python
   import sentry_sdk
   sentry_sdk.init(dsn="your-sentry-dsn")
   ```

### Scaling

1. **Load balancing**: Use nginx or cloud load balancer

2. **Database**: Add PostgreSQL for persistent storage
   ```bash
   pip install sqlalchemy psycopg2-binary
   ```

3. **File storage**: Use S3 instead of local storage
   ```bash
   pip install boto3
   ```

### Backup

```bash
# Backup uploads directory
tar -czf uploads-backup-$(date +%Y%m%d).tar.gz uploads/

# Automated backup with cron
0 2 * * * tar -czf /backups/uploads-$(date +\%Y\%m\%d).tar.gz /app/uploads/
```

---

## Environment Variables Reference

### Backend (.env)

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-xxx

# Optional
HOST=0.0.0.0
PORT=8000
API_SECRET_KEY=your-secret-key
DEBUG=False

# Database (if using)
DATABASE_URL=postgresql://user:pass@localhost/dbname

# Redis (if using)
REDIS_URL=redis://localhost:6379

# Sentry (if using)
SENTRY_DSN=https://xxx@sentry.io/xxx
```

### Frontend

```bash
REACT_APP_API_URL=https://your-backend.com
REACT_APP_API_KEY=your-api-key  # if using API authentication
```

---

## Testing Deployment

```bash
# Test backend health
curl http://localhost:8000/

# Test upload endpoint
curl -X POST http://localhost:8000/api/upload \
  -F "brand_name=Test Brand" \
  -F "texts=@sample.txt" \
  -F "images=@sample.jpg"

# Test analysis
curl -X POST http://localhost:8000/api/analyze \
  -F "brand_name=Test Brand"
```

---

## Troubleshooting

### "Out of memory" errors
- Reduce batch size in analysis
- Use smaller models (e.g., ViT-B/16 instead of ViT-B/32)
- Process files sequentially instead of in parallel

### Slow first request
- Models are loaded on first request
- Consider pre-loading in Docker entrypoint
- Use model caching

### CORS errors
- Check `allow_origins` in CORS middleware
- Ensure frontend URL matches exactly (no trailing slash)

### File upload failures
- Check file size limits
- Verify upload directory permissions
- Check available disk space

---

## Support & Maintenance

- **Updates**: Regularly update dependencies
  ```bash
  pip list --outdated
  pip install --upgrade package-name
  ```

- **Security**: Run security scans
  ```bash
  pip install safety
  safety check
  ```

- **Monitoring**: Set up alerts for downtime, errors, high memory usage

---

**Need help?** Open an issue or consult the main README.md
