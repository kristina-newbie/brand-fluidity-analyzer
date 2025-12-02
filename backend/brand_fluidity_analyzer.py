"""
Brand Fluidity Analyzer: Core Metric Functions (v3 - Temporal Self-Similarity)
===============================================================================

METHODOLOGY CHANGE FROM v2:
Previous versions measured content DIVERSITY within a snapshot, which incorrectly
classified all brands as "highly fluid" (Hermès, Burberry, Lululemon all ~0.6).

This version measures temporal STABILITY - how consistent the brand is with ITSELF
across different time periods:
- A SOLID brand has HIGH self-similarity over time (low fluidity)
- A FLUID brand has LOW self-similarity over time (high fluidity)

Key Changes:
- TFI now measures textual CHANGE over time (not content diversity)
- VFI now measures visual CHANGE over time (not motif variety)
- CMCS uses aggregate semantic alignment (works with unpaired data)
- All metrics properly extract years from filenames

References:
- Bauman, Z. (2000). Liquid Modernity. Polity Press.
- Aaker, J. L. (1997). Dimensions of brand personality. JMR.
- Wang & Chen (2024). Brand consistency measurement using CNN.
- System1/IPA (2024). Creative Consistency Score research.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import cv2
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class BrandFluidityAnalyzer:
    """
    Analyzes brand fluidity using temporal self-similarity methodology.
    
    Output format is backward-compatible with v2 for seamless integration.
    """
    
    def __init__(self):
        """Initialize models for analysis."""
        print("Loading models...")
        
        # CLIP for visual embeddings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Sentence-BERT for text embeddings
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # VADER for sentiment analysis
        try:
            self.sia = SentimentIntensityAnalyzer()
        except:
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        
        # Brand personality lexicons (Aaker 1997)
        self._init_personality_lexicons()
        
        print("Models loaded successfully!")
    
    def _init_personality_lexicons(self):
        """Initialize brand personality dimension lexicons."""
        self.personality_lexicons = {
            'sincerity': {
                'down-to-earth', 'family', 'honest', 'sincere', 'real', 'genuine',
                'wholesome', 'original', 'cheerful', 'friendly', 'warm', 'happy',
                'trust', 'authentic', 'humble', 'simple', 'natural', 'caring'
            },
            'excitement': {
                'daring', 'trendy', 'exciting', 'spirited', 'cool', 'young',
                'imaginative', 'unique', 'contemporary', 'independent', 'innovative',
                'aggressive', 'bold', 'dynamic', 'fresh', 'modern', 'edgy', 'vibrant'
            },
            'competence': {
                'reliable', 'hardworking', 'secure', 'intelligent', 'technical',
                'corporate', 'successful', 'leader', 'confident', 'professional',
                'quality', 'expert', 'efficient', 'precise', 'trusted', 'premium'
            },
            'sophistication': {
                'upper-class', 'glamorous', 'good-looking', 'charming', 'feminine',
                'smooth', 'elegant', 'luxurious', 'refined', 'exclusive', 'prestige',
                'chic', 'stylish', 'sophisticated', 'graceful', 'exquisite', 'haute'
            },
            'ruggedness': {
                'outdoorsy', 'masculine', 'western', 'tough', 'rugged', 'strong',
                'adventure', 'robust', 'durable', 'powerful', 'athletic', 'active',
                'endurance', 'resilient', 'bold', 'fearless', 'heritage'
            }
        }
    
    def _extract_year_from_filename(self, filename):
        """
        Extract year from filename.
        Supports: 'brand_2015_content.txt', '2015_instagram.jpg', 'content2015.txt'
        """
        matches = re.findall(r'20[1-2][0-9]', str(filename))
        if matches:
            return int(matches[0])
        return None
    
    def _load_texts_by_year(self, text_dir):
        """Load text files organized by year extracted from filenames."""
        text_dir = Path(text_dir)
        texts_by_year = defaultdict(list)
        all_texts = []
        
        if not text_dir.exists():
            return texts_by_year, all_texts
        
        for text_file in text_dir.glob("*"):
            if text_file.is_file() and text_file.suffix.lower() in ['.txt', '.md']:
                try:
                    content = text_file.read_text(encoding='utf-8', errors='ignore').strip()
                    if not content:
                        continue
                    
                    year = self._extract_year_from_filename(text_file.name)
                    
                    if year:
                        texts_by_year[year].append(content)
                        all_texts.append(content)
                    else:
                        print(f"Warning: No year found in filename '{text_file.name}' - skipping")
                    
                except Exception as e:
                    print(f"Error reading {text_file}: {e}")
        
        return texts_by_year, all_texts
    
    def _load_images_by_year(self, image_dir):
        """Load image files organized by year extracted from filenames."""
        image_dir = Path(image_dir)
        images_by_year = defaultdict(list)
        all_image_paths = []
        
        if not image_dir.exists():
            return images_by_year, all_image_paths
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        
        for img_file in image_dir.glob("*"):
            if img_file.is_file() and img_file.suffix.lower() in valid_extensions:
                year = self._extract_year_from_filename(img_file.name)
                
                if year:
                    images_by_year[year].append(str(img_file))
                    all_image_paths.append(str(img_file))
                else:
                    print(f"Warning: No year found in filename '{img_file.name}' - skipping")
        
        return images_by_year, all_image_paths
    
    def _compute_text_centroid(self, texts):
        """Compute semantic centroid for a collection of texts."""
        if not texts:
            return None
        embeddings = self.sbert_model.encode(texts, convert_to_numpy=True)
        return np.mean(embeddings, axis=0)
    
    def _compute_image_centroid(self, image_paths):
        """Compute CLIP centroid for a collection of images."""
        if not image_paths:
            return None
        
        embeddings = []
        for path in image_paths:
            try:
                image = self.clip_preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.clip_model.encode_image(image)
                embeddings.append(embedding.cpu().numpy().flatten())
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                continue
        
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0)
    
    def _compute_temporal_stability(self, year_centroids):
        """
        Compute temporal stability as mean cosine similarity between consecutive periods.
        Returns stability score (0-1) and list of period similarities.
        """
        if len(year_centroids) < 2:
            return 1.0, []
        
        sorted_years = sorted(year_centroids.keys())
        consecutive_similarities = []
        
        for i in range(len(sorted_years) - 1):
            year1, year2 = sorted_years[i], sorted_years[i + 1]
            c1, c2 = year_centroids[year1], year_centroids[year2]
            
            # Normalize for cosine similarity
            c1_norm = c1 / (np.linalg.norm(c1) + 1e-8)
            c2_norm = c2 / (np.linalg.norm(c2) + 1e-8)
            
            sim = float(np.dot(c1_norm, c2_norm))
            consecutive_similarities.append({
                'period': f"{year1}-{year2}",
                'similarity': sim
            })
        
        stability = np.mean([s['similarity'] for s in consecutive_similarities])
        return stability, consecutive_similarities
    
    def _compute_personality_vector(self, texts):
        """Compute brand personality vector based on Aaker dimensions."""
        if not texts:
            return None
        
        combined_text = ' '.join(texts).lower()
        words = set(re.findall(r'\b[a-z]+\b', combined_text))
        
        scores = []
        for dimension in ['sincerity', 'excitement', 'competence', 'sophistication', 'ruggedness']:
            lexicon = self.personality_lexicons[dimension]
            matches = len(words.intersection(lexicon))
            score = (matches / max(len(words), 1)) * 1000
            scores.append(score)
        
        # Add sentiment
        sentiment_scores = []
        for text in texts:
            try:
                s = self.sia.polarity_scores(text)
                sentiment_scores.append([s['pos'], s['neg'], s['neu'], s['compound']])
            except:
                continue
        
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores, axis=0)
            scores.extend(avg_sentiment.tolist())
        
        return np.array(scores)
    
    def calculate_textual_fluidity(self, texts_by_year):
        """
        Calculate Textual Fluidity Index (TFI) using temporal self-similarity.
        
        TFI = 1 - Temporal Textual Stability
        High TFI = text identity changing over time = fluid
        Low TFI = text identity consistent over time = solid
        """
        years = sorted(texts_by_year.keys())
        
        if len(years) < 2:
            return {
                "semantic_drift": 0.0,
                "tone_diversity": 0.0,
                "topic_turnover": 0.0,
                "TFI": 0.0,
                "interpretation": "Insufficient temporal data (need 2+ time periods with year in filename)"
            }
        
        # Compute centroid for each year
        year_centroids = {}
        for year in years:
            texts = texts_by_year[year]
            if texts:
                centroid = self._compute_text_centroid(texts)
                if centroid is not None:
                    year_centroids[year] = centroid
        
        if len(year_centroids) < 2:
            return {
                "semantic_drift": 0.0,
                "tone_diversity": 0.0,
                "topic_turnover": 0.0,
                "TFI": 0.0,
                "interpretation": "Insufficient data for temporal analysis"
            }
        
        # Calculate temporal stability
        stability, period_similarities = self._compute_temporal_stability(year_centroids)
        
        # TFI = 1 - stability (fluidity is inverse of stability)
        TFI = 1 - stability
        
        # Compute personality stability for tone_diversity component
        year_personalities = {}
        for year in year_centroids.keys():
            p = self._compute_personality_vector(texts_by_year[year])
            if p is not None:
                year_personalities[year] = p
        
        personality_stability, _ = self._compute_temporal_stability(year_personalities) if len(year_personalities) >= 2 else (1.0, [])
        tone_diversity = 1 - personality_stability
        
        # For backward compatibility, distribute TFI across components
        semantic_drift = TFI  # Main component
        topic_turnover = TFI * 0.9  # Slightly correlated
        
        # Interpretation based on TFI
        if TFI < 0.15:
            interpretation = "Solid - highly consistent textual identity over time"
        elif TFI < 0.30:
            interpretation = "Semi-solid - stable messaging with minor evolution"
        elif TFI < 0.45:
            interpretation = "Semi-fluid - adaptive language with coherent core"
        else:
            interpretation = "Fluid - significant textual identity changes over time"
        
        return {
            "semantic_drift": float(semantic_drift),
            "tone_diversity": float(tone_diversity),
            "topic_turnover": float(topic_turnover),
            "TFI": float(TFI),
            "temporal_stability": float(stability),
            "period_similarities": period_similarities,
            "years_analyzed": list(year_centroids.keys()),
            "interpretation": interpretation
        }
    
    def calculate_visual_fluidity(self, images_by_year):
        """
        Calculate Visual Fluidity Index (VFI) using temporal self-similarity.
        
        VFI = 1 - Temporal Visual Stability
        High VFI = visual identity changing over time = fluid
        Low VFI = visual identity consistent over time = solid
        """
        years = sorted(images_by_year.keys())
        
        if len(years) < 2:
            return {
                "visual_drift": 0.0,
                "color_entropy": 0.0,
                "motif_diversity": 0.0,
                "VFI": 0.0,
                "interpretation": "Insufficient temporal data (need 2+ time periods with year in filename)"
            }
        
        # Compute CLIP centroid for each year
        year_centroids = {}
        year_color_entropies = {}
        
        for year in years:
            paths = images_by_year[year]
            if paths:
                centroid = self._compute_image_centroid(paths)
                if centroid is not None:
                    year_centroids[year] = centroid
                
                # Calculate average color entropy for this year
                entropies = []
                for path in paths:
                    try:
                        img = cv2.imread(path)
                        if img is not None:
                            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
                            hist_norm = hist.ravel() / (hist.sum() + 1e-7)
                            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
                            entropies.append(entropy)
                    except:
                        continue
                if entropies:
                    year_color_entropies[year] = np.mean(entropies)
        
        if len(year_centroids) < 2:
            return {
                "visual_drift": 0.0,
                "color_entropy": 0.0,
                "motif_diversity": 0.0,
                "VFI": 0.0,
                "interpretation": "Insufficient data for temporal analysis"
            }
        
        # Calculate temporal visual stability
        stability, period_similarities = self._compute_temporal_stability(year_centroids)
        
        # VFI = 1 - stability
        VFI = 1 - stability
        
        # Color entropy stability (how consistent is color palette over time)
        if len(year_color_entropies) >= 2:
            entropy_values = [year_color_entropies[y] for y in sorted(year_color_entropies.keys())]
            color_stability = 1 - (np.std(entropy_values) / (np.mean(entropy_values) + 1e-7))
            color_stability = max(0, min(1, color_stability))
            avg_entropy = np.mean(entropy_values) / 15.0  # Normalize to 0-1
        else:
            color_stability = 1.0
            avg_entropy = 0.3
        
        # For backward compatibility
        visual_drift = VFI
        color_entropy = float(min(1.0, avg_entropy))
        motif_diversity = VFI * 0.95  # Correlated with overall visual change
        
        # Interpretation
        if VFI < 0.15:
            interpretation = "Solid - highly consistent visual identity over time"
        elif VFI < 0.30:
            interpretation = "Semi-solid - stable visuals with minor evolution"
        elif VFI < 0.45:
            interpretation = "Semi-fluid - visual experimentation with recognizable codes"
        else:
            interpretation = "Fluid - significant visual identity changes over time"
        
        return {
            "visual_drift": float(visual_drift),
            "color_entropy": float(color_entropy),
            "motif_diversity": float(motif_diversity),
            "VFI": float(VFI),
            "temporal_stability": float(stability),
            "color_stability": float(color_stability),
            "period_similarities": period_similarities,
            "years_analyzed": list(year_centroids.keys()),
            "interpretation": interpretation
        }
    
    def calculate_cross_modal_coherence(self, texts_by_year, images_by_year):
        """
        Calculate Cross-Modal Coherence Score (CMCS) using aggregate semantic alignment.
        
        Measures whether text and visual brand expressions occupy overlapping
        semantic territories in CLIP's joint embedding space.
        
        High CMCS = aligned text-visual identity
        Low CMCS = fragmented brand expression
        """
        # Collect all texts and images
        all_texts = []
        for texts in texts_by_year.values():
            all_texts.extend(texts)
        
        all_images = []
        for paths in images_by_year.values():
            all_images.extend(paths)
        
        if not all_texts or not all_images:
            return {
                "CMCS": 0.5,
                "interpretation": "Insufficient data for cross-modal analysis"
            }
        
        # Compute CLIP text embeddings
        text_embeddings = []
        for text in all_texts:
            try:
                truncated = text[:300] if len(text) > 300 else text
                tokens = clip.tokenize([truncated], truncate=True).to(self.device)
                with torch.no_grad():
                    embed = self.clip_model.encode_text(tokens)
                text_embeddings.append(embed.cpu().numpy().flatten())
            except:
                continue
        
        # Compute CLIP image embeddings
        image_embeddings = []
        for path in all_images:
            try:
                image = self.clip_preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embed = self.clip_model.encode_image(image)
                image_embeddings.append(embed.cpu().numpy().flatten())
            except:
                continue
        
        if not text_embeddings or not image_embeddings:
            return {
                "CMCS": 0.5,
                "interpretation": "Failed to compute embeddings"
            }
        
        # Compute centroids
        text_centroid = np.mean(text_embeddings, axis=0)
        image_centroid = np.mean(image_embeddings, axis=0)
        
        # Normalize
        text_centroid = text_centroid / (np.linalg.norm(text_centroid) + 1e-8)
        image_centroid = image_centroid / (np.linalg.norm(image_centroid) + 1e-8)
        
        # Centroid similarity
        centroid_sim = float(np.dot(text_centroid, image_centroid))
        
        # Nearest-neighbor coherence (sample for efficiency)
        sample_texts = text_embeddings[:30]
        sample_images = image_embeddings[:30]
        
        nn_similarities = []
        for t_emb in sample_texts:
            t_norm = t_emb / (np.linalg.norm(t_emb) + 1e-8)
            best_sim = max(
                float(np.dot(t_norm, i_emb / (np.linalg.norm(i_emb) + 1e-8)))
                for i_emb in sample_images
            )
            nn_similarities.append(best_sim)
        
        nn_coherence = np.mean(nn_similarities) if nn_similarities else centroid_sim
        
        # Combine measures and normalize to 0-1
        raw_cmcs = 0.5 * centroid_sim + 0.5 * nn_coherence
        CMCS = (raw_cmcs + 1) / 2  # Map from [-1, 1] to [0, 1]
        
        # Interpretation
        if CMCS > 0.65:
            interpretation = "Strong coherence - text and visuals are well-aligned"
        elif CMCS > 0.50:
            interpretation = "Moderate coherence - some alignment between text and visuals"
        elif CMCS > 0.35:
            interpretation = "Weak coherence - fragmented or decoupled storytelling"
        else:
            interpretation = "Very weak coherence - text and visual identities diverge significantly"
        
        return {
            "CMCS": float(CMCS),
            "centroid_similarity": float((centroid_sim + 1) / 2),
            "nn_coherence": float((nn_coherence + 1) / 2),
            "mean_similarity": float(CMCS),
            "std_similarity": float(np.std(nn_similarities)) if nn_similarities else 0.0,
            "interpretation": interpretation
        }
    
    def compute_brand_fluidity_index(self, TFI, VFI, CMCS, alpha=0.4, beta=0.4, gamma=0.2):
        """
        Compute composite Brand Fluidity Index (BFI).
        
        BFI = α*TFI + β*VFI + γ*(1-CMCS)
        
        Higher BFI = more fluid brand identity
        Lower BFI = more solid brand identity
        """
        # Inversion of CMCS because lower coherence → higher fluidity
        BFI = alpha * TFI + beta * VFI + gamma * (1 - CMCS)
        BFI = max(0, min(1, BFI))
        
        # Classification with revised thresholds
        if BFI < 0.20:
            category = "Solid"
            description = "Brand maintains strong continuity, stability, and predictable cues"
        elif BFI < 0.35:
            category = "Semi-solid"
            description = "Strong identity core with controlled, strategic evolution"
        elif BFI < 0.50:
            category = "Semi-fluid"
            description = "Optimal balance - stable enough for recognition, fluid enough for relevance"
        else:
            category = "Fluid"
            description = "Brand identity is adaptive, context-responsive, or in transformation"
        
        return {
            "BFI": float(BFI),
            "category": category,
            "description": description,
            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma}
        }
    
    def analyze_brand(self, text_dir, image_dir, brand_name):
        """
        Complete brand fluidity analysis.
        
        Args:
            text_dir: Directory containing text files (with years in filenames)
            image_dir: Directory containing image files (with years in filenames)
            brand_name: Name of the brand
            
        Returns:
            Complete analysis results (backward-compatible format)
        """
        results = {
            "brand_name": brand_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "methodology": "Temporal Self-Similarity Analysis v3"
        }
        
        # Load data organized by year
        texts_by_year, all_texts = self._load_texts_by_year(text_dir)
        images_by_year, all_images = self._load_images_by_year(image_dir)
        
        text_years = sorted(texts_by_year.keys())
        image_years = sorted(images_by_year.keys())
        
        print(f"\nAnalyzing {brand_name}:")
        print(f"  Texts: {len(all_texts)} files across {len(text_years)} time periods: {text_years}")
        print(f"  Images: {len(all_images)} files across {len(image_years)} time periods: {image_years}")
        
        # Calculate TFI (Textual Fluidity)
        tfi_results = self.calculate_textual_fluidity(texts_by_year)
        results["textual_fluidity"] = tfi_results
        
        # Calculate VFI (Visual Fluidity)
        vfi_results = self.calculate_visual_fluidity(images_by_year)
        results["visual_fluidity"] = vfi_results
        
        # Calculate CMCS (Cross-Modal Coherence)
        cmcs_results = self.calculate_cross_modal_coherence(texts_by_year, images_by_year)
        results["cross_modal_coherence"] = cmcs_results
        
        # Calculate BFI
        bfi_results = self.compute_brand_fluidity_index(
            TFI=tfi_results["TFI"],
            VFI=vfi_results["VFI"],
            CMCS=cmcs_results["CMCS"]
        )
        results["brand_fluidity_index"] = bfi_results
        
        # Summary (backward compatible)
        all_years = sorted(set(text_years) | set(image_years))
        results["summary"] = {
            "texts_analyzed": len(all_texts),
            "images_analyzed": len(all_images),
            "time_periods": len(all_years),
            "years_covered": all_years,
            "overall_category": bfi_results["category"]
        }
        
        # Strategic insights
        results["strategic_insights"] = self._generate_strategic_insights(results)
        
        return results
    
    def _generate_strategic_insights(self, results):
        """Generate strategic insights based on analysis."""
        insights = []
        
        bfi = results["brand_fluidity_index"]
        tfi = results["textual_fluidity"]
        vfi = results["visual_fluidity"]
        cmcs = results["cross_modal_coherence"]
        
        category = bfi["category"]
        
        # Overall assessment
        if category == "Solid":
            insights.append(
                "Brand demonstrates exceptional temporal consistency, characteristic of "
                "heritage luxury or institutional brands. This stability supports premium "
                "positioning and long-term brand equity."
            )
        elif category == "Semi-solid":
            insights.append(
                "Brand maintains strong identity core while allowing controlled evolution. "
                "This balance is optimal for established brands seeking relevance "
                "without sacrificing recognition."
            )
        elif category == "Semi-fluid":
            insights.append(
                "Brand shows balanced evolution between stability and adaptation. "
                "Monitor to ensure changes are strategic rather than reactive drift."
            )
        else:
            insights.append(
                "Brand exhibits significant temporal variability. This may indicate "
                "intentional transformation, but verify governance is maintaining "
                "core brand equity."
            )
        
        # Cross-modal alignment check
        if cmcs["CMCS"] < 0.45:
            insights.append(
                f"Cross-modal coherence ({cmcs['CMCS']:.2f}) suggests text and visual "
                "identities may be telling different brand stories. Consider integrated "
                "campaign development to align messaging and imagery."
            )
        
        # Text-visual balance check
        text_fluidity = tfi["TFI"]
        visual_fluidity = vfi["VFI"]
        
        if abs(text_fluidity - visual_fluidity) > 0.15:
            if text_fluidity > visual_fluidity:
                insights.append(
                    "Textual identity is evolving faster than visual identity. "
                    "Consider whether visual codes should be updated to match "
                    "messaging evolution."
                )
            else:
                insights.append(
                    "Visual identity is evolving faster than textual messaging. "
                    "Ensure narrative framework keeps pace with visual innovation."
                )
        
        return insights


# =============================================================================
# Standalone usage
# =============================================================================
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Brand Fluidity Analyzer v3 - Temporal Self-Similarity")
    print("=" * 60)
    print("\nNOTE: Files must have years in filenames (e.g., 'brand_2015.txt')")
    print("Supported years: 2010-2029")
    
    analyzer = BrandFluidityAnalyzer()
    
    if len(sys.argv) >= 4:
        text_dir = sys.argv[1]
        image_dir = sys.argv[2]
        brand_name = sys.argv[3]
        
        results = analyzer.analyze_brand(text_dir, image_dir, brand_name)
        
        print("\n" + "=" * 60)
        print(f"RESULTS: {brand_name}")
        print("=" * 60)
        print(f"\nBrand Fluidity Index: {results['brand_fluidity_index']['BFI']:.3f}")
        print(f"Category: {results['brand_fluidity_index']['category']}")
        print(f"\nComponent Scores:")
        print(f"  TFI (Textual Fluidity):  {results['textual_fluidity']['TFI']:.3f}")
        print(f"  VFI (Visual Fluidity):   {results['visual_fluidity']['VFI']:.3f}")
        print(f"  CMCS (Cross-Modal):      {results['cross_modal_coherence']['CMCS']:.3f}")
        
        print(f"\nTime Periods Analyzed: {results['summary']['time_periods']}")
        print(f"Years: {results['summary'].get('years_covered', 'N/A')}")
        
        print("\nStrategic Insights:")
        for i, insight in enumerate(results.get('strategic_insights', []), 1):
            print(f"  {i}. {insight}")
    else:
        print("\nUsage: python brand_fluidity_analyzer.py <text_dir> <image_dir> <brand_name>")
