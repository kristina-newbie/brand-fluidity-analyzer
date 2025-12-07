"""
Brand Fluidity Analyzer v4: Methodology-Aligned Implementation
==============================================================

This version implements Option C alignment between methodology and code:
1. CMCS temporal coherence component (NEW - genuine improvement)
2. Independent Jaccard-based topic turnover via LDA (NEW)
3. Independent clustering-based motif diversity (NEW)
4. Personality-based tone diversity (KEPT - theoretically superior to VADER std)
5. Temporal self-similarity framework (KEPT from v3)

Scientific References:
- Bauman, Z. (2000). Liquid Modernity. Polity Press.
- Aaker, J. L. (1997). Dimensions of brand personality. JMR, 34(3), 347-356.
- Reimers & Gurevych (2019). Sentence-BERT. EMNLP-IJCNLP.
- Radford et al. (2021). CLIP. ICML.
- Blei, Ng & Jordan (2003). Latent Dirichlet Allocation. JMLR, 3, 993-1022.
- Jaccard, P. (1912). Distribution of flora in alpine zone. New Phytologist, 11(2), 37-50.
- Shannon, C. E. (1948). Mathematical Theory of Communication. Bell Sys Tech J.
- MacQueen, J. (1967). K-means clustering. Berkeley Symposium.
"""

import os
# Fix tokenizer parallelism warning - must be set before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from gensim import corpora, models
from collections import defaultdict, Counter
import cv2
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class BrandFluidityAnalyzer:
    """
    Analyzes brand fluidity using temporal self-similarity methodology
    with methodology-aligned component calculations.
    """
    
    def __init__(self):
        """Initialize models for analysis."""
        print("Loading models...")
        
        # CLIP for visual embeddings (Radford et al., 2021)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Sentence-BERT for text embeddings (Reimers & Gurevych, 2019)
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # VADER for sentiment (Hutto & Gilbert, 2014)
        try:
            self.sia = SentimentIntensityAnalyzer()
        except:
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        
        # Load stopwords for LDA preprocessing
        try:
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('english'))
        except:
            import nltk
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('english'))
        
        # Brand personality lexicons (Aaker, 1997)
        self._init_personality_lexicons()
        
        print("Models loaded successfully!")
    
    def _init_personality_lexicons(self):
        """
        Initialize brand personality dimension lexicons.
        Based on Aaker, J. L. (1997). Dimensions of brand personality.
        Journal of Marketing Research, 34(3), 347-356.
        """
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
        """Extract year from filename. Supports: 'brand_2015_content.txt', '2015_instagram.jpg'"""
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
        """Compute semantic centroid for a collection of texts using SBERT."""
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
                continue
        
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0)
    
    def _compute_temporal_stability(self, year_centroids):
        """
        Compute temporal stability as mean cosine similarity between consecutive periods.
        Based on cosine similarity measure (Salton & McGill, 1983).
        
        Returns: (stability_score, list_of_period_similarities)
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
        """
        Compute brand personality vector based on Aaker (1997) dimensions.
        Returns 9-dimensional vector: 5 personality dimensions + 4 sentiment scores.
        """
        if not texts:
            return None
        
        combined_text = ' '.join(texts).lower()
        words = set(re.findall(r'\b[a-z]+\b', combined_text))
        
        scores = []
        for dimension in ['sincerity', 'excitement', 'competence', 'sophistication', 'ruggedness']:
            lexicon = self.personality_lexicons[dimension]
            matches = len(words.intersection(lexicon))
            score = (matches / max(len(words), 1)) * 1000  # Scale for numerical stability
            scores.append(score)
        
        # Add sentiment dimensions
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
        else:
            scores.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(scores)
    
    def _compute_jaccard_topic_turnover(self, texts_by_year, n_topics=None):
        """
        Compute topic turnover using Jaccard similarity between consecutive periods.
        
        Based on:
        - Blei, Ng & Jordan (2003). Latent Dirichlet Allocation. JMLR.
        - Jaccard, P. (1912). Distribution of flora in alpine zone. New Phytologist.
        
        Topic turnover measures how many dominant topics appear/disappear between periods:
        TT(t, t+1) = 1 - J(T_t, T_{t+1})
        
        Returns: (mean_turnover, list_of_period_turnovers, dominant_topics_by_year)
        """
        years = sorted(texts_by_year.keys())
        
        if len(years) < 2:
            return 0.0, [], {}
        
        # Collect all texts and preprocess
        all_texts = []
        text_year_map = []  # Track which year each text belongs to
        
        for year in years:
            for text in texts_by_year[year]:
                all_texts.append(text)
                text_year_map.append(year)
        
        if len(all_texts) < 5:
            return 0.0, [], {}
        
        # Simple preprocessing (avoid NLTK word_tokenize for stability)
        def preprocess(text):
            # Simple tokenization: lowercase, split on non-alphanumeric, filter
            tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
            return [t for t in tokens if t not in self.stopwords]
        
        try:
            tokenized_texts = [preprocess(text) for text in all_texts]
            
            # Filter out empty documents
            valid_indices = [i for i, tokens in enumerate(tokenized_texts) if len(tokens) > 0]
            if len(valid_indices) < 5:
                return 0.0, [], {}
            
            tokenized_texts = [tokenized_texts[i] for i in valid_indices]
            text_year_map = [text_year_map[i] for i in valid_indices]
            
            # Build dictionary and corpus
            dictionary = corpora.Dictionary(tokenized_texts)
            dictionary.filter_extremes(no_below=2, no_above=0.8, keep_n=1000)
            
            if len(dictionary) < 5:
                return 0.0, [], {}
            
            corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
            
            # Filter out empty documents from corpus
            valid_corpus_indices = [i for i, doc in enumerate(corpus) if len(doc) > 0]
            if len(valid_corpus_indices) < 5:
                return 0.0, [], {}
            
            corpus = [corpus[i] for i in valid_corpus_indices]
            text_year_map = [text_year_map[i] for i in valid_corpus_indices]
            
            # Determine number of topics: k = min(10, n/2) as per methodology
            if n_topics is None:
                n_topics = min(10, len(corpus) // 2)
                n_topics = max(3, n_topics)  # At least 3 topics
            
            # Train LDA model (Blei et al., 2003)
            lda = models.LdaModel(
                corpus,
                num_topics=n_topics,
                id2word=dictionary,
                passes=15,
                random_state=42,
                alpha='auto',
                eta='auto'
            )
            
            # Get dominant topics for each year (threshold > 0.15 as per methodology)
            DOMINANCE_THRESHOLD = 0.15
            dominant_topics_by_year = {}
            
            for year in years:
                year_indices = [i for i, y in enumerate(text_year_map) if y == year]
                year_topics = set()
                
                for idx in year_indices:
                    if idx < len(corpus):
                        doc_topics = lda.get_document_topics(corpus[idx], minimum_probability=0.0)
                        for topic_id, prob in doc_topics:
                            if prob > DOMINANCE_THRESHOLD:
                                year_topics.add(topic_id)
                
                dominant_topics_by_year[year] = year_topics
            
            # Compute Jaccard-based turnover between consecutive years
            turnovers = []
            for i in range(len(years) - 1):
                year1, year2 = years[i], years[i + 1]
                topics1 = dominant_topics_by_year.get(year1, set())
                topics2 = dominant_topics_by_year.get(year2, set())
                
                # Jaccard similarity: |A ∩ B| / |A ∪ B|
                if len(topics1) == 0 and len(topics2) == 0:
                    jaccard = 1.0  # Both empty = perfect similarity
                else:
                    intersection = len(topics1.intersection(topics2))
                    union = len(topics1.union(topics2))
                    jaccard = intersection / union if union > 0 else 0.0
                
                turnover = 1 - jaccard  # Turnover = 1 - similarity
                turnovers.append({
                    'period': f"{year1}-{year2}",
                    'turnover': turnover,
                    'jaccard_similarity': jaccard,
                    'topics_year1': len(topics1),
                    'topics_year2': len(topics2)
                })
            
            mean_turnover = np.mean([t['turnover'] for t in turnovers]) if turnovers else 0.0
            
            return float(mean_turnover), turnovers, dominant_topics_by_year
            
        except Exception as e:
            print(f"LDA topic turnover computation failed: {e}")
            return 0.0, [], {}
    
    def _compute_clustering_motif_diversity(self, image_paths, images_by_year):
        """
        Compute motif diversity using clustering-based approach.
        
        Based on:
        - MacQueen, J. (1967). K-means clustering. Berkeley Symposium.
        - Methodology: k = min(15, sqrt(n)) clusters
        
        MD = (uniqueness + evenness) / 2
        - Uniqueness: proportion of clusters utilized
        - Evenness: normalized entropy of cluster distribution
        
        Returns: (motif_diversity, cluster_stats)
        """
        if len(image_paths) < 5:
            return 0.5, {'error': 'insufficient_images'}
        
        # Compute CLIP embeddings for all images
        embeddings = []
        valid_paths = []
        
        for path in image_paths:
            try:
                image = self.clip_preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.clip_model.encode_image(image)
                embeddings.append(embedding.cpu().numpy().flatten())
                valid_paths.append(path)
            except Exception as e:
                continue
        
        if len(embeddings) < 5:
            return 0.5, {'error': 'failed_to_process_images'}
        
        try:
            embeddings = np.array(embeddings)
            
            # Determine number of clusters: k = min(15, sqrt(n))
            n_clusters = min(15, int(np.sqrt(len(embeddings))))
            n_clusters = max(3, min(n_clusters, len(embeddings) - 1))  # Can't have more clusters than samples
            
            # K-means clustering (MacQueen, 1967)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Compute cluster distribution
            cluster_counts = Counter(labels)
            
            # Uniqueness: proportion of clusters actually utilized
            clusters_used = len(cluster_counts)
            uniqueness = clusters_used / n_clusters
            
            # Evenness: normalized Shannon entropy of cluster distribution
            # Based on Shannon, C. E. (1948). Mathematical Theory of Communication.
            total = len(labels)
            probs = np.array([count / total for count in cluster_counts.values()])
            
            # Shannon entropy
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            
            # Maximum entropy for this number of clusters
            max_entropy = np.log2(clusters_used) if clusters_used > 1 else 1.0
            
            # Normalized evenness (0 = all in one cluster, 1 = perfectly even)
            evenness = entropy / max_entropy if max_entropy > 0 else 0.0
            
            # Motif diversity = average of uniqueness and evenness
            motif_diversity = (uniqueness + evenness) / 2
            
            cluster_stats = {
                'n_clusters_target': n_clusters,
                'n_clusters_used': clusters_used,
                'uniqueness': float(uniqueness),
                'evenness': float(evenness),
                'entropy': float(entropy),
                'max_entropy': float(max_entropy),
                'cluster_distribution': dict(cluster_counts)
            }
            
            return float(motif_diversity), cluster_stats
            
        except Exception as e:
            print(f"Motif diversity clustering failed: {e}")
            return 0.5, {'error': str(e)}
    
    def calculate_textual_fluidity(self, texts_by_year):
        """
        Calculate Textual Fluidity Index (TFI) using temporal self-similarity.
        
        Components:
        - Semantic Drift (SD): Temporal stability of SBERT embeddings
        - Tone Diversity (TD): Temporal stability of brand personality vectors (Aaker, 1997)
        - Topic Turnover (TT): Jaccard-based topic change via LDA (Blei et al., 2003)
        
        TFI = (SD + TD + TT) / 3
        """
        years = sorted(texts_by_year.keys())
        
        if len(years) < 2:
            return {
                "semantic_drift": 0.0,
                "tone_diversity": 0.0,
                "topic_turnover": 0.0,
                "TFI": 0.0,
                "interpretation": "Insufficient temporal data (need 2+ time periods)"
            }
        
        # 1. Semantic Drift (SD) - via temporal centroid stability
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
        
        semantic_stability, sd_similarities = self._compute_temporal_stability(year_centroids)
        semantic_drift = 1 - semantic_stability
        
        # 2. Tone Diversity (TD) - via personality vector stability (Aaker, 1997)
        year_personalities = {}
        for year in year_centroids.keys():
            p = self._compute_personality_vector(texts_by_year[year])
            if p is not None:
                year_personalities[year] = p
        
        if len(year_personalities) >= 2:
            personality_stability, _ = self._compute_temporal_stability(year_personalities)
            tone_diversity = 1 - personality_stability
        else:
            tone_diversity = semantic_drift  # Fallback
        
        # 3. Topic Turnover (TT) - via Jaccard-based LDA (Blei et al., 2003; Jaccard, 1912)
        topic_turnover, tt_details, dominant_topics = self._compute_jaccard_topic_turnover(texts_by_year)
        
        # Combine into TFI (equal weights)
        TFI = (semantic_drift + tone_diversity + topic_turnover) / 3
        TFI = max(0, min(1, TFI))  # Clamp to [0, 1]
        
        # Interpretation
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
            "temporal_stability": float(semantic_stability),
            "period_similarities": sd_similarities,
            "topic_turnover_details": tt_details,
            "years_analyzed": list(year_centroids.keys()),
            "interpretation": interpretation
        }
    
    def calculate_visual_fluidity(self, images_by_year):
        """
        Calculate Visual Fluidity Index (VFI) using temporal self-similarity.
        
        Components:
        - Visual Drift (VD): Temporal stability of CLIP embeddings (Radford et al., 2021)
        - Color Entropy (CE): Shannon entropy of color distributions (Shannon, 1948)
        - Motif Diversity (MD): Clustering-based visual diversity (MacQueen, 1967)
        
        VFI = (VD + CE_change + MD) / 3
        """
        years = sorted(images_by_year.keys())
        all_images = [p for paths in images_by_year.values() for p in paths]
        
        if len(years) < 2:
            return {
                "visual_drift": 0.0,
                "color_entropy": 0.0,
                "motif_diversity": 0.0,
                "VFI": 0.0,
                "interpretation": "Insufficient temporal data (need 2+ time periods)"
            }
        
        # 1. Visual Drift (VD) - via temporal CLIP centroid stability
        year_centroids = {}
        year_color_entropies = {}
        
        for year in years:
            paths = images_by_year[year]
            if paths:
                centroid = self._compute_image_centroid(paths)
                if centroid is not None:
                    year_centroids[year] = centroid
                
                # Calculate average color entropy for this year (Shannon, 1948)
                entropies = []
                for path in paths:
                    try:
                        img = cv2.imread(path)
                        if img is not None:
                            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                                              [0, 180, 0, 256, 0, 256])
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
        visual_stability, vd_similarities = self._compute_temporal_stability(year_centroids)
        visual_drift = 1 - visual_stability
        
        # 2. Color Entropy Change - measure temporal change in color complexity
        if len(year_color_entropies) >= 2:
            entropy_values = [year_color_entropies[y] for y in sorted(year_color_entropies.keys())]
            # Coefficient of variation as measure of color palette change
            color_cv = np.std(entropy_values) / (np.mean(entropy_values) + 1e-7)
            color_entropy_change = min(1.0, color_cv)  # Normalize
            avg_entropy = np.mean(entropy_values)
            # Also normalize average entropy to 0-1 scale: (raw - 3) / 5 per methodology
            avg_entropy_norm = max(0, min(1, (avg_entropy - 3.0) / 5.0))
        else:
            color_entropy_change = 0.0
            avg_entropy_norm = 0.3
        
        # 3. Motif Diversity (MD) - clustering-based (MacQueen, 1967)
        motif_diversity, cluster_stats = self._compute_clustering_motif_diversity(all_images, images_by_year)
        
        # Combine into VFI
        # Weight visual drift more heavily as it's the core temporal measure
        VFI = (visual_drift * 0.5 + color_entropy_change * 0.2 + motif_diversity * 0.3)
        VFI = max(0, min(1, VFI))
        
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
            "color_entropy": float(avg_entropy_norm),
            "color_entropy_change": float(color_entropy_change),
            "motif_diversity": float(motif_diversity),
            "VFI": float(VFI),
            "temporal_stability": float(visual_stability),
            "period_similarities": vd_similarities,
            "cluster_stats": cluster_stats,
            "years_analyzed": list(year_centroids.keys()),
            "interpretation": interpretation
        }
    
    def calculate_cross_modal_coherence(self, texts_by_year, images_by_year):
        """
        Calculate Cross-Modal Coherence Score (CMCS) using aggregate semantic alignment.
        
        CMCS integrates three complementary measures (Radford et al., 2021):
        1. Centroid Similarity (CS): Global text-image semantic alignment
        2. Nearest-Neighbor Coherence (NN): Best-match similarities
        3. Temporal Coherence (TC): Within-period alignment (NEW in v4)
        
        With temporal data: CMCS = 0.3*CS + 0.3*NN + 0.4*TC
        Without temporal: CMCS = 0.5*CS + 0.5*NN
        """
        try:
            # Collect all texts and images
            all_texts = [t for texts in texts_by_year.values() for t in texts]
            all_images = [p for paths in images_by_year.values() for p in paths]
            
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
            
            # 1. Centroid Similarity (CS) - global alignment
            text_centroid = np.mean(text_embeddings, axis=0)
            image_centroid = np.mean(image_embeddings, axis=0)
            
            text_centroid = text_centroid / (np.linalg.norm(text_centroid) + 1e-8)
            image_centroid = image_centroid / (np.linalg.norm(image_centroid) + 1e-8)
            
            centroid_sim = float(np.dot(text_centroid, image_centroid))
            
            # 2. Nearest-Neighbor Coherence (NN)
            sample_size = min(30, len(text_embeddings), len(image_embeddings))
            sample_texts = text_embeddings[:sample_size]
            sample_images = image_embeddings[:sample_size]
            
            # Text-to-image NN
            t2i_similarities = []
            for t_emb in sample_texts:
                t_norm = t_emb / (np.linalg.norm(t_emb) + 1e-8)
                best_sim = max(
                    float(np.dot(t_norm, i_emb / (np.linalg.norm(i_emb) + 1e-8)))
                    for i_emb in sample_images
                )
                t2i_similarities.append(best_sim)
            
            # Image-to-text NN
            i2t_similarities = []
            for i_emb in sample_images:
                i_norm = i_emb / (np.linalg.norm(i_emb) + 1e-8)
                best_sim = max(
                    float(np.dot(i_norm, t_emb / (np.linalg.norm(t_emb) + 1e-8)))
                    for t_emb in sample_texts
                )
                i2t_similarities.append(best_sim)
            
            nn_coherence = (np.mean(t2i_similarities) + np.mean(i2t_similarities)) / 2
            
            # 3. Temporal Coherence (TC) - NEW in v4
            # Compute within-period centroid similarity and average across periods
            common_years = set(texts_by_year.keys()) & set(images_by_year.keys())
            
            temporal_coherences = []
            if len(common_years) >= 2:
                for year in common_years:
                    year_texts = texts_by_year.get(year, [])
                    year_images = images_by_year.get(year, [])
                    
                    if not year_texts or not year_images:
                        continue
                    
                    # Compute year-specific centroids
                    year_text_embeds = []
                    for text in year_texts[:10]:  # Sample for efficiency
                        try:
                            truncated = text[:300] if len(text) > 300 else text
                            tokens = clip.tokenize([truncated], truncate=True).to(self.device)
                            with torch.no_grad():
                                embed = self.clip_model.encode_text(tokens)
                            year_text_embeds.append(embed.cpu().numpy().flatten())
                        except:
                            continue
                    
                    year_image_embeds = []
                    for path in year_images[:10]:  # Sample for efficiency
                        try:
                            image = self.clip_preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                embed = self.clip_model.encode_image(image)
                            year_image_embeds.append(embed.cpu().numpy().flatten())
                        except:
                            continue
                    
                    if year_text_embeds and year_image_embeds:
                        year_text_centroid = np.mean(year_text_embeds, axis=0)
                        year_image_centroid = np.mean(year_image_embeds, axis=0)
                        
                        year_text_centroid = year_text_centroid / (np.linalg.norm(year_text_centroid) + 1e-8)
                        year_image_centroid = year_image_centroid / (np.linalg.norm(year_image_centroid) + 1e-8)
                        
                        year_sim = float(np.dot(year_text_centroid, year_image_centroid))
                        temporal_coherences.append({'year': year, 'similarity': year_sim})
            
            # Combine measures based on temporal data availability
            if temporal_coherences:
                temporal_coherence = np.mean([tc['similarity'] for tc in temporal_coherences])
                # With temporal data: 0.3*CS + 0.3*NN + 0.4*TC (per methodology)
                raw_cmcs = 0.3 * centroid_sim + 0.3 * nn_coherence + 0.4 * temporal_coherence
            else:
                temporal_coherence = None
                # Without temporal data: 0.5*CS + 0.5*NN
                raw_cmcs = 0.5 * centroid_sim + 0.5 * nn_coherence
            
            # Normalize to [0, 1] - CLIP similarities typically in [0.15, 0.40] range
            # Use empirical scaling rather than raw [-1, 1] mapping
            CMCS = max(0, min(1, (raw_cmcs - 0.10) / 0.35))  # Maps ~[0.10, 0.45] to [0, 1]
            
            # Interpretation
            if CMCS > 0.65:
                interpretation = "Strong coherence - text and visuals are well-aligned"
            elif CMCS > 0.45:
                interpretation = "Moderate coherence - partial alignment between text and visuals"
            elif CMCS > 0.30:
                interpretation = "Weak coherence - fragmented or decoupled storytelling"
            else:
                interpretation = "Very weak coherence - text and visual identities diverge significantly"
            
            result = {
                "CMCS": float(CMCS),
                "centroid_similarity": float(centroid_sim),
                "nn_coherence": float(nn_coherence),
                "raw_cmcs": float(raw_cmcs),
                "interpretation": interpretation
            }
            
            if temporal_coherence is not None:
                result["temporal_coherence"] = float(temporal_coherence)
                result["temporal_coherence_by_year"] = temporal_coherences
            
            return result
            
        except Exception as e:
            print(f"CMCS calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "CMCS": 0.5,
                "interpretation": f"Error in CMCS calculation: {str(e)}"
            }
    
    def compute_brand_fluidity_index(self, TFI, VFI, CMCS, alpha=0.4, beta=0.4, gamma=0.2):
        """
        Compute composite Brand Fluidity Index (BFI).
        
        BFI = α*TFI + β*VFI + γ*(1-CMCS)
        
        Default weights: α=0.4, β=0.4, γ=0.2
        CMCS inverted because lower coherence → higher fluidity
        """
        BFI = alpha * TFI + beta * VFI + gamma * (1 - CMCS)
        BFI = max(0, min(1, BFI))
        
        # Classification
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
            Complete analysis results
        """
        results = {
            "brand_name": brand_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "methodology": "Temporal Self-Similarity Analysis v4 (Methodology-Aligned)"
        }
        
        # Load data organized by year
        texts_by_year, all_texts = self._load_texts_by_year(text_dir)
        images_by_year, all_images = self._load_images_by_year(image_dir)
        
        text_years = sorted(texts_by_year.keys())
        image_years = sorted(images_by_year.keys())
        
        print(f"\nAnalyzing {brand_name}:")
        print(f"  Texts: {len(all_texts)} files across {len(text_years)} time periods: {text_years}")
        print(f"  Images: {len(all_images)} files across {len(image_years)} time periods: {image_years}")
        
        # Calculate TFI
        tfi_results = self.calculate_textual_fluidity(texts_by_year)
        results["textual_fluidity"] = tfi_results
        
        # Calculate VFI
        vfi_results = self.calculate_visual_fluidity(images_by_year)
        results["visual_fluidity"] = vfi_results
        
        # Calculate CMCS
        cmcs_results = self.calculate_cross_modal_coherence(texts_by_year, images_by_year)
        results["cross_modal_coherence"] = cmcs_results
        
        # Calculate BFI
        bfi_results = self.compute_brand_fluidity_index(
            TFI=tfi_results["TFI"],
            VFI=vfi_results["VFI"],
            CMCS=cmcs_results["CMCS"]
        )
        results["brand_fluidity_index"] = bfi_results
        
        # Summary
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
                "positioning and long-term brand equity (Keller, 1993)."
            )
        elif category == "Semi-solid":
            insights.append(
                "Brand maintains strong identity core while allowing controlled evolution. "
                "This balance is optimal for established brands seeking relevance "
                "without sacrificing recognition (Erdem & Swait, 1998)."
            )
        elif category == "Semi-fluid":
            insights.append(
                "Brand shows balanced evolution between stability and adaptation. "
                "Monitor to ensure changes are strategic rather than reactive drift."
            )
        else:
            insights.append(
                "Brand exhibits significant temporal variability, consistent with Bauman's (2000) "
                "concept of liquid modernity. Verify governance is maintaining core brand equity."
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
                    "Consider whether visual codes should be updated to match messaging evolution."
                )
            else:
                insights.append(
                    "Visual identity is evolving faster than textual messaging. "
                    "Ensure narrative framework keeps pace with visual innovation."
                )
        
        # Topic turnover insight
        if "topic_turnover" in tfi and tfi["topic_turnover"] > 0.4:
            insights.append(
                f"High topic turnover ({tfi['topic_turnover']:.2f}) indicates significant "
                "thematic evolution. This may reflect intentional repositioning or category expansion."
            )
        
        return insights


# =============================================================================
# Standalone usage
# =============================================================================
if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Brand Fluidity Analyzer v4 - Methodology-Aligned Implementation")
    print("=" * 70)
    print("\nNOTE: Files must have years in filenames (e.g., 'brand_2015.txt')")
    
    analyzer = BrandFluidityAnalyzer()
    
    if len(sys.argv) >= 4:
        text_dir = sys.argv[1]
        image_dir = sys.argv[2]
        brand_name = sys.argv[3]
        
        results = analyzer.analyze_brand(text_dir, image_dir, brand_name)
        
        print("\n" + "=" * 70)
        print(f"RESULTS: {brand_name}")
        print("=" * 70)
        print(f"\nBrand Fluidity Index: {results['brand_fluidity_index']['BFI']:.3f}")
        print(f"Category: {results['brand_fluidity_index']['category']}")
        print(f"\nComponent Scores:")
        print(f"  TFI (Textual Fluidity):  {results['textual_fluidity']['TFI']:.3f}")
        print(f"    - Semantic Drift:      {results['textual_fluidity']['semantic_drift']:.3f}")
        print(f"    - Tone Diversity:      {results['textual_fluidity']['tone_diversity']:.3f}")
        print(f"    - Topic Turnover:      {results['textual_fluidity']['topic_turnover']:.3f}")
        print(f"  VFI (Visual Fluidity):   {results['visual_fluidity']['VFI']:.3f}")
        print(f"    - Visual Drift:        {results['visual_fluidity']['visual_drift']:.3f}")
        print(f"    - Color Entropy:       {results['visual_fluidity']['color_entropy']:.3f}")
        print(f"    - Motif Diversity:     {results['visual_fluidity']['motif_diversity']:.3f}")
        print(f"  CMCS (Cross-Modal):      {results['cross_modal_coherence']['CMCS']:.3f}")
        if 'temporal_coherence' in results['cross_modal_coherence']:
            print(f"    - Temporal Coherence:  {results['cross_modal_coherence']['temporal_coherence']:.3f}")
        
        print(f"\nTime Periods: {results['summary']['time_periods']}")
        print(f"Years: {results['summary'].get('years_covered', 'N/A')}")
        
        print("\nStrategic Insights:")
        for i, insight in enumerate(results.get('strategic_insights', []), 1):
            print(f"  {i}. {insight}")
    else:
        print("\nUsage: python brand_fluidity_analyzer_v4.py <text_dir> <image_dir> <brand_name>")
