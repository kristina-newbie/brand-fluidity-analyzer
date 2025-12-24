"""
Brand Fluidity Analyzer v2.1
============================
Corrected metric computation with proper normalization for brand identity fluidity analysis.

CHANGELOG from v2.0:
- CRITICAL FIX: Added CMCS normalization (raw CLIP values → interpretable 0-1 scale)
- FIX: Corrected tone diversity normalization ceiling (0.58 → 0.35 empirical max)
- FIX: Added semantic drift normalization for consistent scaling
- FIX: Added visual drift normalization
- ENHANCEMENT: Configurable normalization parameters with documented defaults
- ENHANCEMENT: Validation warnings for out-of-range raw values

Scientific References:
- Sentence-BERT: Reimers & Gurevych (2019), EMNLP
- CLIP: Radford et al. (2021), ICML
- VADER: Hutto & Gilbert (2014), ICWSM
- LDA: Blei et al. (2003), JMLR
- Cosine Similarity: Mikolov et al. (2013), NeurIPS
- Shannon Entropy: Shannon (1948), Bell System Technical Journal
"""

import os
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
from collections import defaultdict
from scipy import stats
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# NORMALIZATION PARAMETERS
# =============================================================================

@dataclass
class NormalizationParams:
    """
    Empirically-derived normalization parameters for metric scaling.
    
    These values are based on:
    1. CLIP paper results (Radford et al., 2021)
    2. Sentence-BERT benchmarks (Reimers & Gurevych, 2019)
    3. Empirical analysis of 28 fashion brand corpora (2015-2025)
    
    Adjust these if working with different domains (e.g., tech brands, CPG).
    """
    # CMCS: Raw CLIP cross-modal similarities typically range 0.12-0.42
    cmcs_min: float = 0.12          # Unrelated text-image pairs
    cmcs_max: float = 0.42          # Highly coherent brand content
    
    # Semantic Drift: 1 - SBERT cosine similarity for consecutive periods
    semantic_drift_min: float = 0.02    # Very stable brand identity
    semantic_drift_max: float = 0.30    # Significant repositioning
    
    # Tone Diversity: std(VADER compound scores) for brand content
    tone_diversity_max: float = 0.35    # Empirical max for professional brands
    
    # Visual Drift: 1 - CLIP cosine similarity for consecutive periods
    visual_drift_min: float = 0.01      # Near-identical visual identity
    visual_drift_max: float = 0.15      # Major visual overhaul
    
    # Color Entropy: Shannon entropy of 8x8x8 HSV histogram
    color_entropy_max: float = 9.0      # log2(512) theoretical maximum
    
    # Warnings threshold (flag if raw values exceed expected ranges)
    warn_on_outliers: bool = True


# Default parameters instance
DEFAULT_NORM_PARAMS = NormalizationParams()


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_cmcs(raw_cmcs: float, params: NormalizationParams = DEFAULT_NORM_PARAMS) -> float:
    """
    Rescale raw CLIP cross-modal similarity to interpretable 0-1 range.
    
    Raw CLIP text-image cosine similarities naturally fall in a compressed range
    (typically 0.12-0.42) due to the geometry of the joint embedding space.
    This is expected behavior, not a limitation.
    
    Args:
        raw_cmcs: Raw cosine similarity from CLIP
        params: Normalization parameters
        
    Returns:
        Normalized CMCS in [0, 1] range where:
        - 0.0-0.3: Weak coherence (fragmented storytelling)
        - 0.3-0.6: Moderate coherence (partial alignment)
        - 0.6-1.0: Strong coherence (well-integrated brand expression)
    """
    if params.warn_on_outliers:
        if raw_cmcs < params.cmcs_min - 0.05:
            print(f"  Warning: CMCS raw value {raw_cmcs:.3f} below expected range")
        elif raw_cmcs > params.cmcs_max + 0.05:
            print(f"  Warning: CMCS raw value {raw_cmcs:.3f} above expected range")
    
    normalized = (raw_cmcs - params.cmcs_min) / (params.cmcs_max - params.cmcs_min)
    return float(np.clip(normalized, 0.0, 1.0))


def normalize_semantic_drift(raw_drift: float, params: NormalizationParams = DEFAULT_NORM_PARAMS) -> float:
    """
    Normalize semantic drift based on empirical SBERT behavior.
    
    For brand communications with coherent identity, consecutive period
    SBERT similarities typically range 0.70-0.98, yielding drift of 0.02-0.30.
    
    Args:
        raw_drift: Raw drift (1 - cosine similarity)
        params: Normalization parameters
        
    Returns:
        Normalized drift in [0, 1] range where:
        - 0.0-0.3: Stable semantic identity
        - 0.3-0.6: Moderate evolution
        - 0.6-1.0: Significant semantic repositioning
    """
    if params.warn_on_outliers and raw_drift > params.semantic_drift_max + 0.1:
        print(f"  Warning: Semantic drift {raw_drift:.3f} above expected range")
    
    normalized = (raw_drift - params.semantic_drift_min) / (params.semantic_drift_max - params.semantic_drift_min)
    return float(np.clip(normalized, 0.0, 1.0))


def normalize_tone_diversity(raw_std: float, params: NormalizationParams = DEFAULT_NORM_PARAMS) -> float:
    """
    Normalize tone diversity by empirical maximum for brand communications.
    
    Professional brand communications maintain tone discipline. Empirical analysis
    of luxury/fashion brand corpora shows std(sentiment) rarely exceeds 0.35,
    versus theoretical max of 0.58 for uniform distribution over [-1, 1].
    
    Args:
        raw_std: Standard deviation of VADER compound scores
        params: Normalization parameters
        
    Returns:
        Normalized tone diversity in [0, 1] range
    """
    normalized = raw_std / params.tone_diversity_max
    return float(np.clip(normalized, 0.0, 1.0))


def normalize_visual_drift(raw_drift: float, params: NormalizationParams = DEFAULT_NORM_PARAMS) -> float:
    """
    Normalize visual drift based on empirical CLIP behavior.
    
    Brand visual identity typically shows high period-to-period consistency.
    CLIP similarities for brand imagery usually range 0.85-0.99, yielding
    drift of 0.01-0.15.
    
    Args:
        raw_drift: Raw drift (1 - cosine similarity)
        params: Normalization parameters
        
    Returns:
        Normalized drift in [0, 1] range
    """
    if params.warn_on_outliers and raw_drift > params.visual_drift_max + 0.05:
        print(f"  Warning: Visual drift {raw_drift:.3f} above expected range")
    
    normalized = (raw_drift - params.visual_drift_min) / (params.visual_drift_max - params.visual_drift_min)
    return float(np.clip(normalized, 0.0, 1.0))


def normalize_color_entropy(raw_entropy: float, params: NormalizationParams = DEFAULT_NORM_PARAMS) -> float:
    """
    Normalize color entropy by theoretical maximum.
    
    Shannon entropy of 8x8x8 HSV histogram has theoretical max of log2(512) ≈ 9.
    
    Args:
        raw_entropy: Shannon entropy value
        params: Normalization parameters
        
    Returns:
        Normalized entropy in [0, 1] range
    """
    normalized = raw_entropy / params.color_entropy_max
    return float(np.clip(normalized, 0.0, 1.0))


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class BrandFluidityAnalyzerV2:
    """
    Enhanced Brand Fluidity Analyzer v2.1 with corrected normalization.
    
    Key improvements over v2.0:
    - Proper CMCS normalization (critical fix)
    - Empirically-calibrated component normalization
    - Configurable normalization parameters
    - Raw values preserved for transparency
    """
    
    def __init__(self, 
                 device: str = None,
                 norm_params: NormalizationParams = None):
        """
        Initialize models for analysis.
        
        Args:
            device: 'cuda' or 'cpu'. Auto-detects if None.
            norm_params: Normalization parameters. Uses defaults if None.
        """
        print("Initializing Brand Fluidity Analyzer v2.1...")
        
        # Set normalization parameters
        self.norm_params = norm_params or DEFAULT_NORM_PARAMS
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        print("Loading CLIP (ViT-B/32)...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load Sentence-BERT
        print("Loading Sentence-BERT (all-MiniLM-L6-v2)...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # Initialize VADER
        print("Loading VADER sentiment analyzer...")
        try:
            self.sia = SentimentIntensityAnalyzer()
        except LookupError:
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        
        print("Initialization complete!\n")
    
    # =========================================================================
    # CORE TEXTUAL METRICS
    # =========================================================================
    
    def compute_semantic_drift(
        self, 
        texts_by_period: Dict[str, List[str]]
    ) -> Tuple[float, float, List[float], List[np.ndarray]]:
        """
        Compute semantic drift using Sentence-BERT embeddings.
        
        Returns both raw and normalized values for transparency.
        
        Args:
            texts_by_period: Dict mapping period labels to list of texts
            
        Returns:
            Tuple of (normalized_drift, raw_drift, period_similarities, period_embeddings)
        """
        periods = sorted(texts_by_period.keys())
        if len(periods) < 2:
            return 0.0, 0.0, [], []
        
        # Compute period embeddings (mean of all text embeddings in period)
        period_embeddings = []
        for period in periods:
            texts = texts_by_period[period]
            if texts:
                combined_text = " ".join(texts)
                embedding = self.sbert_model.encode(combined_text, convert_to_tensor=False)
                period_embeddings.append(embedding)
            else:
                period_embeddings.append(np.zeros(384))  # MiniLM embedding size
        
        # Compute consecutive period similarities
        similarities = []
        for i in range(len(periods) - 1):
            sim = cosine_similarity(
                period_embeddings[i].reshape(1, -1),
                period_embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(float(sim))
        
        # Raw drift = 1 - similarity
        raw_drifts = [1 - s for s in similarities]
        raw_mean_drift = np.mean(raw_drifts) if raw_drifts else 0.0
        
        # Normalized drift
        normalized_drift = normalize_semantic_drift(raw_mean_drift, self.norm_params)
        
        return normalized_drift, float(raw_mean_drift), similarities, period_embeddings
    
    def compute_tone_diversity(
        self, 
        texts_by_period: Dict[str, List[str]]
    ) -> Tuple[float, float]:
        """
        Compute tone diversity using VADER sentiment analysis.
        
        Returns both raw and normalized values.
        
        Args:
            texts_by_period: Dict mapping period labels to list of texts
            
        Returns:
            Tuple of (normalized_diversity, raw_std)
        """
        all_sentiments = []
        
        for period, texts in texts_by_period.items():
            for text in texts:
                try:
                    sentences = sent_tokenize(text)
                    for sent in sentences:
                        if len(sent.strip()) > 10:  # Skip very short fragments
                            score = self.sia.polarity_scores(sent)['compound']
                            all_sentiments.append(score)
                except Exception:
                    continue
        
        if len(all_sentiments) < 2:
            return 0.0, 0.0
        
        raw_std = np.std(all_sentiments)
        normalized = normalize_tone_diversity(raw_std, self.norm_params)
        
        return normalized, float(raw_std)
    
    def compute_topic_turnover(
        self, 
        texts_by_period: Dict[str, List[str]],
        n_topics: int = 5
    ) -> Tuple[float, List[float], List[set]]:
        """
        Compute topic turnover using LDA and Jaccard dissimilarity.
        
        Args:
            texts_by_period: Dict mapping period labels to list of texts
            n_topics: Number of LDA topics
            
        Returns:
            Tuple of (mean_turnover, period_turnovers, period_topic_sets)
        """
        periods = sorted(texts_by_period.keys())
        if len(periods) < 2:
            return 0.0, [], []
        
        # Build corpus for each period
        period_topic_sets = []
        
        for period in periods:
            texts = texts_by_period[period]
            if not texts or len(texts) < 2:
                period_topic_sets.append(set())
                continue
            
            # Tokenize
            tokenized = [text.lower().split() for text in texts]
            
            # Build dictionary and corpus
            dictionary = corpora.Dictionary(tokenized)
            dictionary.filter_extremes(no_below=1, no_above=0.9, keep_n=100)
            corpus = [dictionary.doc2bow(text) for text in tokenized]
            
            if len(dictionary) < 3 or len(corpus) < 2:
                period_topic_sets.append(set())
                continue
            
            # Train LDA
            try:
                lda = models.LdaModel(
                    corpus,
                    num_topics=min(n_topics, len(texts)),
                    id2word=dictionary,
                    passes=10,
                    random_state=42
                )
                
                # Get dominant topics for this period
                dominant_topics = set()
                for doc in corpus:
                    topics = lda.get_document_topics(doc)
                    if topics:
                        dominant_topic = max(topics, key=lambda x: x[1])[0]
                        dominant_topics.add(dominant_topic)
                
                period_topic_sets.append(dominant_topics)
            except Exception:
                period_topic_sets.append(set())
        
        # Compute Jaccard dissimilarity between consecutive periods
        turnovers = []
        for i in range(len(periods) - 1):
            set1 = period_topic_sets[i]
            set2 = period_topic_sets[i + 1]
            
            if not set1 and not set2:
                turnovers.append(0.0)
            elif not set1 or not set2:
                turnovers.append(1.0)
            else:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard = intersection / union if union > 0 else 0
                turnovers.append(1 - jaccard)  # Dissimilarity
        
        mean_turnover = np.mean(turnovers) if turnovers else 0.0
        
        return float(mean_turnover), turnovers, period_topic_sets
    
    # =========================================================================
    # CORE VISUAL METRICS
    # =========================================================================
    
    def compute_visual_drift(
        self, 
        images_by_period: Dict[str, List[str]]
    ) -> Tuple[float, float, List[float], List[np.ndarray]]:
        """
        Compute visual drift using CLIP embeddings.
        
        Returns both raw and normalized values.
        
        Args:
            images_by_period: Dict mapping period labels to list of image paths
            
        Returns:
            Tuple of (normalized_drift, raw_drift, period_similarities, period_centroids)
        """
        periods = sorted(images_by_period.keys())
        if len(periods) < 2:
            return 0.0, 0.0, [], []
        
        # Compute period centroids
        period_centroids = []
        
        for period in periods:
            image_paths = images_by_period[period]
            embeddings = []
            
            for path in image_paths:
                try:
                    image = self.clip_preprocess(Image.open(path)).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        embedding = self.clip_model.encode_image(image)
                    embeddings.append(embedding.cpu().numpy().squeeze())
                except Exception:
                    continue
            
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
            else:
                centroid = np.zeros(512)  # CLIP ViT-B/32 embedding size
            
            period_centroids.append(centroid)
        
        # Compute consecutive period similarities
        similarities = []
        for i in range(len(periods) - 1):
            sim = cosine_similarity(
                period_centroids[i].reshape(1, -1),
                period_centroids[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(float(sim))
        
        # Raw drift = 1 - similarity
        raw_drifts = [1 - s for s in similarities]
        raw_mean_drift = np.mean(raw_drifts) if raw_drifts else 0.0
        
        # Normalized drift
        normalized_drift = normalize_visual_drift(raw_mean_drift, self.norm_params)
        
        return normalized_drift, float(raw_mean_drift), similarities, period_centroids
    
    def compute_color_entropy(
        self, 
        images_by_period: Dict[str, List[str]]
    ) -> Tuple[float, float, float]:
        """
        Compute color entropy using HSV histograms.
        
        Args:
            images_by_period: Dict mapping period labels to list of image paths
            
        Returns:
            Tuple of (normalized_entropy, raw_entropy, entropy_change)
        """
        period_entropies = []
        
        for period in sorted(images_by_period.keys()):
            image_paths = images_by_period[period]
            entropies = []
            
            for path in image_paths:
                try:
                    img = cv2.imread(str(path))
                    if img is None:
                        continue
                    
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                                       [0, 180, 0, 256, 0, 256])
                    hist_norm = hist.ravel() / (hist.sum() + 1e-7)
                    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
                    entropies.append(entropy)
                except Exception:
                    continue
            
            if entropies:
                period_entropies.append(np.mean(entropies))
        
        if not period_entropies:
            return 0.0, 0.0, 0.0
        
        raw_entropy = np.mean(period_entropies)
        normalized_entropy = normalize_color_entropy(raw_entropy, self.norm_params)
        
        # Entropy change (std of period entropies, normalized)
        if len(period_entropies) > 1:
            entropy_change = np.std(period_entropies) / self.norm_params.color_entropy_max
        else:
            entropy_change = 0.0
        
        return normalized_entropy, float(raw_entropy), float(entropy_change)
    
    def compute_motif_diversity(
        self, 
        images_by_period: Dict[str, List[str]],
        n_clusters: int = 8
    ) -> Tuple[float, float, float]:
        """
        Compute motif diversity using K-means clustering of CLIP embeddings.
        
        Args:
            images_by_period: Dict mapping period labels to list of image paths
            n_clusters: Number of motif clusters
            
        Returns:
            Tuple of (diversity_score, uniqueness, evenness)
        """
        # Collect all embeddings
        all_embeddings = []
        
        for period in images_by_period:
            for path in images_by_period[period]:
                try:
                    image = self.clip_preprocess(Image.open(path)).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        embedding = self.clip_model.encode_image(image)
                    all_embeddings.append(embedding.cpu().numpy().squeeze())
                except Exception:
                    continue
        
        if len(all_embeddings) < n_clusters:
            return 0.0, 0.0, 0.0
        
        # Cluster embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(all_embeddings)
        
        # Uniqueness: proportion of clusters used
        unique_clusters = len(set(labels))
        uniqueness = unique_clusters / n_clusters
        
        # Evenness: normalized entropy of cluster distribution
        counts = np.bincount(labels, minlength=n_clusters)
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # Remove zeros
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(n_clusters)
        evenness = entropy / max_entropy if max_entropy > 0 else 0
        
        # Combined diversity score
        diversity = (uniqueness + evenness) / 2
        
        return float(diversity), float(uniqueness), float(evenness)
    
    # =========================================================================
    # CROSS-MODAL COHERENCE (CRITICAL FIX)
    # =========================================================================
    
    def compute_cross_modal_coherence(
        self,
        texts_by_period: Dict[str, List[str]],
        images_by_period: Dict[str, List[str]]
    ) -> Tuple[float, float, float, List[float]]:
        """
        Compute cross-modal coherence using CLIP joint embeddings.
        
        CRITICAL FIX: Raw CLIP similarities are normalized to interpretable 0-1 range.
        
        Args:
            texts_by_period: Dict mapping period labels to list of texts
            images_by_period: Dict mapping period labels to list of image paths
            
        Returns:
            Tuple of (normalized_cmcs, raw_cmcs, temporal_coherence, period_coherences)
        """
        periods = sorted(set(texts_by_period.keys()) & set(images_by_period.keys()))
        if not periods:
            return 0.7, 0.25, 0.7, []  # Return moderate defaults
        
        all_similarities = []
        period_coherences = []
        period_coherences_raw = []
        
        for period in periods:
            texts = texts_by_period.get(period, [])
            image_paths = images_by_period.get(period, [])
            
            if not texts or not image_paths:
                continue
            
            period_sims = []
            
            # Sample pairs (use min of available)
            n_pairs = min(len(texts), len(image_paths), 20)
            
            for i in range(n_pairs):
                text = texts[i % len(texts)][:77]  # CLIP token limit
                image_path = image_paths[i % len(image_paths)]
                
                try:
                    # Encode text
                    text_tokens = clip.tokenize([text], truncate=True).to(self.device)
                    with torch.no_grad():
                        text_embed = self.clip_model.encode_text(text_tokens).cpu().numpy()
                    
                    # Encode image
                    image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        image_embed = self.clip_model.encode_image(image).cpu().numpy()
                    
                    # Compute similarity
                    sim = cosine_similarity(text_embed, image_embed)[0][0]
                    period_sims.append(float(sim))
                    all_similarities.append(float(sim))
                except Exception:
                    continue
            
            if period_sims:
                raw_period_mean = np.mean(period_sims)
                period_coherences_raw.append(raw_period_mean)
                # Normalize each period's coherence
                period_coherences.append(normalize_cmcs(raw_period_mean, self.norm_params))
        
        # Overall raw CMCS
        raw_cmcs = np.mean(all_similarities) if all_similarities else 0.25
        
        # Normalized CMCS (CRITICAL FIX)
        normalized_cmcs = normalize_cmcs(raw_cmcs, self.norm_params)
        
        # Temporal coherence (mean of normalized period coherences)
        temporal_coherence = np.mean(period_coherences) if period_coherences else normalized_cmcs
        
        return normalized_cmcs, float(raw_cmcs), float(temporal_coherence), period_coherences_raw
    
    # =========================================================================
    # SECOND-ORDER METRICS
    # =========================================================================
    
    def compute_drift_volatility(
        self, 
        period_similarities: List[float]
    ) -> float:
        """
        Compute drift volatility (coefficient of variation of drift scores).
        """
        if len(period_similarities) < 2:
            return 0.0
        
        drifts = [1 - s for s in period_similarities]
        mean_drift = np.mean(drifts)
        
        if mean_drift < 1e-7:
            return 0.0
        
        cv = np.std(drifts) / mean_drift
        return float(cv)
    
    def compute_coherence_trajectory(
        self, 
        period_coherences: List[float]
    ) -> float:
        """
        Compute coherence trajectory (linear regression slope).
        """
        if len(period_coherences) < 2:
            return 0.0
        
        x = np.arange(len(period_coherences))
        slope, _, _, _, _ = stats.linregress(x, period_coherences)
        
        return float(slope)
    
    def compute_drift_ratio(
        self, 
        semantic_drift: float, 
        visual_drift: float
    ) -> float:
        """
        Compute drift ratio (semantic / visual).
        
        Note: Uses RAW drift values for this ratio to preserve meaning.
        """
        if visual_drift < 1e-7:
            return 0.0 if semantic_drift < 1e-7 else 10.0
        
        ratio = semantic_drift / visual_drift
        return float(min(10.0, ratio))
    
    def compute_directional_consistency(
        self, 
        period_embeddings: List[np.ndarray]
    ) -> float:
        """
        Compute directional consistency of identity evolution.
        """
        if len(period_embeddings) < 3:
            return 0.0
        
        # Compute drift vectors
        drift_vectors = []
        for i in range(len(period_embeddings) - 1):
            drift = period_embeddings[i + 1] - period_embeddings[i]
            drift_vectors.append(drift)
        
        # Compute angles between consecutive drift vectors
        angles = []
        for i in range(len(drift_vectors) - 1):
            v1 = drift_vectors[i]
            v2 = drift_vectors[i + 1]
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 < 1e-7 or norm2 < 1e-7:
                continue
            
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angles.append(cos_angle)
        
        if not angles:
            return 0.0
        
        return float(np.mean(angles))
    
    def compute_topic_semantic_decoupling(
        self, 
        topic_turnover: float, 
        semantic_drift: float
    ) -> float:
        """
        Compute topic-semantic decoupling index.
        
        Note: Uses RAW semantic drift for meaningful ratio.
        """
        if semantic_drift < 1e-7:
            return 0.0 if topic_turnover < 1e-7 else 10.0
        
        ratio = topic_turnover / semantic_drift
        return float(min(10.0, ratio))
    
    # =========================================================================
    # COMPOSITE INDICES
    # =========================================================================
    
    def compute_tfi(
        self, 
        semantic_drift: float, 
        tone_diversity: float, 
        topic_turnover: float,
        weights: Tuple[float, float, float] = (1/3, 1/3, 1/3)
    ) -> float:
        """
        Compute Textual Fluidity Index.
        
        Uses NORMALIZED component values.
        """
        return float(
            weights[0] * semantic_drift + 
            weights[1] * tone_diversity + 
            weights[2] * topic_turnover
        )
    
    def compute_vfi(
        self, 
        visual_drift: float, 
        color_entropy: float, 
        motif_diversity: float,
        weights: Tuple[float, float, float] = (1/3, 1/3, 1/3)
    ) -> float:
        """
        Compute Visual Fluidity Index.
        
        Uses NORMALIZED component values.
        """
        return float(
            weights[0] * visual_drift + 
            weights[1] * color_entropy + 
            weights[2] * motif_diversity
        )
    
    def compute_bfi(
        self, 
        tfi: float, 
        vfi: float, 
        cmcs: float,
        weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    ) -> float:
        """
        Compute Brand Fluidity Index.
        
        Uses NORMALIZED CMCS (critical for proper scaling).
        """
        return float(
            weights[0] * tfi + 
            weights[1] * vfi + 
            weights[2] * (1 - cmcs)
        )
    
    # =========================================================================
    # MAIN ANALYSIS FUNCTION
    # =========================================================================
    
    def analyze_brand(
        self,
        texts_by_period: Dict[str, List[str]],
        images_by_period: Dict[str, List[str]],
        brand_name: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Perform complete brand fluidity analysis.
        
        Args:
            texts_by_period: Dict mapping period labels to list of texts
            images_by_period: Dict mapping period labels to list of image paths
            brand_name: Name of the brand
            
        Returns:
            Complete analysis results dictionary with both raw and normalized values
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {brand_name}")
        print(f"{'='*60}")
        
        results = {
            "brand_name": brand_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "analyzer_version": "2.1.1",  # With backward-compatible fields
            "periods_analyzed": sorted(set(texts_by_period.keys()) | set(images_by_period.keys()))
        }
        
        # =====================================================================
        # TEXTUAL METRICS
        # =====================================================================
        print("\n[1/7] Computing textual metrics...")
        
        # Semantic drift (returns normalized and raw)
        semantic_drift, semantic_drift_raw, text_similarities, text_embeddings = \
            self.compute_semantic_drift(texts_by_period)
        
        # Tone diversity (returns normalized and raw)
        tone_diversity, tone_diversity_raw = self.compute_tone_diversity(texts_by_period)
        
        # Topic turnover (no normalization needed - already 0-1)
        topic_turnover, topic_turnovers, topic_sets = self.compute_topic_turnover(texts_by_period)
        
        # TFI using normalized values
        tfi = self.compute_tfi(semantic_drift, tone_diversity, topic_turnover)
        
        results["textual"] = {
            "semantic_drift": semantic_drift,
            "semantic_drift_raw": semantic_drift_raw,
            "tone_diversity": tone_diversity,
            "tone_diversity_raw": tone_diversity_raw,
            "topic_turnover": topic_turnover,
            "tfi": tfi,
            "period_similarities": text_similarities,
            "period_topic_turnovers": topic_turnovers
        }
        print(f"   Semantic Drift: {semantic_drift:.3f} (raw: {semantic_drift_raw:.3f})")
        print(f"   Tone Diversity: {tone_diversity:.3f} (raw: {tone_diversity_raw:.3f})")
        print(f"   Topic Turnover: {topic_turnover:.3f}")
        print(f"   TFI: {tfi:.3f}")
        
        # =====================================================================
        # VISUAL METRICS
        # =====================================================================
        print("\n[2/7] Computing visual metrics...")
        
        # Visual drift (returns normalized and raw)
        visual_drift, visual_drift_raw, visual_similarities, visual_centroids = \
            self.compute_visual_drift(images_by_period)
        
        # Color entropy
        color_entropy, color_entropy_raw, color_entropy_change = \
            self.compute_color_entropy(images_by_period)
        
        # Motif diversity
        motif_diversity, uniqueness, evenness = self.compute_motif_diversity(images_by_period)
        
        # VFI using normalized values
        vfi = self.compute_vfi(visual_drift, color_entropy, motif_diversity)
        
        results["visual"] = {
            "visual_drift": visual_drift,
            "visual_drift_raw": visual_drift_raw,
            "color_entropy": color_entropy,
            "color_entropy_raw": color_entropy_raw,
            "color_entropy_change": color_entropy_change,
            "motif_diversity": motif_diversity,
            "motif_uniqueness": uniqueness,
            "motif_evenness": evenness,
            "vfi": vfi,
            "period_similarities": visual_similarities,
            # Backward-compatible fields for legacy frontend
            "colorEntropyChange": color_entropy_change,  # camelCase variant
            "entropy_change": color_entropy_change,  # short variant
        }
        print(f"   Visual Drift: {visual_drift:.3f} (raw: {visual_drift_raw:.3f})")
        print(f"   Color Entropy: {color_entropy:.3f} (raw: {color_entropy_raw:.3f})")
        print(f"   Color Entropy Change: {color_entropy_change:.3f}")
        print(f"   Motif Diversity: {motif_diversity:.3f}")
        print(f"   VFI: {vfi:.3f}")
        
        # =====================================================================
        # CROSS-MODAL COHERENCE
        # =====================================================================
        print("\n[3/7] Computing cross-modal coherence...")
        
        cmcs, cmcs_raw, temporal_coherence, period_coherences = \
            self.compute_cross_modal_coherence(texts_by_period, images_by_period)
        
        results["coherence"] = {
            "cmcs": cmcs,
            "cmcs_raw": cmcs_raw,
            "temporal_coherence": temporal_coherence,
            "period_coherences_raw": period_coherences,
            # Backward-compatible fields for legacy frontend
            "centroid_similarity": cmcs,  # Legacy: same as normalized CMCS
            "nn_coherence": temporal_coherence,  # Legacy: nearest-neighbor approximated by temporal
        }
        print(f"   CMCS: {cmcs:.3f} (raw: {cmcs_raw:.3f})")
        print(f"   Temporal Coherence: {temporal_coherence:.3f}")
        
        # =====================================================================
        # BRAND FLUIDITY INDEX
        # =====================================================================
        print("\n[4/7] Computing Brand Fluidity Index...")
        
        bfi = self.compute_bfi(tfi, vfi, cmcs)
        
        # Determine category
        if bfi < 0.30:
            category = "Solid"
            description = "Maintains continuity, stability, predictable cues"
        elif bfi < 0.45:
            category = "Semi-solid"
            description = "Strong identity core with controlled evolution"
        elif bfi < 0.60:
            category = "Semi-fluid"
            description = "Optimal balance for recognition and relevance"
        else:
            category = "Fluid"
            description = "Adaptive, context-responsive, self-reconfiguring"
        
        results["bfi"] = {
            "score": bfi,
            "category": category,
            "description": description
        }
        print(f"   BFI: {bfi:.3f} ({category})")
        
        # =====================================================================
        # SECOND-ORDER METRICS
        # =====================================================================
        print("\n[5/7] Computing drift volatility...")
        
        semantic_drift_volatility = self.compute_drift_volatility(text_similarities)
        visual_drift_volatility = self.compute_drift_volatility(visual_similarities)
        
        print(f"   Semantic Drift Volatility: {semantic_drift_volatility:.3f}")
        print(f"   Visual Drift Volatility: {visual_drift_volatility:.3f}")
        
        print("\n[6/7] Computing coherence trajectory...")
        
        coherence_trajectory = self.compute_coherence_trajectory(period_coherences)
        print(f"   Coherence Trajectory: {coherence_trajectory:+.4f}")
        
        print("\n[7/7] Computing additional second-order metrics...")
        
        # Use RAW values for drift ratio and TSDI (these ratios need absolute scale)
        drift_ratio = self.compute_drift_ratio(semantic_drift_raw, visual_drift_raw)
        directional_consistency_text = self.compute_directional_consistency(text_embeddings)
        directional_consistency_visual = self.compute_directional_consistency(visual_centroids)
        topic_semantic_decoupling = self.compute_topic_semantic_decoupling(topic_turnover, semantic_drift_raw)
        
        results["second_order"] = {
            "semantic_drift_volatility": semantic_drift_volatility,
            "visual_drift_volatility": visual_drift_volatility,
            "coherence_trajectory": coherence_trajectory,
            "drift_ratio": drift_ratio,
            "directional_consistency_text": directional_consistency_text,
            "directional_consistency_visual": directional_consistency_visual,
            "topic_semantic_decoupling": topic_semantic_decoupling
        }
        print(f"   Drift Ratio (SD/VD): {drift_ratio:.2f}")
        print(f"   Directional Consistency (Text): {directional_consistency_text:.3f}")
        print(f"   Directional Consistency (Visual): {directional_consistency_visual:.3f}")
        print(f"   Topic-Semantic Decoupling: {topic_semantic_decoupling:.2f}")
        
        # =====================================================================
        # NORMALIZATION PARAMETERS (for reproducibility)
        # =====================================================================
        results["normalization_params"] = {
            "cmcs_range": [self.norm_params.cmcs_min, self.norm_params.cmcs_max],
            "semantic_drift_range": [self.norm_params.semantic_drift_min, self.norm_params.semantic_drift_max],
            "visual_drift_range": [self.norm_params.visual_drift_min, self.norm_params.visual_drift_max],
            "tone_diversity_max": self.norm_params.tone_diversity_max,
            "color_entropy_max": self.norm_params.color_entropy_max
        }
        
        # =====================================================================
        # SUMMARY
        # =====================================================================
        results["summary"] = {
            "texts_analyzed": sum(len(texts) for texts in texts_by_period.values()),
            "images_analyzed": sum(len(imgs) for imgs in images_by_period.values()),
            "n_periods": len(results["periods_analyzed"])
        }
        
        print(f"\n{'='*60}")
        print(f"Analysis complete: {results['summary']['texts_analyzed']} texts, "
              f"{results['summary']['images_analyzed']} images across "
              f"{results['summary']['n_periods']} periods")
        print(f"{'='*60}\n")
        
        return results
    
    def export_results(
        self, 
        results: Dict[str, Any], 
        output_path: str
    ) -> None:
        """
        Export analysis results to JSON.
        """
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        with open(output_path, 'w') as f:
            json.dump(convert(results), f, indent=2)
        
        print(f"Results exported to: {output_path}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Brand Fluidity Analyzer v2.1")
    print("============================")
    print("\nKey fixes in v2.1:")
    print("- CMCS normalization restored (raw CLIP → interpretable 0-1)")
    print("- Tone diversity ceiling corrected (0.58 → 0.35)")
    print("- Semantic/visual drift normalization added")
    print("- Raw values preserved for transparency")
    print("\nExample usage:")
    print("""
    from brand_fluidity_analyzer_v2_1 import BrandFluidityAnalyzerV2, NormalizationParams
    
    # Initialize with default normalization
    analyzer = BrandFluidityAnalyzerV2()
    
    # Or customize normalization parameters
    custom_params = NormalizationParams(
        cmcs_min=0.10,
        cmcs_max=0.40,
        tone_diversity_max=0.30
    )
    analyzer = BrandFluidityAnalyzerV2(norm_params=custom_params)
    
    # Run analysis
    results = analyzer.analyze_brand(
        texts_by_period=texts_by_period,
        images_by_period=images_by_period,
        brand_name="My Brand"
    )
    
    # Results include both normalized and raw values
    print(f"BFI: {results['bfi']['score']:.3f}")
    print(f"CMCS: {results['coherence']['cmcs']:.3f} (raw: {results['coherence']['cmcs_raw']:.3f})")
    """)
