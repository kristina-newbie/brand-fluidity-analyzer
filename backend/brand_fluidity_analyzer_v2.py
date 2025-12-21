"""
Brand Fluidity Analyzer v2.0
============================
Enhanced metric computation for brand identity fluidity analysis.

Implements core metrics (TFI, VFI, CMCS, BFI) and second-order metrics
(Drift Volatility, Coherence Trajectory, Drift Ratio, Directional Consistency,
Topic-Semantic Decoupling Index) as defined in the IJRM research framework.

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
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FluidityMetrics:
    """Container for all fluidity metrics."""
    # Core metrics
    semantic_drift: float
    tone_diversity: float
    topic_turnover: float
    visual_drift: float
    color_entropy: float
    motif_diversity: float
    cmcs: float
    temporal_coherence: float
    
    # Composite indices
    tfi: float
    vfi: float
    bfi: float
    
    # Second-order metrics
    semantic_drift_volatility: float
    visual_drift_volatility: float
    coherence_trajectory: float
    drift_ratio: float
    directional_consistency_text: float
    directional_consistency_visual: float
    topic_semantic_decoupling: float
    
    # Per-period data
    period_semantic_similarities: List[float]
    period_visual_similarities: List[float]
    period_coherence_scores: List[float]
    period_topic_turnovers: List[float]


class BrandFluidityAnalyzerV2:
    """
    Enhanced Brand Fluidity Analyzer with second-order metrics.
    
    Computes:
    - Core Metrics: TFI, VFI, CMCS, BFI
    - Second-Order Metrics: Drift Volatility, Coherence Trajectory,
      Drift Ratio, Directional Consistency, Topic-Semantic Decoupling
    """
    
    def __init__(self, device: str = None):
        """
        Initialize models for analysis.
        
        Args:
            device: 'cuda' or 'cpu'. Auto-detects if None.
        """
        print("Initializing Brand Fluidity Analyzer v2.0...")
        
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
    ) -> Tuple[float, List[float], List[np.ndarray]]:
        """
        Compute semantic drift using Sentence-BERT embeddings.
        
        Reference: Reimers & Gurevych (2019), EMNLP
        
        Args:
            texts_by_period: Dict mapping period labels to list of texts
            
        Returns:
            Tuple of (mean_drift, period_similarities, period_embeddings)
        """
        periods = sorted(texts_by_period.keys())
        if len(periods) < 2:
            return 0.0, [], []
        
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
        
        # Drift = 1 - similarity
        drifts = [1 - s for s in similarities]
        mean_drift = np.mean(drifts) if drifts else 0.0
        
        return float(mean_drift), similarities, period_embeddings
    
    def compute_tone_diversity(
        self, 
        texts_by_period: Dict[str, List[str]]
    ) -> float:
        """
        Compute tone diversity using VADER sentiment analysis.
        
        Reference: Hutto & Gilbert (2014), ICWSM
        
        Args:
            texts_by_period: Dict mapping period labels to list of texts
            
        Returns:
            Standard deviation of sentiment scores (normalized)
        """
        all_sentiments = []
        
        for period, texts in texts_by_period.items():
            for text in texts:
                try:
                    sentences = sent_tokenize(text)
                    for sent in sentences:
                        score = self.sia.polarity_scores(sent)['compound']
                        all_sentiments.append(score)
                except Exception:
                    continue
        
        if len(all_sentiments) < 2:
            return 0.0
        
        # Standard deviation, normalized by theoretical max (~0.58 for uniform [-1,1])
        tone_std = np.std(all_sentiments)
        normalized_tone = min(1.0, tone_std / 0.58)
        
        return float(normalized_tone)
    
    def compute_topic_turnover(
        self, 
        texts_by_period: Dict[str, List[str]],
        n_topics: int = 5
    ) -> Tuple[float, List[float], List[set]]:
        """
        Compute topic turnover using LDA and Jaccard dissimilarity.
        
        Reference: Blei et al. (2003), JMLR
        
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
    ) -> Tuple[float, List[float], List[np.ndarray]]:
        """
        Compute visual drift using CLIP embeddings.
        
        Reference: Radford et al. (2021), ICML
        
        Args:
            images_by_period: Dict mapping period labels to list of image paths
            
        Returns:
            Tuple of (mean_drift, period_similarities, period_centroids)
        """
        periods = sorted(images_by_period.keys())
        if len(periods) < 2:
            return 0.0, [], []
        
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
        
        # Drift = 1 - similarity
        drifts = [1 - s for s in similarities]
        mean_drift = np.mean(drifts) if drifts else 0.0
        
        return float(mean_drift), similarities, period_centroids
    
    def compute_color_entropy(
        self, 
        images_by_period: Dict[str, List[str]]
    ) -> Tuple[float, float]:
        """
        Compute color entropy using HSV histograms.
        
        Reference: Shannon (1948); Swain & Ballard (1991), IJCV
        
        Args:
            images_by_period: Dict mapping period labels to list of image paths
            
        Returns:
            Tuple of (mean_entropy_normalized, entropy_change)
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
            return 0.0, 0.0
        
        # Mean entropy, normalized by theoretical max (log2(512) ≈ 9)
        mean_entropy = np.mean(period_entropies)
        normalized_entropy = min(1.0, mean_entropy / 9.0)
        
        # Entropy change (std of period entropies, normalized)
        if len(period_entropies) > 1:
            entropy_change = np.std(period_entropies) / 9.0
        else:
            entropy_change = 0.0
        
        return float(normalized_entropy), float(entropy_change)
    
    def compute_motif_diversity(
        self, 
        images_by_period: Dict[str, List[str]],
        n_clusters: int = 8
    ) -> Tuple[float, float, float]:
        """
        Compute motif diversity using K-means clustering of CLIP embeddings.
        
        Reference: Jain (2010), Pattern Recognition Letters
        
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
    # CROSS-MODAL COHERENCE
    # =========================================================================
    
    def compute_cross_modal_coherence(
        self,
        texts_by_period: Dict[str, List[str]],
        images_by_period: Dict[str, List[str]]
    ) -> Tuple[float, float, List[float]]:
        """
        Compute cross-modal coherence using CLIP joint embeddings.
        
        Reference: Radford et al. (2021), ICML; Liang et al. (2022), ACL
        
        Args:
            texts_by_period: Dict mapping period labels to list of texts
            images_by_period: Dict mapping period labels to list of image paths
            
        Returns:
            Tuple of (cmcs, temporal_coherence, period_coherences)
        """
        periods = sorted(set(texts_by_period.keys()) & set(images_by_period.keys()))
        if not periods:
            return 1.0, 1.0, []
        
        all_similarities = []
        period_coherences = []
        
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
                    text_tokens = clip.tokenize([text]).to(self.device)
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
                period_coherences.append(np.mean(period_sims))
        
        # Overall CMCS
        cmcs = np.mean(all_similarities) if all_similarities else 1.0
        
        # Temporal coherence (mean of period coherences)
        temporal_coherence = np.mean(period_coherences) if period_coherences else 1.0
        
        return float(cmcs), float(temporal_coherence), period_coherences
    
    # =========================================================================
    # SECOND-ORDER METRICS
    # =========================================================================
    
    def compute_drift_volatility(
        self, 
        period_similarities: List[float]
    ) -> float:
        """
        Compute drift volatility (coefficient of variation of drift scores).
        
        Reference: Everitt & Skrondal (2010)
        
        Args:
            period_similarities: List of consecutive period similarities
            
        Returns:
            Coefficient of variation of drift scores
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
        
        Reference: Hamilton (1994), Time Series Analysis
        
        Args:
            period_coherences: List of per-period coherence scores
            
        Returns:
            Slope of coherence over time
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
        
        Args:
            semantic_drift: Mean semantic drift
            visual_drift: Mean visual drift
            
        Returns:
            Ratio of semantic to visual drift
        """
        if visual_drift < 1e-7:
            return 0.0 if semantic_drift < 1e-7 else 10.0  # Cap at 10
        
        ratio = semantic_drift / visual_drift
        return float(min(10.0, ratio))
    
    def compute_directional_consistency(
        self, 
        period_embeddings: List[np.ndarray]
    ) -> float:
        """
        Compute directional consistency of identity evolution.
        
        Reference: Hamilton et al. (2016), ACL
        
        Args:
            period_embeddings: List of period centroid embeddings
            
        Returns:
            Mean cosine of angles between consecutive drift vectors
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
        
        Args:
            topic_turnover: Mean topic turnover
            semantic_drift: Mean semantic drift
            
        Returns:
            Ratio of topic turnover to semantic drift
        """
        if semantic_drift < 1e-7:
            return 0.0 if topic_turnover < 1e-7 else 10.0  # Cap at 10
        
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
        
        Args:
            semantic_drift: Mean semantic drift
            tone_diversity: Tone diversity score
            topic_turnover: Mean topic turnover
            weights: Component weights (default equal)
            
        Returns:
            TFI score
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
        
        Args:
            visual_drift: Mean visual drift
            color_entropy: Normalized color entropy
            motif_diversity: Motif diversity score
            weights: Component weights (default equal)
            
        Returns:
            VFI score
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
        
        Args:
            tfi: Textual Fluidity Index
            vfi: Visual Fluidity Index
            cmcs: Cross-Modal Coherence Score
            weights: Component weights (default α=0.4, β=0.4, γ=0.2)
            
        Returns:
            BFI score
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
            Complete analysis results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {brand_name}")
        print(f"{'='*60}")
        
        results = {
            "brand_name": brand_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "periods_analyzed": sorted(set(texts_by_period.keys()) | set(images_by_period.keys()))
        }
        
        # =====================================================================
        # TEXTUAL METRICS
        # =====================================================================
        print("\n[1/7] Computing textual metrics...")
        
        # Semantic drift
        semantic_drift, text_similarities, text_embeddings = self.compute_semantic_drift(texts_by_period)
        
        # Tone diversity
        tone_diversity = self.compute_tone_diversity(texts_by_period)
        
        # Topic turnover
        topic_turnover, topic_turnovers, topic_sets = self.compute_topic_turnover(texts_by_period)
        
        # TFI
        tfi = self.compute_tfi(semantic_drift, tone_diversity, topic_turnover)
        
        results["textual"] = {
            "semantic_drift": semantic_drift,
            "tone_diversity": tone_diversity,
            "topic_turnover": topic_turnover,
            "tfi": tfi,
            "period_similarities": text_similarities,
            "period_topic_turnovers": topic_turnovers
        }
        print(f"   Semantic Drift: {semantic_drift:.3f}")
        print(f"   Tone Diversity: {tone_diversity:.3f}")
        print(f"   Topic Turnover: {topic_turnover:.3f}")
        print(f"   TFI: {tfi:.3f}")
        
        # =====================================================================
        # VISUAL METRICS
        # =====================================================================
        print("\n[2/7] Computing visual metrics...")
        
        # Visual drift
        visual_drift, visual_similarities, visual_centroids = self.compute_visual_drift(images_by_period)
        
        # Color entropy
        color_entropy, color_entropy_change = self.compute_color_entropy(images_by_period)
        
        # Motif diversity
        motif_diversity, uniqueness, evenness = self.compute_motif_diversity(images_by_period)
        
        # VFI
        vfi = self.compute_vfi(visual_drift, color_entropy, motif_diversity)
        
        results["visual"] = {
            "visual_drift": visual_drift,
            "color_entropy": color_entropy,
            "color_entropy_change": color_entropy_change,
            "motif_diversity": motif_diversity,
            "motif_uniqueness": uniqueness,
            "motif_evenness": evenness,
            "vfi": vfi,
            "period_similarities": visual_similarities
        }
        print(f"   Visual Drift: {visual_drift:.3f}")
        print(f"   Color Entropy: {color_entropy:.3f}")
        print(f"   Motif Diversity: {motif_diversity:.3f}")
        print(f"   VFI: {vfi:.3f}")
        
        # =====================================================================
        # CROSS-MODAL COHERENCE
        # =====================================================================
        print("\n[3/7] Computing cross-modal coherence...")
        
        cmcs, temporal_coherence, period_coherences = self.compute_cross_modal_coherence(
            texts_by_period, images_by_period
        )
        
        results["coherence"] = {
            "cmcs": cmcs,
            "temporal_coherence": temporal_coherence,
            "period_coherences": period_coherences
        }
        print(f"   CMCS: {cmcs:.3f}")
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
        
        drift_ratio = self.compute_drift_ratio(semantic_drift, visual_drift)
        directional_consistency_text = self.compute_directional_consistency(text_embeddings)
        directional_consistency_visual = self.compute_directional_consistency(visual_centroids)
        topic_semantic_decoupling = self.compute_topic_semantic_decoupling(topic_turnover, semantic_drift)
        
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
        
        Args:
            results: Analysis results dictionary
            output_path: Path for output JSON file
        """
        # Convert numpy types to native Python types
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
    # Example usage
    print("Brand Fluidity Analyzer v2.0")
    print("============================")
    print("\nExample usage:")
    print("""
    from brand_fluidity_analyzer_v2 import BrandFluidityAnalyzerV2
    
    # Initialize analyzer
    analyzer = BrandFluidityAnalyzerV2()
    
    # Prepare data
    texts_by_period = {
        "2015": ["Brand text from 2015...", "More text..."],
        "2017": ["Brand text from 2017...", "More text..."],
        "2019": ["Brand text from 2019...", "More text..."],
        # ... more periods
    }
    
    images_by_period = {
        "2015": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
        "2017": ["/path/to/image3.jpg", "/path/to/image4.jpg"],
        "2019": ["/path/to/image5.jpg", "/path/to/image6.jpg"],
        # ... more periods
    }
    
    # Run analysis
    results = analyzer.analyze_brand(
        texts_by_period=texts_by_period,
        images_by_period=images_by_period,
        brand_name="My Brand"
    )
    
    # Export results
    analyzer.export_results(results, "brand_analysis.json")
    """)
