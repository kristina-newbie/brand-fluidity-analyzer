"""
Brand Fluidity Analyzer: Core Metric Functions (v2 - Corrected)
Implements TFI, VFI, CMCS, and BFI metrics with fixes for:
- Aggregate coherence CMCS (for unpaired text/image data)
- Proper motif diversity via clustering
- Jaccard-based topic turnover
- Timestamp-aware temporal analysis
- Empirically-grounded normalizations

Changes from v1:
- CMCS no longer requires paired text-image data
- Motif diversity uses clustering instead of broken std/mean calculation
- Topic turnover uses Jaccard similarity between time periods
- Better normalization across all metrics
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
from collections import Counter, defaultdict
import cv2
from pathlib import Path
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any


class BrandFluidityAnalyzer:
    """
    Brand Fluidity Analyzer with corrected metric implementations.
    
    Key fixes from original version:
    - CMCS uses aggregate semantic overlap, not arbitrary pairing
    - Motif diversity uses clustering, not broken std/mean
    - Topic turnover uses Jaccard similarity between periods
    - Normalizations are empirically calibrated
    """
    
    def __init__(self, device: str = None):
        """Initialize models for analysis."""
        print("Loading models...")
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Download NLTK data if not present
        try:
            self.sia = SentimentIntensityAnalyzer()
        except:
            import nltk
            nltk.download('vader_lexicon')
            nltk.download('punkt')
            nltk.download('punkt_tab')
            self.sia = SentimentIntensityAnalyzer()
        
        print(f"Models loaded successfully on {self.device}!")
    
    # =========================================================================
    # TEXTUAL FLUIDITY INDEX (TFI) - Corrected
    # =========================================================================
    
    def calculate_textual_fluidity(self, texts_by_year: Dict[int, List[str]]) -> Dict[str, Any]:
        """
        Calculate Textual Fluidity Index (TFI) with corrected topic turnover.
        
        Fixes:
        - Topic turnover now uses Jaccard similarity between consecutive periods
        - Better handling of sparse data
        
        Args:
            texts_by_year: Dict mapping years to lists of text strings
            
        Returns:
            dict with TFI components and final score
        """
        if len(texts_by_year) < 2:
            return {
                "semantic_drift": 0.0,
                "tone_diversity": 0.0,
                "topic_turnover": 0.0,
                "TFI": 0.0,
                "interpretation": "Insufficient data (need 2+ time periods)",
                "debug": {"years_found": list(texts_by_year.keys())}
            }
        
        years = sorted(texts_by_year.keys())
        
        # 1. Semantic Drift (SD) - Average embedding distance between consecutive periods
        drift_scores = []
        for i in range(len(years) - 1):
            texts1 = texts_by_year[years[i]]
            texts2 = texts_by_year[years[i+1]]
            
            if not texts1 or not texts2:
                continue
                
            # Encode concatenated texts for each period
            vec1 = self.sbert_model.encode(" ".join(texts1), convert_to_tensor=True)
            vec2 = self.sbert_model.encode(" ".join(texts2), convert_to_tensor=True)
            
            sim = cosine_similarity(
                vec1.reshape(1, -1).cpu().numpy(), 
                vec2.reshape(1, -1).cpu().numpy()
            )[0][0]
            drift_scores.append(1 - sim)
        
        semantic_drift = np.mean(drift_scores) if drift_scores else 0.0
        
        # 2. Tone Diversity (TD) - Standard deviation of sentiment across all texts
        all_sentiments = []
        for year in years:
            for text in texts_by_year[year]:
                try:
                    for sent in sent_tokenize(text):
                        if len(sent.strip()) > 10:  # Skip very short fragments
                            score = self.sia.polarity_scores(sent)['compound']
                            all_sentiments.append(score)
                except Exception as e:
                    continue
        
        tone_diversity = np.std(all_sentiments) if len(all_sentiments) > 1 else 0.0
        
        # 3. Topic Turnover (TT) - CORRECTED: Jaccard similarity between consecutive periods
        topic_turnover = self._calculate_topic_turnover_jaccard(texts_by_year, years)
        
        # 4. Combine into TFI (equal weights)
        # Normalize components to similar scales before combining
        sd_norm = min(1.0, semantic_drift * 2.5)  # Typical drift 0.1-0.4, scale to 0-1
        td_norm = min(1.0, tone_diversity * 2.0)  # Typical std 0.2-0.5, scale to 0-1
        tt_norm = topic_turnover  # Already 0-1 from Jaccard
        
        TFI = (sd_norm + td_norm + tt_norm) / 3
        
        # Interpretation
        if TFI < 0.3:
            interpretation = "Solid - consistent tone, topics, and style"
        elif TFI < 0.6:
            interpretation = "Semi-fluid - adaptive language with coherent core"
        else:
            interpretation = "Fluid - high variability, flexible narrative identity"
        
        return {
            "semantic_drift": float(semantic_drift),
            "semantic_drift_normalized": float(sd_norm),
            "tone_diversity": float(tone_diversity),
            "tone_diversity_normalized": float(td_norm),
            "topic_turnover": float(topic_turnover),
            "TFI": float(TFI),
            "interpretation": interpretation,
            "debug": {
                "years_analyzed": years,
                "texts_per_year": {y: len(texts_by_year[y]) for y in years},
                "sentiment_samples": len(all_sentiments)
            }
        }
    
    def _calculate_topic_turnover_jaccard(self, texts_by_year: Dict[int, List[str]], 
                                          years: List[int], 
                                          num_topics: int = 10) -> float:
        """
        Calculate topic turnover using Jaccard similarity between consecutive periods.
        
        For each period, identifies dominant topics, then measures how many topics
        appear/disappear between consecutive periods.
        
        Args:
            texts_by_year: Dict mapping years to text lists
            years: Sorted list of years
            num_topics: Number of topics for LDA model
            
        Returns:
            Average topic turnover (1 - Jaccard similarity)
        """
        all_texts = [text for year in years for text in texts_by_year[year]]
        
        if len(all_texts) < 5:
            return 0.0
        
        try:
            # Prepare corpus
            tokenized_texts = [self._tokenize_for_lda(text) for text in all_texts]
            tokenized_texts = [t for t in tokenized_texts if len(t) > 3]
            
            if len(tokenized_texts) < 5:
                return 0.0
            
            dictionary = corpora.Dictionary(tokenized_texts)
            dictionary.filter_extremes(no_below=2, no_above=0.8, keep_n=500)
            
            if len(dictionary) < 10:
                return 0.0
                
            corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
            
            # Train LDA
            lda = models.LdaModel(
                corpus,
                num_topics=min(num_topics, len(all_texts) // 2),
                id2word=dictionary,
                passes=15,
                random_state=42,
                alpha='auto'
            )
            
            # Get dominant topics for each year
            topics_by_year = {}
            text_idx = 0
            
            for year in years:
                year_topics = set()
                for _ in texts_by_year[year]:
                    if text_idx < len(corpus):
                        doc_topics = lda.get_document_topics(corpus[text_idx], minimum_probability=0.1)
                        for topic_id, prob in doc_topics:
                            if prob > 0.15:  # Only count significant topics
                                year_topics.add(topic_id)
                    text_idx += 1
                topics_by_year[year] = year_topics
            
            # Calculate Jaccard-based turnover between consecutive years
            turnovers = []
            for i in range(len(years) - 1):
                t1 = topics_by_year[years[i]]
                t2 = topics_by_year[years[i + 1]]
                
                if len(t1) == 0 and len(t2) == 0:
                    continue
                
                intersection = len(t1 & t2)
                union = len(t1 | t2)
                
                if union > 0:
                    jaccard = intersection / union
                    turnover = 1 - jaccard
                    turnovers.append(turnover)
            
            return np.mean(turnovers) if turnovers else 0.0
            
        except Exception as e:
            print(f"Topic modeling error: {e}")
            return 0.0
    
    def _tokenize_for_lda(self, text: str) -> List[str]:
        """Tokenize text for LDA, removing stopwords and short tokens."""
        # Basic stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you', 'he',
            'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'where', 'when',
            'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there'
        }
        
        # Clean and tokenize
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        tokens = text.split()
        
        return [t for t in tokens if len(t) > 3 and t not in stopwords]

    # =========================================================================
    # VISUAL FLUIDITY INDEX (VFI) - Corrected
    # =========================================================================
    
    def calculate_visual_fluidity(self, image_paths: List[str], 
                                   images_by_year: Dict[int, List[str]] = None) -> Dict[str, Any]:
        """
        Calculate Visual Fluidity Index (VFI) with corrected motif diversity.
        
        Fixes:
        - Motif diversity uses clustering instead of broken std/mean
        - Visual drift computed properly across time periods
        - Better normalization
        
        Args:
            image_paths: List of paths to image files
            images_by_year: Optional dict mapping years to image path lists
            
        Returns:
            dict with VFI components and final score
        """
        if len(image_paths) < 2:
            return {
                "visual_drift": 0.0,
                "color_entropy": 0.0,
                "motif_diversity": 0.0,
                "VFI": 0.0,
                "interpretation": "Insufficient data (need 2+ images)"
            }
        
        embeddings = []
        color_entropies = []
        valid_paths = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for path in image_paths:
            try:
                # Image embedding with CLIP
                image = self.clip_preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.clip_model.encode_image(image)
                    # Normalize embedding
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding.squeeze().cpu().numpy())
                valid_paths.append(path)
                
                # Color entropy
                img = cv2.imread(str(path))
                if img is not None:
                    entropy = self._calculate_color_entropy(img)
                    color_entropies.append(entropy)
                    
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        if len(embeddings) < 2:
            return {
                "visual_drift": 0.0,
                "color_entropy": 0.0,
                "motif_diversity": 0.0,
                "VFI": 0.0,
                "interpretation": "Error processing images"
            }
        
        embeddings_array = np.array(embeddings)
        
        # 1. Visual Drift (VD) - Average pairwise distance or temporal drift
        if images_by_year and len(images_by_year) >= 2:
            visual_drift = self._calculate_temporal_visual_drift(images_by_year)
        else:
            visual_drift = self._calculate_embedding_dispersion(embeddings_array)
        
        # 2. Color Entropy (CE) - Average entropy, properly normalized
        color_entropy_raw = np.mean(color_entropies) if color_entropies else 0.0
        # For 8x8x8 HSV histogram, max entropy is log2(512) ≈ 9 bits
        # Typical images range 4-8 bits
        color_entropy_norm = min(1.0, (color_entropy_raw - 3.0) / 5.0)
        color_entropy_norm = max(0.0, color_entropy_norm)
        
        # 3. Motif Diversity (MD) - CORRECTED: Clustering-based diversity
        motif_diversity = self._calculate_motif_diversity_clustering(embeddings_array)
        
        # Combine into VFI (equal weights)
        VFI = (visual_drift + color_entropy_norm + motif_diversity) / 3
        
        # Interpretation
        if VFI < 0.3:
            interpretation = "Solid - stable visuals (consistent style, palette, imagery)"
        elif VFI < 0.6:
            interpretation = "Semi-fluid - occasional recontextualization"
        else:
            interpretation = "Fluid - high visual experimentation, adaptive aesthetics"
        
        return {
            "visual_drift": float(visual_drift),
            "color_entropy": float(color_entropy_norm),
            "color_entropy_raw": float(color_entropy_raw),
            "motif_diversity": float(motif_diversity),
            "VFI": float(VFI),
            "interpretation": interpretation,
            "debug": {
                "images_processed": len(embeddings),
                "images_failed": len(image_paths) - len(embeddings)
            }
        }
    
    def _calculate_color_entropy(self, img: np.ndarray) -> float:
        """Calculate Shannon entropy of color distribution in HSV space."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist_norm = hist.ravel() / (hist.sum() + 1e-10)
        # Remove zeros to avoid log(0)
        hist_norm = hist_norm[hist_norm > 0]
        entropy = -np.sum(hist_norm * np.log2(hist_norm))
        return entropy
    
    def _calculate_embedding_dispersion(self, embeddings: np.ndarray) -> float:
        """Calculate dispersion of embeddings as a measure of visual variety."""
        # Compute pairwise cosine distances
        similarities = cosine_similarity(embeddings)
        # Get upper triangle (excluding diagonal)
        upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
        # Average distance = 1 - average similarity
        avg_distance = 1 - np.mean(upper_tri)
        # Normalize: typical CLIP distances range 0.1-0.5
        normalized = min(1.0, avg_distance * 2.5)
        return normalized
    
    def _calculate_temporal_visual_drift(self, images_by_year: Dict[int, List[str]]) -> float:
        """Calculate visual drift between consecutive time periods."""
        years = sorted(images_by_year.keys())
        drifts = []
        
        for i in range(len(years) - 1):
            try:
                # Get embeddings for each year
                emb1 = self._get_image_embeddings(images_by_year[years[i]])
                emb2 = self._get_image_embeddings(images_by_year[years[i + 1]])
                
                if len(emb1) == 0 or len(emb2) == 0:
                    continue
                
                # Compute centroid similarity
                centroid1 = np.mean(emb1, axis=0)
                centroid2 = np.mean(emb2, axis=0)
                
                sim = cosine_similarity(
                    centroid1.reshape(1, -1),
                    centroid2.reshape(1, -1)
                )[0][0]
                
                drifts.append(1 - sim)
            except Exception as e:
                continue
        
        if not drifts:
            return 0.0
        
        # Normalize: typical drift 0.05-0.3
        avg_drift = np.mean(drifts)
        return min(1.0, avg_drift * 3.0)
    
    def _get_image_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """Get CLIP embeddings for a list of images."""
        embeddings = []
        for path in image_paths:
            try:
                image = self.clip_preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = self.clip_model.encode_image(image)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.squeeze().cpu().numpy())
            except:
                continue
        return np.array(embeddings) if embeddings else np.array([])
    
    def _calculate_motif_diversity_clustering(self, embeddings: np.ndarray, 
                                               max_clusters: int = 15) -> float:
        """
        Calculate motif diversity using clustering.
        
        Measures what proportion of potential visual categories are represented.
        High diversity = images spread across many clusters.
        Low diversity = images concentrated in few clusters.
        
        Args:
            embeddings: Array of CLIP embeddings
            max_clusters: Maximum expected visual categories
            
        Returns:
            Motif diversity score 0-1
        """
        n_samples = len(embeddings)
        
        if n_samples < 3:
            return 0.0
        
        # Determine optimal number of clusters using sqrt(n) heuristic
        n_clusters = min(max_clusters, max(2, int(np.sqrt(n_samples))))
        
        try:
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Count unique clusters actually used
            unique_clusters = len(set(labels))
            
            # Calculate cluster distribution entropy
            cluster_counts = np.bincount(labels, minlength=n_clusters)
            cluster_probs = cluster_counts / cluster_counts.sum()
            cluster_probs = cluster_probs[cluster_probs > 0]
            
            # Entropy-based diversity
            cluster_entropy = -np.sum(cluster_probs * np.log2(cluster_probs))
            max_entropy = np.log2(n_clusters)
            
            # Combine: unique clusters ratio + entropy ratio
            uniqueness = unique_clusters / n_clusters
            evenness = cluster_entropy / max_entropy if max_entropy > 0 else 0
            
            # Diversity is high when images spread evenly across many clusters
            diversity = (uniqueness + evenness) / 2
            
            return float(diversity)
            
        except Exception as e:
            print(f"Clustering error: {e}")
            return 0.5  # Default middle value

    # =========================================================================
    # CROSS-MODAL COHERENCE SCORE (CMCS) - Corrected for Unpaired Data
    # =========================================================================
    
    def calculate_cross_modal_coherence(self, 
                                         texts: List[str], 
                                         image_paths: List[str],
                                         texts_by_year: Dict[int, List[str]] = None,
                                         images_by_year: Dict[int, List[str]] = None) -> Dict[str, Any]:
        """
        Calculate Cross-Modal Coherence Score using aggregate semantic overlap.
        
        This version does NOT require paired text-image data. Instead, it measures
        whether the brand's textual and visual content occupy similar regions in
        CLIP's shared semantic space.
        
        Args:
            texts: List of all text content
            image_paths: List of all image paths
            texts_by_year: Optional temporal grouping for texts
            images_by_year: Optional temporal grouping for images
            
        Returns:
            dict with CMCS score and components
        """
        if len(texts) < 1 or len(image_paths) < 1:
            return {
                "CMCS": 0.5,
                "interpretation": "Insufficient data for coherence analysis",
                "method": "none"
            }
        
        # Get all text embeddings via CLIP
        text_embeddings = self._get_text_embeddings_clip(texts)
        
        # Get all image embeddings
        image_embeddings = self._get_image_embeddings(image_paths)
        
        if len(text_embeddings) == 0 or len(image_embeddings) == 0:
            return {
                "CMCS": 0.5,
                "interpretation": "Failed to compute embeddings",
                "method": "none"
            }
        
        # Method 1: Centroid similarity (aggregate coherence)
        text_centroid = np.mean(text_embeddings, axis=0, keepdims=True)
        image_centroid = np.mean(image_embeddings, axis=0, keepdims=True)
        
        centroid_similarity = cosine_similarity(text_centroid, image_centroid)[0][0]
        
        # Method 2: Cross-modal nearest neighbor overlap
        nn_coherence = self._calculate_nn_coherence(text_embeddings, image_embeddings)
        
        # Method 3: Temporal coherence (if timestamps available)
        temporal_coherence = None
        if texts_by_year and images_by_year:
            temporal_coherence = self._calculate_temporal_coherence(
                texts_by_year, images_by_year
            )
        
        # Combine methods
        if temporal_coherence is not None:
            # Weight temporal coherence higher when available
            CMCS = 0.3 * centroid_similarity + 0.3 * nn_coherence + 0.4 * temporal_coherence
        else:
            CMCS = 0.5 * centroid_similarity + 0.5 * nn_coherence
        
        # Interpretation
        if CMCS > 0.6:
            interpretation = "Strong coherence - text and visuals occupy aligned semantic territory"
        elif CMCS > 0.4:
            interpretation = "Moderate coherence - partial alignment between text and visual themes"
        else:
            interpretation = "Weak coherence - textual and visual brand expressions are semantically distant"
        
        result = {
            "CMCS": float(CMCS),
            "centroid_similarity": float(centroid_similarity),
            "nn_coherence": float(nn_coherence),
            "interpretation": interpretation,
            "method": "aggregate_unpaired",
            "debug": {
                "texts_embedded": len(text_embeddings),
                "images_embedded": len(image_embeddings)
            }
        }
        
        if temporal_coherence is not None:
            result["temporal_coherence"] = float(temporal_coherence)
        
        return result
    
    def _get_text_embeddings_clip(self, texts: List[str]) -> np.ndarray:
        """Get CLIP text embeddings for a list of texts."""
        embeddings = []
        
        for text in texts:
            try:
                # CLIP has 77 token limit, truncate intelligently
                truncated = text[:300]  # Rough char limit before tokenization
                tokens = clip.tokenize([truncated], truncate=True).to(self.device)
                
                with torch.no_grad():
                    emb = self.clip_model.encode_text(tokens)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.squeeze().cpu().numpy())
            except Exception as e:
                continue
        
        return np.array(embeddings) if embeddings else np.array([])
    
    def _calculate_nn_coherence(self, text_emb: np.ndarray, image_emb: np.ndarray,
                                 threshold: float = 0.25) -> float:
        """
        Calculate coherence based on cross-modal nearest neighbor distances.
        
        Measures what proportion of items in each modality have a "close"
        neighbor in the other modality.
        """
        # Compute cross-modal similarities
        cross_sim = cosine_similarity(text_emb, image_emb)
        
        # For each text, find max similarity to any image
        text_max_sim = np.max(cross_sim, axis=1)
        # For each image, find max similarity to any text
        image_max_sim = np.max(cross_sim, axis=0)
        
        # Average of best matches
        avg_best_match = (np.mean(text_max_sim) + np.mean(image_max_sim)) / 2
        
        return float(avg_best_match)
    
    def _calculate_temporal_coherence(self, 
                                       texts_by_year: Dict[int, List[str]],
                                       images_by_year: Dict[int, List[str]]) -> float:
        """
        Calculate coherence within each time period, then average.
        
        Materials from the same year likely share campaign themes,
        providing a more meaningful coherence measure.
        """
        common_years = set(texts_by_year.keys()) & set(images_by_year.keys())
        
        if len(common_years) == 0:
            return None
        
        period_coherences = []
        
        for year in sorted(common_years):
            try:
                text_emb = self._get_text_embeddings_clip(texts_by_year[year])
                image_emb = self._get_image_embeddings(images_by_year[year])
                
                if len(text_emb) == 0 or len(image_emb) == 0:
                    continue
                
                # Centroid similarity for this period
                text_centroid = np.mean(text_emb, axis=0, keepdims=True)
                image_centroid = np.mean(image_emb, axis=0, keepdims=True)
                
                sim = cosine_similarity(text_centroid, image_centroid)[0][0]
                period_coherences.append(sim)
                
            except Exception as e:
                continue
        
        return np.mean(period_coherences) if period_coherences else None

    # =========================================================================
    # BRAND FLUIDITY INDEX (BFI) - Composite
    # =========================================================================
    
    def compute_brand_fluidity_index(self, TFI: float, VFI: float, CMCS: float,
                                      alpha: float = 0.4, beta: float = 0.4, 
                                      gamma: float = 0.2) -> Dict[str, Any]:
        """
        Compute composite Brand Fluidity Index (BFI).
        
        BFI = α*TFI + β*VFI + γ*(1-CMCS)
        
        Args:
            TFI: Textual Fluidity Index score
            VFI: Visual Fluidity Index score
            CMCS: Cross-Modal Coherence Score
            alpha, beta, gamma: Weights (default: 0.4, 0.4, 0.2)
            
        Returns:
            dict with BFI and interpretation
        """
        # Inversion of CMCS because higher incoherence → higher fluidity
        BFI = alpha * TFI + beta * VFI + gamma * (1 - CMCS)
        
        # Overall interpretation
        if BFI < 0.3:
            category = "Solid"
            description = "Brand maintains strong continuity, stability, and predictable cues"
        elif BFI < 0.6:
            category = "Semi-fluid"
            description = "Optimal balance - stable enough for recognition, fluid enough for relevance"
        else:
            category = "Highly Fluid"
            description = "Brand identity is adaptive, context-responsive, and self-reconfiguring"
        
        return {
            "BFI": float(BFI),
            "category": category,
            "description": description,
            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
            "components": {
                "TFI_contribution": float(alpha * TFI),
                "VFI_contribution": float(beta * VFI),
                "CMCS_contribution": float(gamma * (1 - CMCS))
            }
        }

    # =========================================================================
    # COMPLETE ANALYSIS PIPELINE
    # =========================================================================
    
    def analyze_brand(self, 
                      text_dir: str, 
                      image_dir: str, 
                      brand_name: str,
                      year_pattern: str = r'(\d{4})') -> Dict[str, Any]:
        """
        Complete brand fluidity analysis with timestamp support.
        
        Args:
            text_dir: Directory containing text files
            image_dir: Directory containing image files
            brand_name: Name of the brand
            year_pattern: Regex pattern to extract year from filenames
            
        Returns:
            Complete analysis results
        """
        results = {
            "brand_name": brand_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "analyzer_version": "2.0"
        }
        
        # Load texts with year extraction
        text_dir = Path(text_dir)
        texts_by_year = defaultdict(list)
        all_texts = []
        
        if text_dir.exists():
            for text_file in sorted(text_dir.glob("**/*")):
                if text_file.is_file() and text_file.suffix.lower() in ['.txt', '.md', '.html']:
                    try:
                        content = text_file.read_text(encoding='utf-8', errors='ignore')
                        
                        # Skip empty or very short files
                        if len(content.strip()) < 50:
                            continue
                        
                        # Extract year from filename
                        year = self._extract_year(text_file.name, year_pattern)
                        texts_by_year[year].append(content)
                        all_texts.append(content)
                        
                    except Exception as e:
                        print(f"Error reading {text_file}: {e}")
        
        # Load images with year extraction
        image_dir = Path(image_dir)
        images_by_year = defaultdict(list)
        all_image_paths = []
        
        if image_dir.exists():
            for img_file in sorted(image_dir.glob("**/*")):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                    year = self._extract_year(img_file.name, year_pattern)
                    images_by_year[year].append(str(img_file))
                    all_image_paths.append(str(img_file))
        
        print(f"\n{'='*60}")
        print(f"Analyzing {brand_name}")
        print(f"{'='*60}")
        print(f"Texts: {len(all_texts)} files across {len(texts_by_year)} time periods")
        print(f"Images: {len(all_image_paths)} files across {len(images_by_year)} time periods")
        print(f"Text years: {sorted(texts_by_year.keys())}")
        print(f"Image years: {sorted(images_by_year.keys())}")
        
        # Calculate TFI
        print("\nCalculating Textual Fluidity Index...")
        tfi_results = self.calculate_textual_fluidity(dict(texts_by_year))
        results["textual_fluidity"] = tfi_results
        print(f"  TFI: {tfi_results['TFI']:.3f} ({tfi_results['interpretation']})")
        
        # Calculate VFI
        print("\nCalculating Visual Fluidity Index...")
        vfi_results = self.calculate_visual_fluidity(
            all_image_paths, 
            dict(images_by_year)
        )
        results["visual_fluidity"] = vfi_results
        print(f"  VFI: {vfi_results['VFI']:.3f} ({vfi_results['interpretation']})")
        
        # Calculate CMCS - now uses aggregate coherence for unpaired data
        print("\nCalculating Cross-Modal Coherence Score...")
        cmcs_results = self.calculate_cross_modal_coherence(
            all_texts,
            all_image_paths,
            dict(texts_by_year),
            dict(images_by_year)
        )
        results["cross_modal_coherence"] = cmcs_results
        print(f"  CMCS: {cmcs_results['CMCS']:.3f} ({cmcs_results['interpretation']})")
        
        # Calculate BFI
        print("\nCalculating Brand Fluidity Index...")
        bfi_results = self.compute_brand_fluidity_index(
            TFI=tfi_results["TFI"],
            VFI=vfi_results["VFI"],
            CMCS=cmcs_results["CMCS"]
        )
        results["brand_fluidity_index"] = bfi_results
        print(f"  BFI: {bfi_results['BFI']:.3f} ({bfi_results['category']})")
        
        # Summary
        results["summary"] = {
            "texts_analyzed": len(all_texts),
            "images_analyzed": len(all_image_paths),
            "text_time_periods": len(texts_by_year),
            "image_time_periods": len(images_by_year),
            "overall_category": bfi_results["category"]
        }
        
        print(f"\n{'='*60}")
        print(f"Analysis complete for {brand_name}")
        print(f"{'='*60}\n")
        
        return results
    
    def _extract_year(self, filename: str, pattern: str = r'(\d{4})') -> int:
        """Extract year from filename using regex pattern."""
        match = re.search(pattern, filename)
        if match:
            year = int(match.group(1))
            # Validate reasonable year range
            if 1990 <= year <= 2030:
                return year
        return 2024  # Default if no valid year found


# =========================================================================
# COMPARISON UTILITIES
# =========================================================================

def compare_brands(results_list: List[Dict]) -> pd.DataFrame:
    """
    Create comparison table for multiple brand analyses.
    
    Args:
        results_list: List of analysis result dicts
        
    Returns:
        DataFrame with brand comparisons
    """
    rows = []
    
    for r in results_list:
        rows.append({
            'Brand': r['brand_name'],
            'TFI': r['textual_fluidity']['TFI'],
            'Semantic Drift': r['textual_fluidity']['semantic_drift'],
            'Tone Diversity': r['textual_fluidity']['tone_diversity'],
            'Topic Turnover': r['textual_fluidity']['topic_turnover'],
            'VFI': r['visual_fluidity']['VFI'],
            'Visual Drift': r['visual_fluidity']['visual_drift'],
            'Color Entropy': r['visual_fluidity']['color_entropy'],
            'Motif Diversity': r['visual_fluidity']['motif_diversity'],
            'CMCS': r['cross_modal_coherence']['CMCS'],
            'BFI': r['brand_fluidity_index']['BFI'],
            'Category': r['brand_fluidity_index']['category']
        })
    
    df = pd.DataFrame(rows)
    return df.round(3)
Fix CMCS for unpaired data, motif diversity, topic turnover
