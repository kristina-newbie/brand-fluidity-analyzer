"""
Brand Liquidity Analyzer v5.0
Temporal Liquidity & Coherence Measurement

v5.0 Changes (methodology alignment):
- Renamed all terminology from "Fluidity" to "Liquidity" (Bauman, 2000)
  TFI -> TLI, VFI -> VLI, BFI -> BLI
- CMCS rewritten: centroid-to-centroid cosine similarity in CLIP's joint
  latent space (holistic brand territory alignment per Kapferer's Brand
  Identity Prism). Replaces paired-diagonal approach. Text and image
  corpora are treated as independent collections per period.
- Temporal Coherence now uses sample std (ddof=1), consistent with
  Drift Volatility and Bessel's correction as documented in methodology.
- Color Entropy: per-image pixel sampling (max 1000) applied before
  period aggregation, matching methodology description.
- Removed redundant character-level CLIP text truncation; relies on
  clip.tokenize(truncate=True) for proper token-level handling.

v4.1 Changes:
- Drift volatility now uses sample standard deviation (ddof=1)
- Added edge case handling for single transition periods

v4 Changes:
- TLI = Semantic Drift only (Topic Turnover is diagnostic)
- VLI = Visual Drift only
- BLI = 0.5*TLI + 0.5*VLI (CMCS is separate axis, not in BLI)
"""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from gensim import corpora, models

warnings.filterwarnings('ignore')


@dataclass
class NormalizationParams:
    """
    Empirically-derived normalization parameters.

    These bounds are calibrated for period-centroid comparisons:
    - Semantic drift: Period centroids have high similarity (0.75-0.95),
      so drift values (1-similarity) typically range 0.05-0.25
    - Visual drift: CLIP embeddings show similar patterns, 0.02-0.12
    - CMCS: Cross-modal centroid-to-centroid similarities typically 0.20-0.45.
      Bounds recalibrated for v5.0 centroid method (previously 0.15-0.32
      for paired-diagonal method). Centroid averaging eliminates within-
      group variance, mechanically increasing cosine similarity relative
      to individual-pair measurement (Robinson, 1950; ecological
      correlation effect). Upper bound reflects CLIP's inherent modality
      gap ceiling (Liang et al., 2022).
    """
    cmcs_min: float = 0.20
    cmcs_max: float = 0.45
    semantic_drift_min: float = 0.05
    semantic_drift_max: float = 0.25
    visual_drift_min: float = 0.02
    visual_drift_max: float = 0.12


class BrandLiquidityAnalyzer:
    """
    Brand Liquidity Analyzer v5.0

    Operationalizes Bauman's (2000) liquid modernity theory for brand
    identity measurement through computational text and image analysis.

    Core indices:
    - TLI = Semantic Drift (normalized)   [Textual Liquidity Index]
    - VLI = Visual Drift (normalized)     [Visual Liquidity Index]
    - BLI = 0.5*TLI + 0.5*VLI            [Brand Liquidity Index]

    CMCS is reported on a separate axis (not in BLI).
    Cross-Modal Coherence measures holistic alignment between textual
    and visual brand expressions via centroid comparison in CLIP's
    joint latent space (Radford et al., 2021).
    """

    def __init__(self, device: str = None, norm_params: NormalizationParams = None):
        self.norm_params = norm_params or NormalizationParams()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INIT] Using device: {self.device}")

        print("[INIT] Loading CLIP (ViT-B/32)...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        print("  CLIP loaded")

        print("[INIT] Loading Sentence-BERT (all-MiniLM-L6-v2)...")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  Sentence-BERT loaded")

        print("[INIT] Loading VADER sentiment analyzer...")
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            nltk.download("punkt", quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        print("  VADER loaded")

    # -------------------- Utility --------------------

    @staticmethod
    def _normalize(value: float, vmin: float, vmax: float) -> float:
        if vmax == vmin:
            return 0.0
        norm = (value - vmin) / (vmax - vmin)
        return float(np.clip(norm, 0.0, 1.0))

    # -------------------- Text Embeddings & Drift --------------------

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Sentence-BERT (all-MiniLM-L6-v2).
        
        Used for TLI semantic drift computation. Returns individual
        text embeddings (not centroids).
        """
        if not texts:
            return np.zeros((1, self.sbert_model.get_sentence_embedding_dimension()))
        return self.sbert_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def compute_semantic_drift(
        self, texts_by_period: Dict[str, List[str]]
    ) -> Tuple[float, float, Dict[str, float], Dict[str, np.ndarray]]:
        """Compute Semantic Drift as average cosine distance between
        consecutive period embedding centroids (Sentence-BERT).
        
        Methodology reference: TLI computation, Hamilton et al. (2016)
        temporal embedding methodology.
        """
        period_keys = sorted(texts_by_period.keys())
        period_embeddings: Dict[str, np.ndarray] = {}

        for period in period_keys:
            texts = texts_by_period[period]
            if not texts:
                emb = np.zeros((1, self.sbert_model.get_sentence_embedding_dimension()))
            else:
                emb = self.embed_texts(texts).mean(axis=0, keepdims=True)
            period_embeddings[period] = emb

        raw_drift, similarities = self._compute_consecutive_cosine_drift(period_keys, period_embeddings)
        norm_drift = self._normalize(raw_drift, self.norm_params.semantic_drift_min, self.norm_params.semantic_drift_max)
        return norm_drift, raw_drift, similarities, period_embeddings

    # -------------------- Topic Turnover (Diagnostic) --------------------

    @staticmethod
    def _tokenize_for_topic(text: str) -> List[str]:
        return [t.lower() for t in text.split() if len(t) > 2]

    def compute_topic_turnover(
        self, texts_by_period: Dict[str, List[str]], num_topics: int = 5
    ) -> Tuple[float, Dict[str, float], Dict[str, List[str]]]:
        """Compute Topic Turnover via period-specific LDA and Jaccard distance.
        
        Diagnostic metric — does not enter BLI.
        Methodology reference: Blei et al. (2003), Wang & McCallum (2006).
        """
        period_keys = sorted(texts_by_period.keys())
        topic_sets: Dict[str, List[str]] = {}
        period_turnovers: Dict[str, float] = {}

        for period in period_keys:
            texts = texts_by_period[period]
            tokens = [self._tokenize_for_topic(t) for t in texts if t.strip()]
            tokens = [t for t in tokens if t]
            if not tokens:
                topic_sets[period] = []
                continue

            dictionary = corpora.Dictionary(tokens)
            dictionary.filter_extremes(no_below=1, no_above=0.9)
            if len(dictionary.keys()) == 0:
                topic_sets[period] = []
                continue

            corpus = [dictionary.doc2bow(t) for t in tokens]
            try:
                lda = models.LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=min(num_topics, len(dictionary.keys())),
                    random_state=42,
                    passes=5,
                    iterations=50,
                )
            except Exception:
                topic_sets[period] = []
                continue

            period_topics = set()
            for topic_id in range(lda.num_topics):
                terms = lda.get_topic_terms(topic_id, topn=3)
                words = [dictionary[w_id] for w_id, _ in terms]
                period_topics.update(words)
            topic_sets[period] = sorted(list(period_topics))

        turnovers: List[float] = []
        for i in range(len(period_keys) - 1):
            p1, p2 = period_keys[i], period_keys[i + 1]
            s1, s2 = set(topic_sets[p1]), set(topic_sets[p2])
            if not s1 and not s2:
                turnover = 0.0
            else:
                inter = len(s1.intersection(s2))
                union = len(s1.union(s2))
                turnover = 1.0 - (inter / union if union > 0 else 0.0)
            period_turnovers[f"{p1}->{p2}"] = turnover
            turnovers.append(turnover)

        avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0
        return avg_turnover, period_turnovers, topic_sets

    # -------------------- Tone Diversity (Descriptive) --------------------

    def compute_tone_diversity(self, texts_by_period: Dict[str, List[str]]) -> Tuple[float, float]:
        """Compute Tone Diversity as normalized Shannon entropy of VADER
        sentiment score distribution across five equal-width bins.
        
        Descriptive metric — does not enter BLI.
        """
        sentiments: List[float] = []
        for texts in texts_by_period.values():
            for text in texts:
                if not text.strip():
                    continue
                sents = sent_tokenize(text)
                for s in sents:
                    score = self.sia.polarity_scores(s)["compound"]
                    sentiments.append(score)

        if not sentiments:
            return 0.0, 0.0

        bins = np.linspace(-1, 1, 6)
        hist, _ = np.histogram(sentiments, bins=bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        raw_div = -np.sum(probs * np.log(probs))
        norm_div = float(raw_div / np.log(5))
        return norm_div, raw_div

    # -------------------- Image Embeddings & Drift --------------------

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        """Encode images using CLIP ViT-B/32 with L2-normalization.
        
        Returns individual image embeddings (not centroids).
        Used for VLI drift, Motif Diversity, and CMCS computation.
        """
        if not image_paths:
            return np.zeros((1, 512))

        images = []
        for path in image_paths:
            if not os.path.exists(path):
                continue
            try:
                img = Image.open(path).convert("RGB")
                images.append(self.clip_preprocess(img))
            except Exception:
                continue

        if not images:
            return np.zeros((1, 512))

        batch = torch.stack(images).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()

    def compute_visual_drift(
        self, images_by_period: Dict[str, List[str]]
    ) -> Tuple[float, float, Dict[str, float], Dict[str, np.ndarray]]:
        """Compute Visual Drift as average cosine distance between
        consecutive period image embedding centroids (CLIP ViT-B/32).
        
        Methodology reference: VLI computation, Radford et al. (2021),
        Dosovitskiy et al. (2020).
        """
        period_keys = sorted(images_by_period.keys())
        period_centroids: Dict[str, np.ndarray] = {}

        for period in period_keys:
            paths = images_by_period[period]
            emb = self.embed_images(paths)
            centroid = emb.mean(axis=0, keepdims=True)
            period_centroids[period] = centroid

        raw_drift, similarities = self._compute_consecutive_cosine_drift(period_keys, period_centroids)
        norm_drift = self._normalize(raw_drift, self.norm_params.visual_drift_min, self.norm_params.visual_drift_max)
        return norm_drift, raw_drift, similarities, period_centroids

    # -------------------- Color Entropy (Descriptive) --------------------

    def compute_color_entropy(
        self, images_by_period: Dict[str, List[str]], num_colors: int = 5
    ) -> Tuple[float, float, Dict[str, float]]:
        """Compute Color Entropy via K-Means color quantization and
        normalized Shannon entropy.
        
        FIX v5.0: Per-image pixel sampling (max 1000 pixels) applied
        before period aggregation, preventing large images from
        dominating and matching methodology description.
        
        Descriptive metric — does not enter BLI.
        """
        all_colors: List[np.ndarray] = []
        period_color_entropies: Dict[str, float] = {}

        for period, paths in images_by_period.items():
            period_pixels = []
            for path in paths:
                if not os.path.exists(path):
                    continue
                try:
                    img = Image.open(path).convert("RGB")
                    arr = np.array(img).reshape(-1, 3)
                    # Per-image pixel sampling (max 1000) as per methodology
                    if len(arr) > 1000:
                        arr = arr[np.random.choice(len(arr), 1000, replace=False)]
                    period_pixels.append(arr)
                except Exception:
                    continue

            if not period_pixels:
                period_color_entropies[period] = 0.0
                continue

            pixels = np.vstack(period_pixels)
            if len(pixels) < num_colors:
                period_color_entropies[period] = 0.0
                continue

            norm_entropy, _raw_entropy = self._kmeans_entropy(pixels, num_colors)
            period_color_entropies[period] = norm_entropy
            all_colors.append(pixels)

        if not all_colors:
            return 0.0, 0.0, period_color_entropies

        all_colors_arr = np.vstack(all_colors)
        norm_entropy, raw_entropy = self._kmeans_entropy(all_colors_arr, num_colors)
        return norm_entropy, raw_entropy, period_color_entropies


    @staticmethod
    def _kmeans_entropy(arr: np.ndarray, n_clusters: int) -> Tuple[float, float]:
        """Run KMeans and return (normalized_entropy, raw_entropy)."""
        if arr.size == 0 or len(arr) < n_clusters or n_clusters <= 1:
            return 0.0, 0.0

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(arr)

        counts = np.bincount(labels, minlength=n_clusters)
        probs = counts / counts.sum()
        probs = probs[probs > 0]

        raw_entropy = float(-np.sum(probs * np.log(probs)))
        norm_entropy = float(raw_entropy / np.log(n_clusters))
        return norm_entropy, raw_entropy
    # -------------------- Motif Diversity (Descriptive) --------------------

    def compute_motif_diversity(
        self, images_by_period: Dict[str, List[str]], num_clusters: int = 8
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """Compute Motif Diversity via K-Means clustering of CLIP embeddings.
        
        Returns (diversity, uniqueness, evenness, period_diversity).
        Descriptive metric — does not enter BLI.
        """
        all_paths: List[str] = []
        for paths in images_by_period.values():
            all_paths.extend(paths)

        if not all_paths:
            return 0.0, 0.0, 0.0, {p: 0.0 for p in images_by_period.keys()}

        emb = self.embed_images(all_paths)
        if emb.shape[0] < num_clusters:
            return 0.0, 0.0, 0.0, {p: 0.0 for p in images_by_period.keys()}

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(emb)

        # Uniqueness: proportion of clusters used
        unique_clusters = len(set(labels))
        uniqueness = unique_clusters / num_clusters

        # Evenness: normalized entropy
        global_counts = np.bincount(labels, minlength=num_clusters)
        global_probs = global_counts / global_counts.sum()
        global_probs = global_probs[global_probs > 0]
        global_entropy = -np.sum(global_probs * np.log(global_probs))
        evenness = float(global_entropy / np.log(num_clusters))

        # Diversity: combination
        diversity = (uniqueness + evenness) / 2

        # Per-period diversity
        idx = 0
        period_diversity: Dict[str, float] = {}
        for period, paths in images_by_period.items():
            if not paths:
                period_diversity[period] = 0.0
                continue
            period_labels = labels[idx:idx + len(paths)]
            idx += len(paths)
            counts = np.bincount(period_labels, minlength=num_clusters)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            if len(probs) == 0:
                period_diversity[period] = 0.0
            else:
                entropy = -np.sum(probs * np.log(probs))
                period_diversity[period] = float(entropy / np.log(num_clusters))

        return diversity, uniqueness, evenness, period_diversity

    # -------------------- Cross-Modal Coherence (Separate Axis) --------------------

    def compute_cross_modal_coherence(
        self, texts_by_period: Dict[str, List[str]], images_by_period: Dict[str, List[str]]
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """Compute Cross-Modal Coherence Score via centroid-to-centroid
        cosine similarity in CLIP's joint latent space.

        FIX v5.0: Rewritten from paired-diagonal to centroid method.
        Text and image corpora are treated as independent collections
        (no pairing assumed). For each period:
          1. Encode ALL texts with CLIP text encoder -> text centroid
          2. Encode ALL images with CLIP image encoder -> image centroid
          3. CMCS_period = cosine_similarity(text_centroid, image_centroid)

        This operationalizes Kapferer's holistic brand identity coherence:
        whether the brand's overall verbal territory occupies similar
        regions in semantic space as its overall visual territory.

        Methodology reference: Radford et al. (2021), Kapferer (2012).

        Returns:
            (normalized_cmcs, raw_cmcs, temporal_coherence, period_coherences)
        """
        period_keys = sorted(texts_by_period.keys())
        period_coherences: Dict[str, float] = {}
        valid_period_scores: List[float] = []

        for period in period_keys:
            texts = texts_by_period.get(period, [])
            image_paths = images_by_period.get(period, [])

            if not texts or not image_paths:
                period_coherences[period] = 0.0
                continue

            try:
                with torch.no_grad():
                    # --- Text centroid via CLIP text encoder ---
                    # Encode ALL texts for this period (no sampling/pairing)
                    text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
                    text_features = self.clip_model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    text_centroid = text_features.mean(dim=0, keepdim=True)
                    text_centroid = text_centroid / text_centroid.norm(dim=-1, keepdim=True)

                    # --- Image centroid via CLIP image encoder ---
                    # Encode ALL images for this period (no sampling/pairing)
                    image_tensors = []
                    for p in image_paths:
                        if not os.path.exists(p):
                            continue
                        try:
                            img = Image.open(p).convert("RGB")
                            image_tensors.append(self.clip_preprocess(img))
                        except Exception:
                            continue

                    if not image_tensors:
                        period_coherences[period] = 0.0
                        continue

                    image_batch = torch.stack(image_tensors).to(self.device)
                    image_features = self.clip_model.encode_image(image_batch)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    image_centroid = image_features.mean(dim=0, keepdim=True)
                    image_centroid = image_centroid / image_centroid.norm(dim=-1, keepdim=True)

                # Cosine similarity between text and image centroids
                period_cmcs = float((text_centroid @ image_centroid.T).item())
                period_coherences[period] = period_cmcs
                valid_period_scores.append(period_cmcs)

            except Exception as e:
                print(f"  Warning: Coherence computation failed for {period}: {e}")
                period_coherences[period] = 0.0
                continue

        if not valid_period_scores:
            return 0.0, 0.0, 0.0, period_coherences

        raw_cmcs = float(np.mean(valid_period_scores))
        norm_cmcs = self._normalize(raw_cmcs, self.norm_params.cmcs_min, self.norm_params.cmcs_max)

        # Temporal Coherence: sample std (ddof=1) of period CMCS scores
        # FIX v5.0: uses ddof=1 consistent with Drift Volatility and
        # methodology section on Bessel's correction
        if len(valid_period_scores) > 2:
            temporal_coherence = float(np.std(valid_period_scores, ddof=1))
        else:
            temporal_coherence = 0.0

        return norm_cmcs, raw_cmcs, temporal_coherence, period_coherences

    # -------------------- Index Composition --------------------

    @staticmethod
    def compute_tli(semantic_drift: float) -> float:
        """TLI = Semantic Drift (normalized). Textual Liquidity Index."""
        return float(semantic_drift)

    @staticmethod
    def compute_vli(visual_drift: float) -> float:
        """VLI = Visual Drift (normalized). Visual Liquidity Index."""
        return float(visual_drift)

    @staticmethod
    def compute_bli(tli: float, vli: float, weights: Tuple[float, float] = (0.5, 0.5)) -> float:
        """BLI = alpha*TLI + beta*VLI. Brand Liquidity Index.
        
        CMCS is deliberately excluded (orthogonal dimension).
        Default weights: alpha = beta = 0.5 (equal weighting).
        """
        return float(weights[0] * tli + weights[1] * vli)

    # -------------------- Second-Order Metrics --------------------

    @staticmethod
    def compute_drift_volatility(similarities: Dict[str, float]) -> float:
        """
        Compute drift volatility as sample standard deviation of period-to-period drift values.

        Uses Bessel's correction (ddof=1) for unbiased estimation of population variance
        from sample data. This aligns with standard statistical practice and financial
        volatility conventions (Engle, 1982; Andersen et al., 2006).

        Args:
            similarities: Dict mapping period transitions (e.g., "2015->2017") to similarity scores

        Returns:
            Sample standard deviation of drift values (1 - similarity).
            Returns 0.0 if fewer than 2 transitions (cannot compute sample std with k=1).

        Note:
            With n time periods, there are k = n-1 transitions. Sample standard deviation
            requires k >= 2 observations, hence minimum 3 time periods for meaningful volatility.
        """
        if not similarities:
            return 0.0
        drifts = [1.0 - s for s in similarities.values()]
        # Need at least 2 values to compute sample standard deviation
        if len(drifts) < 2:
            return 0.0
        return float(np.std(drifts, ddof=1))

    @staticmethod
    def compute_coherence_trajectory(period_coherences: Dict[str, float]) -> float:
        """Compute Coherence Trajectory as OLS slope of period CMCS scores
        against time. Positive slope = improving integration."""
        if len(period_coherences) < 2:
            return 0.0
        periods = sorted(period_coherences.keys())
        x = np.arange(len(periods))
        y = np.array([period_coherences[p] for p in periods])
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])

    @staticmethod
    def compute_drift_ratio(semantic_drift_raw: float, visual_drift_raw: float) -> float:
        """Drift Ratio = raw SD / raw VD. Uses raw (unnormalized) values
        to preserve natural scaling relationship."""
        if visual_drift_raw == 0:
            return 10.0 if semantic_drift_raw > 0 else 0.0
        return float(min(10.0, semantic_drift_raw / visual_drift_raw))

    @staticmethod
    def compute_directional_consistency(period_embeddings: Dict[str, np.ndarray]) -> float:
        """Compute Directional Consistency as mean cosine similarity between
        consecutive drift vectors.
        
        Adapted from directional persistence in movement ecology and
        random-walk theory (Codling et al., 2008).
        """
        periods = sorted(period_embeddings.keys())
        if len(periods) < 3:
            return 0.0

        moves: List[np.ndarray] = []
        for i in range(len(periods) - 1):
            p1, p2 = periods[i], periods[i + 1]
            moves.append((period_embeddings[p2] - period_embeddings[p1]).reshape(1, -1))

        if len(moves) < 2:
            return 0.0

        sims: List[float] = []
        for i in range(len(moves) - 1):
            sim = float(cosine_similarity(moves[i], moves[i + 1])[0, 0])
            sims.append(sim)
        return float(np.mean(sims)) if sims else 0.0

    @staticmethod
    def compute_tsdi(topic_turnover: float, semantic_drift_raw: float) -> float:
        """Topic-Semantic Decoupling Index = Topic Turnover / raw Semantic Drift.
        
        Distinguishes cultural agility (high TT, low SD) from identity
        fragmentation (high TT, high SD). Hamilton et al. (2016) inspired.
        """
        if semantic_drift_raw == 0:
            return 10.0 if topic_turnover > 0 else 0.0
        return float(min(10.0, topic_turnover / semantic_drift_raw))

    @staticmethod
    def _compute_consecutive_cosine_drift(
        period_keys: List[str],
        period_centroids: Dict[str, np.ndarray],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute mean cosine distance (1 - cosine similarity) across consecutive periods.

        Returns:
            raw_drift: float
            similarities: Dict like "2019->2020" -> cosine similarity
        """
        similarities: Dict[str, float] = {}
        drifts: List[float] = []

        for i in range(len(period_keys) - 1):
            p1, p2 = period_keys[i], period_keys[i + 1]
            sim = float(cosine_similarity(period_centroids[p1], period_centroids[p2])[0, 0])
            similarities[f"{p1}->{p2}"] = sim
            drifts.append(1.0 - sim)

        raw_drift = float(np.mean(drifts)) if drifts else 0.0
        return raw_drift, similarities
    # -------------------- Main Analysis Pipeline --------------------

    def analyze_brand(
        self,
        texts_by_period: Dict[str, List[str]],
        images_by_period: Dict[str, List[str]],
        brand_name: str = "Unknown",
    ) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("Brand Liquidity Analyzer v5.0")
        print(f"Analyzing: {brand_name}")
        print("=" * 60)

        results: Dict[str, Any] = {
            "brand_name": brand_name,
            "analyzer_version": "5.0",
            "periods_analyzed": sorted(set(texts_by_period.keys()) | set(images_by_period.keys()))
        }

        # ========== TEXTUAL METRICS ==========
        print("\n[1/6] Computing textual metrics...")
        semantic_drift, semantic_drift_raw, text_similarities, text_embeddings = \
            self.compute_semantic_drift(texts_by_period)
        topic_turnover, topic_turnovers, topic_sets = self.compute_topic_turnover(texts_by_period)
        tone_diversity, tone_diversity_raw = self.compute_tone_diversity(texts_by_period)

        tli = self.compute_tli(semantic_drift)

        results["textual"] = {
            "semantic_drift": semantic_drift,
            "semantic_drift_raw": semantic_drift_raw,
            "topic_turnover": topic_turnover,
            "tone_diversity": tone_diversity,
            "tone_diversity_raw": tone_diversity_raw,
            "tli": tli,
            "period_similarities": text_similarities,
            "period_topic_turnovers": topic_turnovers,
        }

        print(f"   Semantic Drift: {semantic_drift:.3f} (raw: {semantic_drift_raw:.3f})")
        print(f"   Topic Turnover: {topic_turnover:.3f} [diagnostic]")
        print(f"   Tone Diversity: {tone_diversity:.3f}")
        print(f"   TLI: {tli:.3f}")

        # ========== VISUAL METRICS ==========
        print("\n[2/6] Computing visual metrics...")
        visual_drift, visual_drift_raw, visual_similarities, visual_centroids = \
            self.compute_visual_drift(images_by_period)
        color_entropy, color_entropy_raw, period_color_entropies = \
            self.compute_color_entropy(images_by_period)
        motif_diversity, motif_uniqueness, motif_evenness, period_motif_diversity = \
            self.compute_motif_diversity(images_by_period)

        vli = self.compute_vli(visual_drift)

        results["visual"] = {
            "visual_drift": visual_drift,
            "visual_drift_raw": visual_drift_raw,
            "vli": vli,
            "color_entropy": color_entropy,
            "color_entropy_raw": color_entropy_raw,
            "motif_diversity": motif_diversity,
            "motif_uniqueness": motif_uniqueness,
            "motif_evenness": motif_evenness,
            "period_similarities": visual_similarities,
        }

        print(f"   Visual Drift: {visual_drift:.3f} (raw: {visual_drift_raw:.3f})")
        print(f"   VLI: {vli:.3f}")
        print(f"   Color Entropy: {color_entropy:.3f}")
        print(f"   Motif Diversity: {motif_diversity:.3f}")

        # ========== CROSS-MODAL COHERENCE ==========
        print("\n[3/6] Computing cross-modal coherence...")
        cmcs, cmcs_raw, temporal_coherence, period_coherences = \
            self.compute_cross_modal_coherence(texts_by_period, images_by_period)

        results["coherence"] = {
            "cmcs": cmcs,
            "cmcs_raw": cmcs_raw,
            "temporal_coherence": temporal_coherence,
            "period_coherences": period_coherences,
        }

        print(f"   CMCS: {cmcs:.3f} (raw: {cmcs_raw:.3f})")
        print(f"   Temporal Coherence: {temporal_coherence:.4f}")

        # ========== BRAND LIQUIDITY INDEX ==========
        print("\n[4/6] Computing Brand Liquidity Index...")
        bli = self.compute_bli(tli, vli)

        if bli < 0.30:
            category = "Solid"
            description = "Strong continuity across both dimensions"
        elif bli < 0.60:
            category = "Semi-solid"
            description = "Controlled evolution with stable core identity"
        elif bli < 0.80:
            category = "Semi-liquid"
            description = "Balanced adaptation - stable enough for recognition, liquid for relevance"
        else:
            category = "Liquid"
            description = "Highly adaptive, context-responsive, self-reconfiguring identity"

        results["bli"] = {
            "score": bli,
            "category": category,
            "description": description,
            "components": {
                "tli_contribution": 0.5 * tli,
                "vli_contribution": 0.5 * vli,
            },
        }

        print(f"   BLI: {bli:.3f} ({category})")

        # ========== SECOND-ORDER METRICS ==========
        print("\n[5/6] Computing evolution patterns...")
        semantic_drift_volatility = self.compute_drift_volatility(text_similarities)
        visual_drift_volatility = self.compute_drift_volatility(visual_similarities)
        coherence_trajectory = self.compute_coherence_trajectory(period_coherences)
        drift_ratio = self.compute_drift_ratio(semantic_drift_raw, visual_drift_raw)

        print(f"   Semantic Drift Volatility: {semantic_drift_volatility:.3f}")
        print(f"   Visual Drift Volatility: {visual_drift_volatility:.3f}")
        print(f"   Coherence Trajectory: {coherence_trajectory:+.4f}")
        print(f"   Drift Ratio: {drift_ratio:.1f}")

        print("\n[6/6] Computing directional consistency...")
        text_direction = self.compute_directional_consistency(text_embeddings)
        visual_direction = self.compute_directional_consistency(visual_centroids)
        tsdi = self.compute_tsdi(topic_turnover, semantic_drift_raw)

        print(f"   Text Directional Consistency: {text_direction:.3f}")
        print(f"   Visual Directional Consistency: {visual_direction:.3f}")
        print(f"   TSDI: {tsdi:.1f}")

        # Field names matching frontend expectations
        results["second_order"] = {
            "semantic_drift_volatility": semantic_drift_volatility,
            "visual_drift_volatility": visual_drift_volatility,
            "coherence_trajectory": coherence_trajectory,
            "drift_ratio": drift_ratio,
            "directional_consistency_text": text_direction,
            "directional_consistency_visual": visual_direction,
            "topic_semantic_decoupling": tsdi,
        }

        # Summary
        text_count = sum(len(texts) for texts in texts_by_period.values())
        image_count = sum(len(imgs) for imgs in images_by_period.values())

        results["summary"] = {
            "texts_analyzed": text_count,
            "images_analyzed": image_count,
            "n_periods": len(results["periods_analyzed"]),
            "overall_category": category,
        }

        print("\n" + "=" * 60)
        print(f"Analysis Complete: {brand_name}")
        print(f"BLI: {bli:.3f} - {category}")
        print("=" * 60 + "\n")

        return results
