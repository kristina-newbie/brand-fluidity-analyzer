"""
Brand Fluidity Analyzer v4.0
Temporal Fluidity & Coherence Measurement

v4 Changes:
- TFI = Semantic Drift only (Topic Turnover is diagnostic)
- VFI = Visual Drift only
- BFI = 0.5*TFI + 0.5*VFI (CMCS is separate axis, not in BFI)
"""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from collections import defaultdict

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
    - CMCS: Cross-modal similarities typically 0.15-0.32
    """
    cmcs_min: float = 0.15
    cmcs_max: float = 0.32
    semantic_drift_min: float = 0.05
    semantic_drift_max: float = 0.25
    visual_drift_min: float = 0.02
    visual_drift_max: float = 0.12


class BrandFluidityAnalyzerV2:
    """
    Brand Fluidity Analyzer v4.0

    Core indices:
    - TFI = Semantic Drift (normalized)
    - VFI = Visual Drift (normalized)
    - BFI = 0.5*TFI + 0.5*VFI

    CMCS is reported separately (not in BFI).
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
        if not texts:
            return np.zeros((1, self.sbert_model.get_sentence_embedding_dimension()))
        return self.sbert_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def compute_semantic_drift(
        self, texts_by_period: Dict[str, List[str]]
    ) -> Tuple[float, float, Dict[str, float], Dict[str, np.ndarray]]:
        period_keys = sorted(texts_by_period.keys())
        period_embeddings: Dict[str, np.ndarray] = {}

        for period in period_keys:
            texts = texts_by_period[period]
            if not texts:
                emb = np.zeros((1, self.sbert_model.get_sentence_embedding_dimension()))
            else:
                emb = self.embed_texts(texts).mean(axis=0, keepdims=True)
            period_embeddings[period] = emb

        similarities: Dict[str, float] = {}
        drifts: List[float] = []
        for i in range(len(period_keys) - 1):
            p1, p2 = period_keys[i], period_keys[i + 1]
            sim = float(cosine_similarity(period_embeddings[p1], period_embeddings[p2])[0, 0])
            similarities[f"{p1}->{p2}"] = sim
            drifts.append(1.0 - sim)

        raw_drift = float(np.mean(drifts)) if drifts else 0.0
        norm_drift = self._normalize(raw_drift, self.norm_params.semantic_drift_min, self.norm_params.semantic_drift_max)
        return norm_drift, raw_drift, similarities, period_embeddings

    # -------------------- Topic Turnover (Diagnostic) --------------------

    @staticmethod
    def _tokenize_for_topic(text: str) -> List[str]:
        return [t.lower() for t in text.split() if len(t) > 2]

    def compute_topic_turnover(
        self, texts_by_period: Dict[str, List[str]], num_topics: int = 5
    ) -> Tuple[float, Dict[str, float], Dict[str, List[str]]]:
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
        period_keys = sorted(images_by_period.keys())
        period_centroids: Dict[str, np.ndarray] = {}

        for period in period_keys:
            paths = images_by_period[period]
            emb = self.embed_images(paths)
            centroid = emb.mean(axis=0, keepdims=True)
            period_centroids[period] = centroid

        similarities: Dict[str, float] = {}
        drifts: List[float] = []
        for i in range(len(period_keys) - 1):
            p1, p2 = period_keys[i], period_keys[i + 1]
            sim = float(cosine_similarity(period_centroids[p1], period_centroids[p2])[0, 0])
            similarities[f"{p1}->{p2}"] = sim
            drifts.append(1.0 - sim)

        raw_drift = float(np.mean(drifts)) if drifts else 0.0
        norm_drift = self._normalize(raw_drift, self.norm_params.visual_drift_min, self.norm_params.visual_drift_max)
        return norm_drift, raw_drift, similarities, period_centroids

    # -------------------- Color Entropy (Descriptive) --------------------

    def compute_color_entropy(
        self, images_by_period: Dict[str, List[str]], num_colors: int = 5
    ) -> Tuple[float, float, Dict[str, float]]:
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

            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            counts = np.bincount(labels, minlength=num_colors)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            norm_entropy = float(entropy / np.log(num_colors))
            period_color_entropies[period] = norm_entropy
            all_colors.append(pixels[np.random.choice(len(pixels), size=min(1000, len(pixels)), replace=False)])

        if not all_colors:
            return 0.0, 0.0, period_color_entropies

        all_colors_arr = np.vstack(all_colors)
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(all_colors_arr)
        counts = np.bincount(labels, minlength=num_colors)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        raw_entropy = -np.sum(probs * np.log(probs))
        norm_entropy = float(raw_entropy / np.log(num_colors))
        return norm_entropy, raw_entropy, period_color_entropies

    # -------------------- Motif Diversity (Descriptive) --------------------

    def compute_motif_diversity(
        self, images_by_period: Dict[str, List[str]], num_clusters: int = 8
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """Returns (diversity, uniqueness, evenness, period_diversity)"""
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
        period_keys = sorted(texts_by_period.keys())
        period_coherences: Dict[str, float] = {}
        all_sims: List[float] = []

        for period in period_keys:
            texts = texts_by_period[period]
            image_paths = images_by_period.get(period, [])
            if not texts or not image_paths:
                period_coherences[period] = 0.0
                continue

            n_pairs = min(len(texts), len(image_paths), 20)
            texts_sample = texts[:n_pairs]
            images_sample = image_paths[:n_pairs]

            try:
                with torch.no_grad():
                    # Truncate texts to avoid CLIP token limit
                    texts_truncated = [t[:77] for t in texts_sample]
                    text_tokens = clip.tokenize(texts_truncated, truncate=True).to(self.device)
                    text_features = self.clip_model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    image_tensors = []
                    for p in images_sample:
                        if not os.path.exists(p):
                            continue
                        img = Image.open(p).convert("RGB")
                        image_tensors.append(self.clip_preprocess(img))

                    if not image_tensors:
                        period_coherences[period] = 0.0
                        continue

                    image_batch = torch.stack(image_tensors).to(self.device)
                    image_features = self.clip_model.encode_image(image_batch)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                n_valid = min(text_features.shape[0], image_features.shape[0])
                sims = (text_features[:n_valid] @ image_features[:n_valid].T).diag().cpu().numpy()
                period_mean = float(np.mean(sims))
                period_coherences[period] = period_mean
                all_sims.extend(sims.tolist())
            except Exception as e:
                print(f"  Warning: Coherence computation failed for {period}: {e}")
                period_coherences[period] = 0.0
                continue

        if not all_sims:
            return 0.0, 0.0, 0.0, period_coherences

        raw_cmcs = float(np.mean(all_sims))
        norm_cmcs = self._normalize(raw_cmcs, self.norm_params.cmcs_min, self.norm_params.cmcs_max)
        temporal_coherence = float(np.std(list(period_coherences.values()))) if len(period_coherences) > 1 else 0.0

        return norm_cmcs, raw_cmcs, temporal_coherence, period_coherences

    # -------------------- Index Composition (v4) --------------------

    def compute_tfi(self, semantic_drift: float) -> float:
        """v4: TFI = Semantic Drift (normalized)"""
        return float(semantic_drift)

    def compute_vfi(self, visual_drift: float) -> float:
        """v4: VFI = Visual Drift (normalized)"""
        return float(visual_drift)

    def compute_bfi(self, tfi: float, vfi: float, weights: Tuple[float, float] = (0.5, 0.5)) -> float:
        """v4: BFI = alpha*TFI + beta*VFI (CMCS not included)"""
        return float(weights[0] * tfi + weights[1] * vfi)

    # -------------------- Second-Order Metrics --------------------

    @staticmethod
    def compute_drift_volatility(similarities: Dict[str, float]) -> float:
        if not similarities:
            return 0.0
        drifts = [1.0 - s for s in similarities.values()]
        return float(np.std(drifts))

    @staticmethod
    def compute_coherence_trajectory(period_coherences: Dict[str, float]) -> float:
        if len(period_coherences) < 2:
            return 0.0
        periods = sorted(period_coherences.keys())
        x = np.arange(len(periods))
        y = np.array([period_coherences[p] for p in periods])
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])

    @staticmethod
    def compute_drift_ratio(semantic_drift_raw: float, visual_drift_raw: float) -> float:
        if visual_drift_raw == 0:
            return 10.0 if semantic_drift_raw > 0 else 0.0
        return float(min(10.0, semantic_drift_raw / visual_drift_raw))

    @staticmethod
    def compute_directional_consistency(period_embeddings: Dict[str, np.ndarray]) -> float:
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
        if semantic_drift_raw == 0:
            return 10.0 if topic_turnover > 0 else 0.0
        return float(min(10.0, topic_turnover / semantic_drift_raw))

    # -------------------- Main Analysis Pipeline --------------------

    def analyze_brand(
        self,
        texts_by_period: Dict[str, List[str]],
        images_by_period: Dict[str, List[str]],
        brand_name: str = "Unknown",
    ) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("Brand Fluidity Analyzer v4.0")
        print(f"Analyzing: {brand_name}")
        print("=" * 60)

        results: Dict[str, Any] = {
            "brand_name": brand_name,
            "analyzer_version": "4.0",
            "periods_analyzed": sorted(set(texts_by_period.keys()) | set(images_by_period.keys()))
        }

        # ========== TEXTUAL METRICS ==========
        print("\n[1/6] Computing textual metrics...")
        semantic_drift, semantic_drift_raw, text_similarities, text_embeddings = \
            self.compute_semantic_drift(texts_by_period)
        topic_turnover, topic_turnovers, topic_sets = self.compute_topic_turnover(texts_by_period)
        tone_diversity, tone_diversity_raw = self.compute_tone_diversity(texts_by_period)

        tfi = self.compute_tfi(semantic_drift)

        results["textual"] = {
            "semantic_drift": semantic_drift,
            "semantic_drift_raw": semantic_drift_raw,
            "topic_turnover": topic_turnover,
            "tone_diversity": tone_diversity,
            "tone_diversity_raw": tone_diversity_raw,
            "tfi": tfi,
            "period_similarities": text_similarities,
            "period_topic_turnovers": topic_turnovers,
        }

        print(f"   Semantic Drift: {semantic_drift:.3f} (raw: {semantic_drift_raw:.3f})")
        print(f"   Topic Turnover: {topic_turnover:.3f} [diagnostic]")
        print(f"   Tone Diversity: {tone_diversity:.3f}")
        print(f"   TFI: {tfi:.3f}")

        # ========== VISUAL METRICS ==========
        print("\n[2/6] Computing visual metrics...")
        visual_drift, visual_drift_raw, visual_similarities, visual_centroids = \
            self.compute_visual_drift(images_by_period)
        color_entropy, color_entropy_raw, period_color_entropies = \
            self.compute_color_entropy(images_by_period)
        motif_diversity, motif_uniqueness, motif_evenness, period_motif_diversity = \
            self.compute_motif_diversity(images_by_period)

        vfi = self.compute_vfi(visual_drift)

        results["visual"] = {
            "visual_drift": visual_drift,
            "visual_drift_raw": visual_drift_raw,
            "vfi": vfi,
            "color_entropy": color_entropy,
            "color_entropy_raw": color_entropy_raw,
            "motif_diversity": motif_diversity,
            "motif_uniqueness": motif_uniqueness,
            "motif_evenness": motif_evenness,
            "period_similarities": visual_similarities,
        }

        print(f"   Visual Drift: {visual_drift:.3f} (raw: {visual_drift_raw:.3f})")
        print(f"   VFI: {vfi:.3f}")
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

        # ========== BRAND FLUIDITY INDEX ==========
        print("\n[4/6] Computing Brand Fluidity Index...")
        bfi = self.compute_bfi(tfi, vfi)

        if bfi < 0.30:
            category = "Solid"
            description = "Strong continuity across both dimensions"
        elif bfi < 0.60:
            category = "Semi-solid"
            description = "Controlled evolution with stable core identity"
        elif bfi < 0.80:
            category = "Semi-fluid"
            description = "Balanced adaptation - stable enough for recognition, fluid for relevance"
        else:
            category = "Fluid"
            description = "Highly adaptive, context-responsive, self-reconfiguring identity"

        results["bfi"] = {
            "score": bfi,
            "category": category,
            "description": description,
            "components": {
                "tfi_contribution": 0.5 * tfi,
                "vfi_contribution": 0.5 * vfi,
            },
        }

        print(f"   BFI: {bfi:.3f} ({category})")

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
        print(f"BFI: {bfi:.3f} - {category}")
        print("=" * 60 + "\n")

        return results
