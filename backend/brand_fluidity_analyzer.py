"""
Brand Fluidity Analyzer: Core Metric Functions
Implements TFI, VFI, CMCS, and BFI metrics as defined in the research document.
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

class BrandFluidityAnalyzer:
    def __init__(self):
        """Initialize models for analysis."""
        print("Loading models...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Download NLTK data if not present
        try:
            self.sia = SentimentIntensityAnalyzer()
        except:
            import nltk
            nltk.download('vader_lexicon')
            nltk.download('punkt')
            self.sia = SentimentIntensityAnalyzer()
        
        print("Models loaded successfully!")
    
    def calculate_textual_fluidity(self, texts_by_year):
        """
        Calculate Textual Fluidity Index (TFI).
        
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
                "interpretation": "Insufficient data (need 2+ time periods)"
            }
        
        years = sorted(texts_by_year.keys())
        drift_scores = []
        
        # 1. Semantic Drift (SD)
        for i in range(len(years) - 1):
            texts1 = texts_by_year[years[i]]
            texts2 = texts_by_year[years[i+1]]
            
            vec1 = self.sbert_model.encode(" ".join(texts1), convert_to_tensor=True)
            vec2 = self.sbert_model.encode(" ".join(texts2), convert_to_tensor=True)
            
            sim = cosine_similarity(vec1.reshape(1, -1).cpu(), vec2.reshape(1, -1).cpu())[0][0]
            drift_scores.append(1 - sim)
        
        semantic_drift = np.mean(drift_scores)
        
        # 2. Tone Diversity (TD)
        all_sentiments = []
        for year in years:
            for text in texts_by_year[year]:
                for sent in sent_tokenize(text):
                    score = self.sia.polarity_scores(sent)['compound']
                    all_sentiments.append(score)
        
        tone_diversity = np.std(all_sentiments) if all_sentiments else 0.0
        
        # 3. Topic Turnover (TT)
        all_texts = [text for year in texts_by_year for text in texts_by_year[year]]
        
        if len(all_texts) >= 5:
            # Tokenize and prepare for LDA
            tokenized_texts = [text.lower().split() for text in all_texts]
            dictionary = corpora.Dictionary(tokenized_texts)
            dictionary.filter_extremes(no_below=1, no_above=0.9, keep_n=100)
            corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
            
            if len(dictionary) > 0 and len(corpus) > 0:
                lda = models.LdaModel(
                    corpus, 
                    num_topics=min(5, len(all_texts)), 
                    id2word=dictionary,
                    passes=10,
                    random_state=42
                )
                
                # Calculate topic distribution variance across documents
                topic_distributions = []
                for doc in corpus:
                    dist = lda.get_document_topics(doc, minimum_probability=0.0)
                    topic_distributions.append([score for _, score in dist])
                
                topic_turnover = np.std(topic_distributions) if topic_distributions else 0.0
            else:
                topic_turnover = 0.0
        else:
            topic_turnover = 0.0
        
        # 4. Combine into TFI (equal weights)
        TFI = (semantic_drift + tone_diversity + topic_turnover) / 3
        
        # Interpretation
        if TFI < 0.3:
            interpretation = "Solid - consistent tone, topics, and style"
        elif TFI < 0.6:
            interpretation = "Semi-fluid - adaptive language with coherent core"
        else:
            interpretation = "Fluid - high variability, flexible narrative identity"
        
        return {
            "semantic_drift": float(semantic_drift),
            "tone_diversity": float(tone_diversity),
            "topic_turnover": float(topic_turnover),
            "TFI": float(TFI),
            "interpretation": interpretation
        }
    
    def calculate_visual_fluidity(self, image_paths):
        """
        Calculate Visual Fluidity Index (VFI).
        
        Args:
            image_paths: List of paths to image files
            
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
        
        for path in image_paths:
            try:
                # Image embedding with CLIP
                image = self.clip_preprocess(Image.open(path)).unsqueeze(0)
                with torch.no_grad():
                    embedding = self.clip_model.encode_image(image)
                embeddings.append(embedding.squeeze().cpu().numpy())
                
                # Color entropy
                img = cv2.imread(str(path))
                if img is not None:
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
                    hist_norm = hist.ravel() / (hist.sum() + 1e-7)
                    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
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
        
        # Visual Drift (VD) - variance in embeddings
        n_clusters = min(3, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
        visual_drift = np.mean([
            np.linalg.norm(e - kmeans.cluster_centers_[label]) 
            for e, label in zip(embeddings, kmeans.labels_)
        ])
        
        # Normalize visual drift to 0-1 range (heuristic)
        visual_drift = min(1.0, visual_drift / 10.0)
        
        # Color Entropy (CE)
        color_entropy = np.mean(color_entropies) if color_entropies else 0.0
        # Normalize (typical range 0-15, map to 0-1)
        color_entropy_norm = min(1.0, color_entropy / 15.0)
        
        # Motif Diversity (MD) - simplified as embedding diversity
        motif_diversity = np.std(embeddings) / (np.mean(np.abs(embeddings)) + 1e-7)
        motif_diversity = min(1.0, motif_diversity)
        
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
            "motif_diversity": float(motif_diversity),
            "VFI": float(VFI),
            "interpretation": interpretation
        }
    
    def calculate_cross_modal_coherence(self, text_image_pairs):
        """
        Calculate Cross-Modal Coherence Score (CMCS).
        
        Args:
            text_image_pairs: List of (text, image_path) tuples
            
        Returns:
            dict with CMCS score
        """
        if len(text_image_pairs) < 1:
            return {
                "CMCS": 1.0,
                "interpretation": "No paired data available"
            }
        
        similarities = []
        
        for text, image_path in text_image_pairs:
            try:
                # Encode text and image
                text_tokens = clip.tokenize([text]).to(next(self.clip_model.parameters()).device)
                image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(next(self.clip_model.parameters()).device)
                
                with torch.no_grad():
                    text_embed = self.clip_model.encode_text(text_tokens).cpu().numpy()
                    image_embed = self.clip_model.encode_image(image).cpu().numpy()
                
                sim = cosine_similarity(text_embed, image_embed)[0][0]
                similarities.append(sim)
            except Exception as e:
                print(f"Error processing pair ({text[:30]}..., {image_path}): {e}")
                continue
        
        if not similarities:
            return {
                "CMCS": 1.0,
                "interpretation": "Unable to calculate coherence"
            }
        
        # CMCS: Higher = more coherent
        CMCS = np.mean(similarities)
        
        # Interpretation
        if CMCS > 0.7:
            interpretation = "Strong coherence - text and visuals are well-aligned"
        elif CMCS > 0.4:
            interpretation = "Moderate coherence - some alignment between text and visuals"
        else:
            interpretation = "Weak coherence - fragmented or decoupled storytelling"
        
        return {
            "CMCS": float(CMCS),
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)) if len(similarities) > 1 else 0.0,
            "interpretation": interpretation
        }
    
    def compute_brand_fluidity_index(self, TFI, VFI, CMCS, alpha=0.4, beta=0.4, gamma=0.2):
        """
        Compute composite Brand Fluidity Index (BFI).
        
        Args:
            TFI: Textual Fluidity Index score
            VFI: Visual Fluidity Index score
            CMCS: Cross-Modal Coherence Score
            alpha, beta, gamma: Weights (default: 0.4, 0.4, 0.2)
            
        Returns:
            dict with BFI and interpretation
        """
        # BFI formula: α*TFI + β*VFI + γ*(1-CMCS)
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
            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma}
        }
    
    def analyze_brand(self, text_dir, image_dir, brand_name):
        """
        Complete brand fluidity analysis.
        
        Args:
            text_dir: Directory containing text files
            image_dir: Directory containing image files
            brand_name: Name of the brand
            
        Returns:
            Complete analysis results
        """
        results = {
            "brand_name": brand_name,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Load texts
        text_dir = Path(text_dir)
        texts_by_year = defaultdict(list)
        all_texts = []
        
        if text_dir.exists():
            for text_file in text_dir.glob("*"):
                if text_file.is_file() and text_file.suffix in ['.txt', '.md', '.doc', '.docx']:
                    try:
                        content = text_file.read_text(encoding='utf-8', errors='ignore')
                        # Try to extract year from filename
                        filename = text_file.stem
                        year = None
                        for part in filename.split('_'):
                            if part.isdigit() and len(part) == 4:
                                year = int(part)
                                break
                        
                        if year:
                            texts_by_year[year].append(content)
                        else:
                            texts_by_year[2024].append(content)  # Default year
                        
                        all_texts.append(content)
                    except Exception as e:
                        print(f"Error reading {text_file}: {e}")
        
        # Load images
        image_dir = Path(image_dir)
        image_paths = []
        
        if image_dir.exists():
            for img_file in image_dir.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                    image_paths.append(str(img_file))
        
        # Calculate metrics
        print(f"Analyzing {len(all_texts)} texts and {len(image_paths)} images...")
        
        # TFI
        tfi_results = self.calculate_textual_fluidity(texts_by_year)
        results["textual_fluidity"] = tfi_results
        
        # VFI
        vfi_results = self.calculate_visual_fluidity(image_paths)
        results["visual_fluidity"] = vfi_results
        
        # CMCS - pair texts with images (simple pairing by order)
        text_image_pairs = []
        if all_texts and image_paths:
            for i, (text, img_path) in enumerate(zip(all_texts, image_paths)):
                # Take first 200 chars of text as caption
                caption = text[:200] if len(text) > 200 else text
                text_image_pairs.append((caption, img_path))
        
        cmcs_results = self.calculate_cross_modal_coherence(text_image_pairs)
        results["cross_modal_coherence"] = cmcs_results
        
        # BFI
        bfi_results = self.compute_brand_fluidity_index(
            TFI=tfi_results["TFI"],
            VFI=vfi_results["VFI"],
            CMCS=cmcs_results["CMCS"]
        )
        results["brand_fluidity_index"] = bfi_results
        
        # Summary
        results["summary"] = {
            "texts_analyzed": len(all_texts),
            "images_analyzed": len(image_paths),
            "time_periods": len(texts_by_year),
            "overall_category": bfi_results["category"]
        }
        
        return results
