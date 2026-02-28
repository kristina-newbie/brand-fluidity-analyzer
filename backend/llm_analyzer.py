"""
LLM Analyzer for Brand Liquidity Metrics (v5.0)
Uses Claude API to generate strategic insights from brand liquidity analysis.

Updated for v5 methodology:
- Terminology aligned with Bauman's (2000) liquid modernity: Liquidity not Fluidity
- TLI (Textual Liquidity Index), VLI (Visual Liquidity Index), BLI (Brand Liquidity Index)
- CMCS computed via centroid-to-centroid cosine similarity in CLIP joint space
- Integrates second-order metrics (drift volatility, coherence trajectory, etc.)
- Aligns with IJRM research framework
- Grounded in peer-reviewed computational methods
"""

import anthropic
from anthropic.types import MessageParam
import os
from typing import Dict

class LLMAnalyzer:
    def __init__(self, api_key: str = None):
        """Initialize Claude API client."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            print("Warning: No ANTHROPIC_API_KEY found. LLM insights will be unavailable.")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=self.api_key)
    
    async def generate_insights(self, analysis_results: Dict) -> Dict:
        """
        Generate strategic insights from brand liquidity analysis.
        
        Args:
            analysis_results: Complete analysis results from BrandLiquidityAnalyzer
            
        Returns:
            Dict containing insights, recommendations, and strategic analysis
        """
        if not self.client:
            return {
                "error": "Claude API key not configured",
                "insights": "LLM insights unavailable. Please set ANTHROPIC_API_KEY environment variable."
            }
        
        try:
            # Prepare the analysis data for Claude
            prompt = self._create_analysis_prompt(analysis_results)
            messages: list[MessageParam] = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            # Call Claude API
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=3500,
                messages=messages
            )
            
            # Parse the response
            insights_text = message.content[0].text
            
            # Structure the insights
            return {
                "insights": insights_text,
                "model": "claude-sonnet-4-5",
                "timestamp": analysis_results.get("timestamp")
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "insights": f"Error generating insights: {str(e)}"
            }

    @staticmethod
    def _safe_float(value, default=0.0) -> float:
        """Safely convert value to float."""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def _create_analysis_prompt(self, results: Dict) -> str:
        """Create a detailed prompt for Claude to analyze the metrics."""
        
        brand_name = results.get("brand_name", "Unknown Brand")
        
        # Extract nested results (v5 structure)
        textual = results.get("textual", {})
        visual = results.get("visual", {})
        coherence = results.get("coherence", {})
        bli = results.get("bli", {})
        second_order = results.get("second_order", {})
        summary = results.get("summary", {})
        periods = results.get("periods_analyzed", [])
        
        # Core metrics - safely convert to float
        tli = self._safe_float(textual.get("tli"))
        semantic_drift = self._safe_float(textual.get("semantic_drift"))
        tone_diversity = self._safe_float(textual.get("tone_diversity"))
        topic_turnover = self._safe_float(textual.get("topic_turnover"))
        
        vli = self._safe_float(visual.get("vli"))
        visual_drift = self._safe_float(visual.get("visual_drift"))
        color_entropy = self._safe_float(visual.get("color_entropy"))
        motif_diversity = self._safe_float(visual.get("motif_diversity"))
        motif_uniqueness = self._safe_float(visual.get("motif_uniqueness"))
        motif_evenness = self._safe_float(visual.get("motif_evenness"))
        
        cmcs = self._safe_float(coherence.get("cmcs"))
        temporal_coherence = self._safe_float(coherence.get("temporal_coherence"))
        
        bli_score = self._safe_float(bli.get("score"))
        bli_category = bli.get("category", "Unknown")
        bli_description = bli.get("description", "N/A")
        
        # Second-order metrics
        semantic_drift_volatility = self._safe_float(second_order.get("semantic_drift_volatility"))
        visual_drift_volatility = self._safe_float(second_order.get("visual_drift_volatility"))
        coherence_trajectory = self._safe_float(second_order.get("coherence_trajectory"))
        drift_ratio = self._safe_float(second_order.get("drift_ratio"))
        directional_consistency_text = self._safe_float(second_order.get("directional_consistency_text"))
        directional_consistency_visual = self._safe_float(second_order.get("directional_consistency_visual"))
        topic_semantic_decoupling = self._safe_float(second_order.get("topic_semantic_decoupling"))
        
        # Period similarities for evolution analysis (these are DICTS, not lists)
        text_period_sims = textual.get("period_similarities", {})
        visual_period_sims = visual.get("period_similarities", {})
        period_coherences = coherence.get("period_coherences", {})
        
        # Format period-by-period evolution from dict format {"2015->2017": 0.85, ...}
        text_evolution = ""
        if isinstance(text_period_sims, dict) and text_period_sims:
            text_evolution = "\n   Period-by-period textual stability:\n"
            for transition, sim in sorted(text_period_sims.items()):
                sim_val = self._safe_float(sim)
                text_evolution += f"   - {transition}: {sim_val:.3f} similarity (drift: {1-sim_val:.3f})\n"
        
        visual_evolution = ""
        if isinstance(visual_period_sims, dict) and visual_period_sims:
            visual_evolution = "\n   Period-by-period visual stability:\n"
            for transition, sim in sorted(visual_period_sims.items()):
                sim_val = self._safe_float(sim)
                visual_evolution += f"   - {transition}: {sim_val:.3f} similarity (drift: {1-sim_val:.3f})\n"
        
        coherence_evolution = ""
        if isinstance(period_coherences, dict) and period_coherences:
            coherence_evolution = "\n   Period-by-period cross-modal coherence:\n"
            for period, coh in sorted(period_coherences.items()):
                coh_val = self._safe_float(coh)
                coherence_evolution += f"   - {period}: {coh_val:.3f}\n"

        prompt = f"""You are a brand strategy analyst specializing in brand identity evolution and liquid modernity theory (Bauman, 2000). Analyze the following Brand Liquidity metrics for {brand_name} and provide strategic insights.

## METHODOLOGY CONTEXT (v5.0)

This analysis uses the **Brand Liquidity Framework** grounded in Bauman's (2000) liquid modernity theory, with peer-reviewed computational methods:
- **Sentence-BERT** for semantic embeddings (Reimers & Gurevych, 2019)
- **CLIP** for visual embeddings and cross-modal coherence (Radford et al., 2021)
- **VADER** for sentiment analysis (Hutto & Gilbert, 2014)
- **LDA** for topic modeling (Blei et al., 2003)

**v5 Methodology**:
- TLI = Semantic Drift only (Topic Turnover is diagnostic, not part of TLI)
- VLI = Visual Drift only
- BLI = 0.5 x TLI + 0.5 x VLI (CMCS is a separate coherence axis, not part of BLI)
- CMCS = Centroid-to-centroid cosine similarity in CLIP's joint latent space (holistic brand territory alignment, not paired text-image matching)

**Key insight from preliminary research**: Configuration of liquidity components predicts brand performance better than aggregate liquidity levels. Successful brands exhibit a "visual fortress with narrative flexibility" pattern — low visual drift combined with higher topic turnover.

---

## Brand Liquidity Analysis Results

**Brand:** {brand_name}
**Periods Analyzed:** {', '.join(map(str, periods)) if periods else 'Not specified'}

### Summary
- Texts analyzed: {summary.get('texts_analyzed', 0)}
- Images analyzed: {summary.get('images_analyzed', 0)}
- Time periods: {summary.get('n_periods', 0)}

---

## CORE METRICS

### 1. Textual Liquidity Index (TLI): {tli:.3f}
   *Measures how much the brand's semantic identity changed over time*
   *TLI = Semantic Drift (normalized)*
   - Semantic Drift: {semantic_drift:.3f}
   - Tone Diversity: {tone_diversity:.3f} (descriptive, not in TLI)
   - Topic Turnover: {topic_turnover:.3f} (diagnostic, not in TLI){text_evolution}

### 2. Visual Liquidity Index (VLI): {vli:.3f}
   *Measures how much the brand's visual identity changed over time*
   *VLI = Visual Drift (normalized)*
   - Visual Drift: {visual_drift:.3f}
   - Color Entropy: {color_entropy:.3f} (descriptive, not in VLI)
   - Motif Diversity: {motif_diversity:.3f} (descriptive, not in VLI)
     - Uniqueness (cluster utilization): {motif_uniqueness:.3f}
     - Evenness (distribution entropy): {motif_evenness:.3f}{visual_evolution}

### 3. Cross-Modal Coherence Score (CMCS): {cmcs:.3f}
   *Measures holistic semantic alignment between textual and visual brand territories*
   *Centroid-to-centroid cosine similarity in CLIP joint space (Kapferer's Brand Identity Prism)*
   *Separate axis — not included in BLI*
   - Temporal Coherence: {temporal_coherence:.3f}{coherence_evolution}

### 4. Brand Liquidity Index (BLI): {bli_score:.3f}
   *Composite: BLI = 0.5 x TLI + 0.5 x VLI*
   - Category: **{bli_category}**
   - Description: {bli_description}

---

## SECOND-ORDER METRICS (Pattern & Stability Analysis)

These metrics capture HOW the brand evolves, not just how much:

### Drift Volatility
*Standard deviation of drift scores — measures stability of evolution*
- Semantic Drift Volatility: {semantic_drift_volatility:.3f}
- Visual Drift Volatility: {visual_drift_volatility:.3f}
- **Interpretation**: Low volatility = stable, predictable evolution. High volatility = erratic, reactive changes.

### Coherence Trajectory: {coherence_trajectory:+.4f}
*Slope of cross-modal coherence over time*
- **Interpretation**: Positive = improving brand integration. Negative = degrading storytelling coherence.

### Drift Ratio: {drift_ratio:.1f}
*Semantic Drift (raw) / Visual Drift (raw)*
- **Interpretation**: >1 = messaging evolving faster than visuals. <1 = visuals evolving faster than messaging.
- **Research benchmark**: High-performing brands cluster around 3.0-3.5 (visual anchor with narrative flexibility).

### Directional Consistency
*Whether evolution follows a consistent direction vs. random oscillation*
- Textual: {directional_consistency_text:.3f}
- Visual: {directional_consistency_visual:.3f}
- **Interpretation**: Near +1 = purposeful evolution. Near 0 = random drift. Near -1 = oscillatory repositioning.

### Topic-Semantic Decoupling Index (TSDI): {topic_semantic_decoupling:.1f}
*Topic Turnover / Semantic Drift (raw)*
- **Interpretation**: High TSDI = brand rotates themes while maintaining core semantic identity (cultural agility).
- **Research benchmark**: Growing brands average ~3.5, declining brands ~2.7.

---

## INTERPRETATION FRAMEWORK

| BLI Range | Category | Meaning |
|-----------|----------|---------|
| 0.00-0.30 | **Solid** | Highly consistent identity; minimal change |
| 0.30-0.60 | **Semi-solid** | Strong core with controlled evolution |
| 0.60-0.80 | **Semi-liquid** | Balanced stability and adaptability |
| 0.80-1.00 | **Liquid** | Significant identity transformation |

**Critical pattern**: The "Visual Fortress" configuration (low visual drift + high topic turnover + stable semantic core) characterizes high-performing brands.

---

Please provide your analysis in the following structure:

## 1. Executive Summary
2-3 sentences on what this brand's liquidity profile reveals about their identity strategy and the {len(periods) if periods else 'analyzed'}-period evolution.

## 2. Core Metric Insights
- What do the TLI/VLI/CMCS scores reveal individually?
- Are textual and visual identities evolving in sync or diverging?
- Comment on the topic turnover vs. semantic drift relationship (TSDI)

## 3. Second-Order Pattern Analysis
This is crucial — analyze the PATTERN of evolution:
- Is the brand's evolution stable and controlled (low drift volatility) or erratic?
- What does the coherence trajectory suggest about brand management quality?
- Does the drift ratio indicate "visual fortress" strategy or visual instability?
- Is directional consistency suggesting purposeful repositioning or reactive drift?
- What does TSDI reveal about cultural agility vs. identity fragmentation?

## 4. Strategic Implications
- How does this pattern position the brand competitively?
- What are the specific risks of the current configuration?
- What opportunities exist?

## 5. Actionable Recommendations
5-6 specific suggestions addressing:
- Which components need tightening vs. loosening?
- How to improve cross-modal coherence if needed?
- Visual identity governance recommendations
- Messaging/topic strategy recommendations
- Specific period-based interventions if evolution patterns show concerning shifts

Ground your analysis in brand theory, semiotics, and the empirical patterns from the research framework. Focus on actionable insights that connect the metric configuration to strategic brand decisions."""

        return prompt
