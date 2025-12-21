"""
LLM Analyzer for Brand Fluidity Metrics (v2.0)
Uses Claude API to generate strategic insights from brand fluidity analysis.

Updated for v2 methodology:
- Integrates second-order metrics (drift volatility, coherence trajectory, etc.)
- Aligns with IJRM research framework
- Grounded in peer-reviewed computational methods
"""

import anthropic
import os
from typing import Dict
import json

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
        Generate strategic insights from brand fluidity analysis.
        
        Args:
            analysis_results: Complete analysis results from BrandFluidityAnalyzerV2
            
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
            
            # Call Claude API
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=3500,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
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
    
    def _create_analysis_prompt(self, results: Dict) -> str:
        """Create a detailed prompt for Claude to analyze the metrics."""
        
        brand_name = results.get("brand_name", "Unknown Brand")
        
        # Extract nested results (v2 structure)
        textual = results.get("textual", {})
        visual = results.get("visual", {})
        coherence = results.get("coherence", {})
        bfi = results.get("bfi", {})
        second_order = results.get("second_order", {})
        summary = results.get("summary", {})
        periods = results.get("periods_analyzed", [])
        
        # Core metrics
        tfi = textual.get("tfi", 0)
        semantic_drift = textual.get("semantic_drift", 0)
        tone_diversity = textual.get("tone_diversity", 0)
        topic_turnover = textual.get("topic_turnover", 0)
        
        vfi = visual.get("vfi", 0)
        visual_drift = visual.get("visual_drift", 0)
        color_entropy = visual.get("color_entropy", 0)
        motif_diversity = visual.get("motif_diversity", 0)
        motif_uniqueness = visual.get("motif_uniqueness", 0)
        motif_evenness = visual.get("motif_evenness", 0)
        
        cmcs = coherence.get("cmcs", 0)
        temporal_coherence = coherence.get("temporal_coherence", 0)
        
        bfi_score = bfi.get("score", 0)
        bfi_category = bfi.get("category", "Unknown")
        bfi_description = bfi.get("description", "N/A")
        
        # Second-order metrics
        semantic_drift_volatility = second_order.get("semantic_drift_volatility", 0)
        visual_drift_volatility = second_order.get("visual_drift_volatility", 0)
        coherence_trajectory = second_order.get("coherence_trajectory", 0)
        drift_ratio = second_order.get("drift_ratio", 0)
        directional_consistency_text = second_order.get("directional_consistency_text", 0)
        directional_consistency_visual = second_order.get("directional_consistency_visual", 0)
        topic_semantic_decoupling = second_order.get("topic_semantic_decoupling", 0)
        
        # Period similarities for evolution analysis
        text_period_sims = textual.get("period_similarities", [])
        visual_period_sims = visual.get("period_similarities", [])
        period_coherences = coherence.get("period_coherences", [])
        
        # Format period-by-period evolution
        text_evolution = ""
        if text_period_sims and len(periods) > 1:
            text_evolution = "\n   Period-by-period textual stability:\n"
            for i, sim in enumerate(text_period_sims):
                if i < len(periods) - 1:
                    text_evolution += f"   - {periods[i]} → {periods[i+1]}: {sim:.3f} similarity (drift: {1-sim:.3f})\n"
        
        visual_evolution = ""
        if visual_period_sims and len(periods) > 1:
            visual_evolution = "\n   Period-by-period visual stability:\n"
            for i, sim in enumerate(visual_period_sims):
                if i < len(periods) - 1:
                    visual_evolution += f"   - {periods[i]} → {periods[i+1]}: {sim:.3f} similarity (drift: {1-sim:.3f})\n"
        
        coherence_evolution = ""
        if period_coherences:
            coherence_evolution = "\n   Period-by-period cross-modal coherence:\n"
            for i, coh in enumerate(period_coherences):
                if i < len(periods):
                    coherence_evolution += f"   - {periods[i]}: {coh:.3f}\n"

        prompt = f"""You are a brand strategy analyst specializing in brand identity evolution and liquid modernity theory (Bauman, 2000). Analyze the following Brand Fluidity metrics for {brand_name} and provide strategic insights.

## METHODOLOGY CONTEXT (v2.0)

This analysis uses the **Brand Fluidity Framework** with peer-reviewed computational methods:
- **Sentence-BERT** for semantic embeddings (Reimers & Gurevych, 2019)
- **CLIP** for visual embeddings (Radford et al., 2021)
- **VADER** for sentiment analysis (Hutto & Gilbert, 2014)
- **LDA** for topic modeling (Blei et al., 2003)

**Key insight from preliminary research**: Configuration of fluidity components predicts brand performance better than aggregate fluidity levels. Successful brands exhibit a "visual fortress with narrative flexibility" pattern—low visual drift combined with higher topic turnover.

---

## Brand Fluidity Analysis Results

**Brand:** {brand_name}
**Periods Analyzed:** {', '.join(map(str, periods)) if periods else 'Not specified'}

### Summary
- Texts analyzed: {summary.get('texts_analyzed', 0)}
- Images analyzed: {summary.get('images_analyzed', 0)}
- Time periods: {summary.get('n_periods', 0)}

---

## CORE METRICS

### 1. Textual Fluidity Index (TFI): {tfi:.3f}
   *Measures how much the brand's textual identity changed over time*
   - Semantic Drift: {semantic_drift:.3f}
   - Tone Diversity: {tone_diversity:.3f}
   - Topic Turnover: {topic_turnover:.3f}{text_evolution}

### 2. Visual Fluidity Index (VFI): {vfi:.3f}
   *Measures how much the brand's visual identity changed over time*
   - Visual Drift: {visual_drift:.3f}
   - Color Entropy: {color_entropy:.3f}
   - Motif Diversity: {motif_diversity:.3f}
     - Uniqueness (cluster utilization): {motif_uniqueness:.3f}
     - Evenness (distribution entropy): {motif_evenness:.3f}{visual_evolution}

### 3. Cross-Modal Coherence Score (CMCS): {cmcs:.3f}
   *Measures semantic alignment between text and visual brand expressions*
   - Temporal Coherence: {temporal_coherence:.3f}{coherence_evolution}

### 4. Brand Fluidity Index (BFI): {bfi_score:.3f}
   *Composite: BFI = 0.4×TFI + 0.4×VFI + 0.2×(1-CMCS)*
   - Category: **{bfi_category}**
   - Description: {bfi_description}

---

## SECOND-ORDER METRICS (Pattern & Stability Analysis)

These metrics capture HOW the brand evolves, not just how much:

### Drift Volatility
*Coefficient of variation of drift scores—measures stability of evolution*
- Semantic Drift Volatility: {semantic_drift_volatility:.3f}
- Visual Drift Volatility: {visual_drift_volatility:.3f}
- **Interpretation**: Low volatility = stable, predictable evolution. High volatility = erratic, reactive changes.

### Coherence Trajectory: {coherence_trajectory:+.4f}
*Slope of cross-modal coherence over time*
- **Interpretation**: Positive = improving brand integration. Negative = degrading storytelling coherence.

### Drift Ratio: {drift_ratio:.2f}
*Semantic Drift / Visual Drift*
- **Interpretation**: >1 = messaging evolving faster than visuals. <1 = visuals evolving faster than messaging.
- **Research benchmark**: High-performing brands cluster around 3.0-3.5 (visual anchor with narrative flexibility).

### Directional Consistency
*Whether evolution follows a consistent direction vs. random oscillation*
- Textual: {directional_consistency_text:.3f}
- Visual: {directional_consistency_visual:.3f}
- **Interpretation**: Near 1 = purposeful evolution. Near 0 = random drift. Near -1 = oscillatory repositioning.

### Topic-Semantic Decoupling Index: {topic_semantic_decoupling:.2f}
*Topic Turnover / Semantic Drift*
- **Interpretation**: High TSDI = brand rotates themes while maintaining core semantic identity (cultural agility).
- **Research benchmark**: Growing brands average ~3.5, declining brands ~2.7.

---

## INTERPRETATION FRAMEWORK

| BFI Range | Category | Meaning |
|-----------|----------|---------|
| 0.00-0.30 | **Solid** | Highly consistent identity; minimal change |
| 0.30-0.45 | **Semi-solid** | Strong core with controlled evolution |
| 0.45-0.60 | **Semi-fluid** | Balanced stability and adaptability |
| 0.60-1.00 | **Fluid** | Significant identity transformation |

**Critical pattern**: The "Visual Fortress" configuration (low visual drift + high topic turnover + stable semantic core) characterizes high-performing brands.

---

Please provide your analysis in the following structure:

## 1. Executive Summary
2-3 sentences on what this brand's fluidity profile reveals about their identity strategy and the {len(periods) if periods else 'analyzed'}-period evolution.

## 2. Core Metric Insights
- What do the TFI/VFI/CMCS scores reveal individually?
- Are textual and visual identities evolving in sync or diverging?
- Comment on the topic turnover vs. semantic drift relationship

## 3. Second-Order Pattern Analysis
This is crucial—analyze the PATTERN of evolution:
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

## 6. Comparative Context
- Is this fluidity configuration appropriate for the brand's category?
- How does it compare to the "visual fortress" benchmark pattern?
- Prognosis: Based on second-order metrics, is the brand trajectory healthy or concerning?

Ground your analysis in brand theory, semiotics, and the empirical patterns from the research framework. Focus on actionable insights that connect the metric configuration to strategic brand decisions."""

        return prompt
