"""
LLM Analyzer for Brand Fluidity Metrics (v3)
Uses Claude API to generate strategic insights from temporal self-similarity analysis.

Updated to match v3 methodology:
- Metrics now measure temporal CHANGE (not content diversity)
- TFI/VFI = 1 - temporal stability (higher = more change over time)
- New interpretation thresholds aligned with academic framework
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
            analysis_results: Complete analysis results from BrandFluidityAnalyzer
            
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
                max_tokens=2500,
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
        tfi = results.get("textual_fluidity", {})
        vfi = results.get("visual_fluidity", {})
        cmcs = results.get("cross_modal_coherence", {})
        bfi = results.get("brand_fluidity_index", {})
        summary = results.get("summary", {})
        
        # Get temporal details if available
        years_covered = summary.get('years_covered', [])
        text_periods = tfi.get('period_similarities', [])
        visual_periods = vfi.get('period_similarities', [])
        
        # Format period-by-period changes if available
        text_evolution = ""
        if text_periods:
            text_evolution = "\n   Period-by-period textual stability:\n"
            for p in text_periods:
                text_evolution += f"   - {p['period']}: {p['similarity']:.3f} similarity\n"
        
        visual_evolution = ""
        if visual_periods:
            visual_evolution = "\n   Period-by-period visual stability:\n"
            for p in visual_periods:
                visual_evolution += f"   - {p['period']}: {p['similarity']:.3f} similarity\n"

        prompt = f"""You are a brand strategy analyst specializing in brand identity evolution and liquid modernity theory (Bauman, 2000). Analyze the following Brand Fluidity metrics for {brand_name} and provide strategic insights.

## IMPORTANT: Methodology Context

This analysis uses **Temporal Self-Similarity** methodology:
- **Fluidity = 1 - Stability**: We measure how consistent the brand is with ITSELF across different time periods
- **Low fluidity scores** indicate the brand maintained consistent identity over time (SOLID)
- **High fluidity scores** indicate the brand's identity CHANGED significantly between time periods (FLUID)
- This is NOT about content diversity within a snapshot—it's about temporal evolution

## Brand Fluidity Analysis Results

**Brand:** {brand_name}

### Summary
- Texts analyzed: {summary.get('texts_analyzed', 0)}
- Images analyzed: {summary.get('images_analyzed', 0)}
- Time periods covered: {summary.get('time_periods', 0)}
- Years analyzed: {', '.join(map(str, years_covered)) if years_covered else 'Not specified'}
- Overall category: {summary.get('overall_category', 'Unknown')}

### 1. Textual Fluidity Index (TFI): {tfi.get('TFI', 0):.3f}
   *Measures how much the brand's textual identity CHANGED over time*
   - Temporal Textual Stability: {tfi.get('temporal_stability', 1 - tfi.get('TFI', 0)):.3f}
   - Semantic Drift: {tfi.get('semantic_drift', 0):.3f}
   - Tone/Personality Shift: {tfi.get('tone_diversity', 0):.3f}
   - Interpretation: {tfi.get('interpretation', 'N/A')}{text_evolution}

### 2. Visual Fluidity Index (VFI): {vfi.get('VFI', 0):.3f}
   *Measures how much the brand's visual identity CHANGED over time*
   - Temporal Visual Stability: {vfi.get('temporal_stability', 1 - vfi.get('VFI', 0)):.3f}
   - Visual Drift: {vfi.get('visual_drift', 0):.3f}
   - Color Palette Stability: {vfi.get('color_stability', 'N/A')}
   - Interpretation: {vfi.get('interpretation', 'N/A')}{visual_evolution}

### 3. Cross-Modal Coherence Score (CMCS): {cmcs.get('CMCS', 0):.3f}
   *Measures semantic alignment between text and visual brand expressions*
   - Centroid Similarity: {cmcs.get('centroid_similarity', 'N/A')}
   - Interpretation: {cmcs.get('interpretation', 'N/A')}

### 4. Brand Fluidity Index (BFI): {bfi.get('BFI', 0):.3f}
   *Composite measure: BFI = 0.4×TFI + 0.4×VFI + 0.2×(1-CMCS)*
   - Category: {bfi.get('category', 'Unknown')}
   - Description: {bfi.get('description', 'N/A')}

---

## Interpretation Framework (Temporal Self-Similarity)

| BFI Range | Category | Meaning |
|-----------|----------|---------|
| 0.00-0.20 | **Solid** | Highly consistent identity over time; minimal temporal change |
| 0.20-0.35 | **Semi-solid** | Strong identity core with controlled, strategic evolution |
| 0.35-0.50 | **Semi-fluid** | Balanced evolution; stable recognition with adaptive relevance |
| 0.50-1.00 | **Fluid** | Significant identity changes; transformation or governance issues |

**Key insight**: For heritage/luxury brands, BFI < 0.25 is expected. For fashion-forward brands, BFI 0.30-0.45 may be optimal. BFI > 0.50 often indicates brand management challenges rather than strategic fluidity.

---

Please provide your analysis in the following structure:

## 1. Executive Summary
2-3 sentences on what this brand's temporal fluidity profile reveals about their identity strategy and evolution over the analyzed period ({', '.join(map(str, years_covered)) if years_covered else 'the time period'}).

## 2. Key Insights
3-5 bullet points covering:
- What do the stability/change patterns reveal about brand management?
- Are text and visual identities evolving in sync or diverging?
- What does the cross-modal coherence tell us about integrated brand storytelling?

## 3. Strategic Implications
- How does this temporal evolution pattern position the brand competitively?
- What are the risks of the current fluidity level?
- What opportunities exist?

## 4. Recommendations
3-4 actionable suggestions:
- Should the brand increase or decrease temporal consistency in specific areas?
- How can cross-modal coherence be improved if needed?
- What governance or creative direction changes might help?

## 5. Comparative Context
- How does this profile compare to typical patterns for this brand category?
- Is the fluidity level appropriate for the brand's positioning (heritage vs. contemporary)?
- Note any concerning divergences between textual and visual evolution rates

Ground your analysis in brand theory, semiotics, and the temporal self-similarity framework. Focus on actionable insights that connect the metrics to strategic brand decisions."""

        return prompt
