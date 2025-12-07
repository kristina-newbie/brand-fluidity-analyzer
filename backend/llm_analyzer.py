"""
LLM Analyzer for Brand Fluidity Metrics (v4)
Uses Claude API to generate strategic insights from temporal self-similarity analysis.

Updated for v4 methodology alignment:
- Leverages Jaccard-based topic turnover details
- Includes clustering-based motif diversity stats
- Uses temporal coherence component from CMCS
- All metrics grounded in peer-reviewed methodology
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
                max_tokens=3000,
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
        
        # Get temporal details
        years_covered = summary.get('years_covered', [])
        text_periods = tfi.get('period_similarities', [])
        visual_periods = vfi.get('period_similarities', [])
        
        # NEW v4: Topic turnover details (Jaccard-based)
        topic_turnover_details = tfi.get('topic_turnover_details', [])
        
        # NEW v4: Cluster stats for motif diversity
        cluster_stats = vfi.get('cluster_stats', {})
        
        # NEW v4: Temporal coherence from CMCS
        temporal_coherence = cmcs.get('temporal_coherence')
        temporal_coherence_by_year = cmcs.get('temporal_coherence_by_year', [])
        
        # Format period-by-period textual stability
        text_evolution = ""
        if text_periods:
            text_evolution = "\n   Period-by-period textual stability:\n"
            for p in text_periods:
                text_evolution += f"   - {p['period']}: {p['similarity']:.3f} similarity\n"
        
        # Format period-by-period visual stability
        visual_evolution = ""
        if visual_periods:
            visual_evolution = "\n   Period-by-period visual stability:\n"
            for p in visual_periods:
                visual_evolution += f"   - {p['period']}: {p['similarity']:.3f} similarity\n"
        
        # NEW v4: Format topic turnover details
        topic_evolution = ""
        if topic_turnover_details:
            topic_evolution = "\n   Topic turnover by period (Jaccard-based):\n"
            for t in topic_turnover_details:
                topic_evolution += f"   - {t['period']}: {t['turnover']:.3f} turnover (Jaccard sim: {t['jaccard_similarity']:.3f})\n"
        
        # NEW v4: Format cluster/motif stats
        motif_stats = ""
        if cluster_stats and 'uniqueness' in cluster_stats:
            motif_stats = f"""
   Motif Diversity Breakdown (K-means clustering):
   - Clusters used: {cluster_stats.get('n_clusters_used', 'N/A')} of {cluster_stats.get('n_clusters_target', 'N/A')} target
   - Uniqueness (cluster utilization): {cluster_stats.get('uniqueness', 0):.3f}
   - Evenness (distribution entropy): {cluster_stats.get('evenness', 0):.3f}"""
        
        # NEW v4: Format temporal coherence details
        temporal_coherence_section = ""
        if temporal_coherence is not None:
            temporal_coherence_section = f"\n   - Temporal Coherence (within-period alignment): {temporal_coherence:.3f}"
            if temporal_coherence_by_year:
                temporal_coherence_section += "\n   Per-year text-image coherence:\n"
                for tc in sorted(temporal_coherence_by_year, key=lambda x: x['year']):
                    temporal_coherence_section += f"   - {tc['year']}: {tc['similarity']:.3f}\n"

        prompt = f"""You are a brand strategy analyst specializing in brand identity evolution and liquid modernity theory (Bauman, 2000). Analyze the following Brand Fluidity metrics for {brand_name} and provide strategic insights.

## IMPORTANT: Methodology Context (v4)

This analysis uses **Temporal Self-Similarity** methodology with peer-reviewed computational methods:
- **Fluidity = 1 - Stability**: We measure how consistent the brand is with ITSELF across different time periods
- **Low fluidity scores** indicate the brand maintained consistent identity over time (SOLID)
- **High fluidity scores** indicate the brand's identity CHANGED significantly between time periods (FLUID)
- **Topic Turnover**: Measured via Jaccard similarity of LDA topic sets between periods (Blei et al., 2003)
- **Motif Diversity**: Measured via K-means clustering of CLIP embeddings (MacQueen, 1967)
- **Cross-Modal Coherence**: Includes temporal coherence—within-period text-image alignment

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
   - Semantic Drift (SBERT centroids): {tfi.get('semantic_drift', 0):.3f}
   - Tone/Personality Shift (Aaker dimensions): {tfi.get('tone_diversity', 0):.3f}
   - Topic Turnover (Jaccard-based LDA): {tfi.get('topic_turnover', 0):.3f}
   - Interpretation: {tfi.get('interpretation', 'N/A')}{text_evolution}{topic_evolution}

### 2. Visual Fluidity Index (VFI): {vfi.get('VFI', 0):.3f}
   *Measures how much the brand's visual identity CHANGED over time*
   - Temporal Visual Stability: {vfi.get('temporal_stability', 1 - vfi.get('VFI', 0)):.3f}
   - Visual Drift (CLIP centroids): {vfi.get('visual_drift', 0):.3f}
   - Color Entropy: {vfi.get('color_entropy', 0):.3f}
   - Color Entropy Change: {vfi.get('color_entropy_change', 'N/A')}
   - Motif Diversity (clustering-based): {vfi.get('motif_diversity', 0):.3f}
   - Interpretation: {vfi.get('interpretation', 'N/A')}{visual_evolution}{motif_stats}

### 3. Cross-Modal Coherence Score (CMCS): {cmcs.get('CMCS', 0):.3f}
   *Measures semantic alignment between text and visual brand expressions*
   - Centroid Similarity (global): {cmcs.get('centroid_similarity', 'N/A')}
   - Nearest-Neighbor Coherence: {cmcs.get('nn_coherence', 'N/A')}{temporal_coherence_section}
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
4-5 bullet points covering:
- What do the stability/change patterns reveal about brand management?
- Are text and visual identities evolving in sync or diverging?
- What does the topic turnover pattern reveal about messaging strategy evolution?
- What does the cross-modal coherence (especially temporal coherence) tell us about integrated brand storytelling?
- Any notable period-specific changes (identify inflection points if visible)?

## 3. Strategic Implications
- How does this temporal evolution pattern position the brand competitively?
- What are the risks of the current fluidity level?
- What opportunities exist?
- Comment on the motif diversity pattern (uniqueness vs evenness)—is visual content appropriately varied or too scattered/concentrated?

## 4. Recommendations
4-5 actionable suggestions:
- Should the brand increase or decrease temporal consistency in specific areas?
- How can cross-modal coherence be improved if needed (especially within-period alignment)?
- What governance or creative direction changes might help?
- Specific suggestions for topic/messaging strategy if topic turnover is notable
- Visual consistency recommendations based on motif clustering patterns

## 5. Comparative Context
- How does this profile compare to typical patterns for this brand category?
- Is the fluidity level appropriate for the brand's positioning (heritage vs. contemporary)?
- Note any concerning divergences between textual and visual evolution rates
- Comment on whether the temporal coherence trend suggests improving or degrading brand integration

Ground your analysis in brand theory, semiotics, and the temporal self-similarity framework. Focus on actionable insights that connect the metrics to strategic brand decisions."""

        return prompt
