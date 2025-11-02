"""
LLM Analyzer for Brand Fluidity Metrics
Uses Claude API to generate strategic insights from the calculated metrics.
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
                max_tokens=2000,
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
        
        prompt = f"""You are a brand strategy analyst specializing in brand identity evolution and semiotics. Analyze the following Brand Fluidity metrics for {brand_name} and provide strategic insights.

## Brand Fluidity Analysis Results

**Brand:** {brand_name}

### Summary
- Texts analyzed: {summary.get('texts_analyzed', 0)}
- Images analyzed: {summary.get('images_analyzed', 0)}
- Time periods covered: {summary.get('time_periods', 0)}
- Overall category: {summary.get('overall_category', 'Unknown')}

### 1. Textual Fluidity Index (TFI): {tfi.get('TFI', 0):.3f}
- Semantic Drift: {tfi.get('semantic_drift', 0):.3f}
- Tone Diversity: {tfi.get('tone_diversity', 0):.3f}
- Topic Turnover: {tfi.get('topic_turnover', 0):.3f}
- Interpretation: {tfi.get('interpretation', 'N/A')}

### 2. Visual Fluidity Index (VFI): {vfi.get('VFI', 0):.3f}
- Visual Drift: {vfi.get('visual_drift', 0):.3f}
- Color Entropy: {vfi.get('color_entropy', 0):.3f}
- Motif Diversity: {vfi.get('motif_diversity', 0):.3f}
- Interpretation: {vfi.get('interpretation', 'N/A')}

### 3. Cross-Modal Coherence Score (CMCS): {cmcs.get('CMCS', 0):.3f}
- Interpretation: {cmcs.get('interpretation', 'N/A')}

### 4. Brand Fluidity Index (BFI): {bfi.get('BFI', 0):.3f}
- Category: {bfi.get('category', 'Unknown')}
- Description: {bfi.get('description', 'N/A')}

---

Please provide:

1. **Executive Summary** (2-3 sentences)
   - What does this brand's fluidity profile tell us about their identity strategy?

2. **Key Insights** (3-5 bullet points)
   - What are the most important takeaways from these metrics?
   - What patterns or tensions do you observe?

3. **Strategic Implications**
   - How does this fluidity profile position the brand in the market?
   - What are the risks and opportunities?

4. **Recommendations** (3-4 actionable suggestions)
   - What should the brand consider doing to optimize their identity fluidity?
   - Should they increase or decrease fluidity in specific areas?

5. **Comparative Context**
   - How does this profile compare to typical patterns?
   - Reference the theoretical framework: solid (0.0-0.3), semi-fluid (0.3-0.6), fluid (0.6-1.0)

Keep the analysis strategic, specific to the metrics provided, and grounded in brand theory and semiotics. Focus on actionable insights."""

        return prompt
