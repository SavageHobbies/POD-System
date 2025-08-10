from typing import Dict, List, Any
import re
from ..mcp_client import MCPClient
from ..config import load_config

class CopyrightReviewService:
    """Comprehensive copyright review system for POD products"""
    
    def __init__(self):
        self.config = load_config()
        self.mcp_client = MCPClient.from_env(
            self.config.google_mcp_url, 
            self.config.google_mcp_auth_token
        )
        
        # Copyright risk patterns
        self.high_risk_patterns = [
            r"mario", r"sonic", r"pokemon", r"zelda", r"final fantasy",
            r"street fighter", r"mortal kombat", r"resident evil",
            r"nintendo", r"sega", r"capcom", r"square enix"
        ]
        
        self.moderate_risk_patterns = [
            r"retro", r"vintage", r"classic", r"arcade", r"console",
            r"pixel", r"8-bit", r"16-bit", r"gaming", r"gamer"
        ]
    
    async def review_design(self, design_concept: str, keywords: List[str]) -> Dict[str, Any]:
        """Review design for copyright infringement risks"""
        
        # Pattern-based screening
        risk_score = self._calculate_risk_score(design_concept, keywords)
        
        # AI-powered analysis using MCP
        ai_analysis = await self._ai_copyright_review(design_concept, keywords)
        
        # Legal compliance check
        compliance_status = self._check_legal_compliance(design_concept)
        
        return {
            "risk_score": risk_score,
            "ai_analysis": ai_analysis,
            "compliance_status": compliance_status,
            "recommendations": self._generate_recommendations(risk_score, ai_analysis),
            "requires_legal_review": risk_score > 7.0
        }
    
    def _calculate_risk_score(self, design: str, keywords: List[str]) -> float:
        """Calculate copyright risk score (0-10)"""
        score = 0.0
        
        # Check for high-risk patterns
        for pattern in self.high_risk_patterns:
            if re.search(pattern, design.lower()):
                score += 3.0
        
        # Check keywords
        for keyword in keywords:
            if any(re.search(pattern, keyword.lower()) for pattern in self.high_risk_patterns):
                score += 2.0
        
        return min(score, 10.0)
    
    async def _ai_copyright_review(self, design: str, keywords: List[str]) -> Dict[str, Any]:
        """Use AI to analyze copyright risks"""
        
        prompt = f"""
        Analyze this design concept for copyright infringement risks:
        
        Design: {design}
        Keywords: {', '.join(keywords)}
        
        Assess:
        1. Likelihood of copyright violation (0-100%)
        2. Specific copyright concerns
        3. Fair use considerations
        4. Recommended modifications
        5. Alternative approaches
        
        Focus on vintage gaming context and POD business model.
        """
        
        response = await self.mcp_client.orchestrator_ai({
            "prompt": prompt
        })
        
        return response
    
    def _check_legal_compliance(self, design: str) -> Dict[str, Any]:
        """Check basic legal compliance"""
        return {
            "trademark_compliant": not self._has_trademarks(design),
            "copyright_compliant": not self._has_copyrighted_content(design),
            "fair_use_applicable": self._assess_fair_use(design),
            "legal_notes": self._generate_legal_notes(design)
        }
    
    def _has_trademarks(self, design: str) -> bool:
        """Check if design contains trademarked terms"""
        trademark_terms = [
            "nintendo", "sony", "microsoft", "sega", "capcom",
            "square enix", "konami", "namco", "bandai"
        ]
        return any(term in design.lower() for term in trademark_terms)
    
    def _has_copyrighted_content(self, design: str) -> bool:
        """Check if design contains copyrighted content"""
        copyrighted_terms = [
            "mario", "sonic", "link", "cloud", "ryu", "ken",
            "pokemon", "zelda", "final fantasy", "street fighter"
        ]
        return any(term in design.lower() for term in copyrighted_terms)
    
    def _assess_fair_use(self, design: str) -> bool:
        """Assess if fair use might apply"""
        # Fair use is complex and requires legal analysis
        # This is a simplified assessment
        return False  # Always consult legal counsel for fair use questions
    
    def _generate_legal_notes(self, design: str) -> str:
        """Generate legal compliance notes"""
        notes = []
        
        if self._has_trademarks(design):
            notes.append("Contains trademarked terms - high risk")
        
        if self._has_copyrighted_content(design):
            notes.append("Contains copyrighted content - high risk")
        
        if not notes:
            notes.append("No obvious copyright/trademark issues detected")
        
        return "; ".join(notes)
    
    def _generate_recommendations(self, risk_score: float, ai_analysis: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if risk_score > 8.0:
            recommendations.append("HIGH RISK: Consult legal counsel before proceeding")
            recommendations.append("Consider completely original design approach")
        
        elif risk_score > 5.0:
            recommendations.append("MODERATE RISK: Modify design to reduce copyright concerns")
            recommendations.append("Focus on generic gaming elements rather than specific IP")
        
        else:
            recommendations.append("LOW RISK: Proceed with current design")
            recommendations.append("Continue monitoring for any copyright issues")
        
        return recommendations
    
    async def batch_review_designs(self, designs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Review multiple designs in batch"""
        results = []
        
        for design in designs:
            result = await self.review_design(
                design.get("concept", ""),
                design.get("keywords", [])
            )
            result["design_id"] = design.get("id", "unknown")
            results.append(result)
        
        return results
    
    def generate_copyright_report(self, review_results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive copyright review report"""
        
        high_risk_count = sum(1 for r in review_results if r["risk_score"] > 7.0)
        moderate_risk_count = sum(1 for r in review_results if 4.0 < r["risk_score"] <= 7.0)
        low_risk_count = sum(1 for r in review_results if r["risk_score"] <= 4.0)
        
        report = f"""
# Copyright Review Report

## Summary
- Total Designs Reviewed: {len(review_results)}
- High Risk: {high_risk_count}
- Moderate Risk: {moderate_risk_count}
- Low Risk: {low_risk_count}

## Risk Distribution
- High Risk (7.0+): {high_risk_count} designs
- Moderate Risk (4.0-7.0): {moderate_risk_count} designs
- Low Risk (0.0-4.0): {low_risk_count} designs

## Recommendations
"""
        
        if high_risk_count > 0:
            report += "- High-risk designs require immediate legal review\n"
        
        if moderate_risk_count > 0:
            report += "- Moderate-risk designs should be modified before use\n"
        
        if low_risk_count > 0:
            report += "- Low-risk designs can proceed with monitoring\n"
        
        report += "\n## Detailed Results\n"
        
        for result in review_results:
            report += f"""
### Design {result.get('design_id', 'Unknown')}
- Risk Score: {result['risk_score']}/10
- Status: {'HIGH RISK' if result['risk_score'] > 7.0 else 'MODERATE RISK' if result['risk_score'] > 4.0 else 'LOW RISK'}
- Legal Review Required: {'Yes' if result['requires_legal_review'] else 'No'}
- Recommendations: {', '.join(result['recommendations'])}
"""
        
        return report
