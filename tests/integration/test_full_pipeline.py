"""
Integration Tests for Helios Phase 1 Pipeline
Tests the complete workflow from market research to ethical code generation
"""

import asyncio
import pytest
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Import from helios package
from helios.services.market_research import MarketResearchService
from helios.services.copyright_review import CopyrightReviewService
from helios.services.supplier_vetting import SupplierVettingService
from helios.services.ethical_code import EthicalCodeService
from helios.services.phase1_executor import Phase1Executor


class TestFullPipeline:
    """Test the complete Phase 1 pipeline integration"""
    
    @pytest.fixture(autouse=True)
    def setup_services(self):
        """Initialize all services for testing"""
        self.market_research = MarketResearchService()
        self.copyright_review = CopyrightReviewService()
        self.supplier_vetting = SupplierVettingService()
        self.ethical_code = EthicalCodeService()
        self.phase1_executor = Phase1Executor()
        
        # Test data
        self.test_designs = [
            {
                "name": "Retro Gaming Console",
                "description": "Vintage gaming console design",
                "category": "gaming",
                "risk_factors": ["copyright", "trademark"]
            },
            {
                "name": "Pixel Art Character",
                "description": "8-bit style gaming character",
                "category": "art",
                "risk_factors": ["original_art"]
            },
            {
                "name": "Vintage Gaming Logo",
                "description": "Classic gaming logo design",
                "category": "branding",
                "risk_factors": ["trademark", "copyright"]
            }
        ]
    
    @pytest.mark.asyncio
    async def test_market_research_real_time_data(self):
        """Test that market research uses real-time data"""
        print("\n🔍 Testing Market Research Real-Time Data...")
        
        # Run market research
        research_data = await self.market_research.research_vintage_gaming_niches()
        
        # Validate real-time data
        assert research_data.get('data_freshness') == 'real-time', "Data should be real-time"
        assert 'trend_analysis' in research_data, "Should include trend analysis"
        assert 'keywords' in research_data['trend_analysis'], "Should include trending keywords"
        assert len(research_data['trend_analysis']['keywords']) > 0, "Should have trending keywords"
        assert 'opportunity_score' in research_data['trend_analysis'], "Should include opportunity score"
        
        # Validate niche opportunities
        assert 'niche_opportunities' in research_data, "Should include niche opportunities"
        assert len(research_data['niche_opportunities']) > 0, "Should have niche opportunities"
        
        # Check for trend velocity and opportunity scores
        for niche in research_data['niche_opportunities']:
            assert 'trend_velocity' in niche, "Each niche should have trend velocity"
            assert 'opportunity_score' in niche, "Each niche should have opportunity score"
        
        print(f"✅ Market Research: {len(research_data['niche_opportunities'])} niches identified")
        print(f"   Trending keywords: {len(research_data['trend_analysis']['keywords'])}")
        print(f"   Opportunity score: {research_data['trend_analysis']['opportunity_score']}")
    
    @pytest.mark.asyncio
    async def test_copyright_review_comprehensive(self):
        """Test comprehensive copyright review functionality"""
        print("\n📋 Testing Copyright Review Service...")
        
        # Run copyright review
        review_results = await self.copyright_review.review_designs(self.test_designs)
        
        # Validate results structure
        assert 'reviewed_designs' in review_results, "Should include reviewed designs"
        assert 'risk_summary' in review_results, "Should include risk summary"
        assert 'recommendations' in review_results, "Should include recommendations"
        
        # Validate each design was reviewed
        assert len(review_results['reviewed_designs']) == len(self.test_designs), "All designs should be reviewed"
        
        # Check risk assessment
        for design in review_results['reviewed_designs']:
            assert 'risk_level' in design, "Each design should have risk level"
            assert 'risk_factors' in design, "Each design should have risk factors"
            assert 'recommendation' in design, "Each design should have recommendation"
        
        # Validate risk summary
        risk_summary = review_results['risk_summary']
        assert 'total_designs' in risk_summary, "Risk summary should include total designs"
        assert 'high_risk_count' in risk_summary, "Risk summary should include high risk count"
        assert 'medium_risk_count' in risk_summary, "Risk summary should include medium risk count"
        assert 'low_risk_count' in risk_summary, "Risk summary should include low risk count"
        
        print(f"✅ Copyright Review: {risk_summary['total_designs']} designs reviewed")
        print(f"   High risk: {risk_summary['high_risk_count']}")
        print(f"   Medium risk: {risk_summary['medium_risk_count']}")
        print(f"   Low risk: {risk_summary['low_risk_count']}")
    
    @pytest.mark.asyncio
    async def test_supplier_vetting_operational(self):
        """Test supplier vetting operational status"""
        print("\n🏭 Testing Supplier Vetting Service...")
        
        # Run supplier vetting
        vetting_results = await self.supplier_vetting.vet_pod_suppliers()
        
        # Validate results structure
        assert 'primary_provider' in vetting_results, "Should include primary provider"
        assert 'alternative_platforms' in vetting_results, "Should include alternative platforms"
        assert 'recommendations' in vetting_results, "Should include recommendations"
        
        # Check primary provider status
        primary = vetting_results['primary_provider']
        assert 'name' in primary, "Primary provider should have name"
        assert 'status' in primary, "Primary provider should have status"
        assert 'health_check' in primary, "Primary provider should have health check"
        
        # Validate alternative platforms
        alternatives = vetting_results['alternative_platforms']
        assert len(alternatives) > 0, "Should have alternative platforms"
        
        for platform in alternatives:
            assert 'name' in platform, "Each platform should have name"
            assert 'pros' in platform, "Each platform should have pros"
            assert 'cons' in platform, "Each platform should have cons"
            assert 'best_for' in platform, "Each platform should have best_for"
        
        print(f"✅ Supplier Vetting: {primary['name']} status: {primary['status']}")
        print(f"   Alternative platforms: {len(alternatives)}")
        print(f"   Health check: {primary['health_check']}")
    
    @pytest.mark.asyncio
    async def test_ethical_code_generation(self):
        """Test ethical code generation and training materials"""
        print("\n⚖️ Testing Ethical Code Service...")
        
        # Run ethical code generation
        ethical_results = await self.ethical_code.generate_ethical_framework()
        
        # Validate results structure
        assert 'core_values' in ethical_results, "Should include core values"
        assert 'code_of_conduct' in ethical_results, "Should include code of conduct"
        assert 'training_materials' in ethical_results, "Should include training materials"
        assert 'compliance_checklist' in ethical_results, "Should include compliance checklist"
        
        # Validate core values
        core_values = ethical_results['core_values']
        assert len(core_values) > 0, "Should have core values"
        for value in core_values:
            assert 'name' in value, "Each value should have name"
            assert 'description' in value, "Each value should have description"
            assert 'implementation' in value, "Each value should have implementation"
        
        # Validate training materials
        training = ethical_results['training_materials']
        assert 'employee_handbook' in training, "Should include employee handbook"
        assert 'manager_guide' in training, "Should include manager guide"
        assert 'customer_communication' in training, "Should include customer communication"
        
        print(f"✅ Ethical Code: {len(core_values)} core values defined")
        print(f"   Training materials: {len(training)} documents generated")
        print(f"   Compliance checklist: {len(ethical_results['compliance_checklist'])} items")
    
    @pytest.mark.asyncio
    async def test_full_phase1_execution(self):
        """Test complete Phase 1 execution pipeline"""
        print("\n🚀 Testing Complete Phase 1 Execution...")
        
        # Run full Phase 1
        phase1_results = await self.phase1_executor.execute_phase1()
        
        # Validate overall results
        assert 'status' in phase1_results, "Should include execution status"
        assert 'services_executed' in phase1_results, "Should include services executed"
        assert 'execution_time' in phase1_results, "Should include execution time"
        assert 'reports_generated' in phase1_results, "Should include reports generated"
        
        # Check execution status
        assert phase1_results['status'] == 'completed', "Phase 1 should complete successfully"
        
        # Validate all services were executed
        services = phase1_results['services_executed']
        expected_services = ['market_research', 'copyright_review', 'supplier_vetting', 'ethical_code']
        for service in expected_services:
            assert service in services, f"Service {service} should be executed"
            assert services[service]['status'] == 'success', f"Service {service} should succeed"
        
        # Check reports were generated
        reports = phase1_results['reports_generated']
        assert len(reports) > 0, "Should generate reports"
        
        print(f"✅ Phase 1 Execution: {phase1_results['status']}")
        print(f"   Services executed: {len(services)}")
        print(f"   Execution time: {phase1_results['execution_time']:.2f}s")
        print(f"   Reports generated: {len(reports)}")
    
    @pytest.mark.asyncio
    async def test_data_freshness_and_consistency(self):
        """Test that all services use consistent, fresh data"""
        print("\n🔄 Testing Data Freshness and Consistency...")
        
        # Run all services and collect timestamps
        start_time = datetime.now()
        
        market_data = await self.market_research.research_vintage_gaming_niches()
        copyright_data = await self.copyright_review.review_designs(self.test_designs)
        supplier_data = await self.supplier_vetting.vet_pod_suppliers()
        ethical_data = await self.ethical_code.generate_ethical_framework()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Validate data freshness
        assert market_data.get('data_freshness') == 'real-time', "Market data should be real-time"
        
        # Check execution time is reasonable (should complete within 5 minutes)
        assert execution_time < 300, f"Execution should complete within 5 minutes, took {execution_time:.2f}s"
        
        # Validate data consistency across services
        market_keywords = market_data['trend_analysis'].get('keywords', [])
        if market_keywords:
            # Check that trending keywords are used consistently
            for keyword in market_keywords[:5]:  # Check first 5 keywords
                keyword_str = str(keyword).lower()
                # Keywords should appear in relevant service outputs
                if 'retro' in keyword_str or 'vintage' in keyword_str:
                    assert any('retro' in str(supplier_data).lower() or 'vintage' in str(supplier_data).lower()), \
                        "Vintage/retro keywords should appear in supplier data"
        
        print(f"✅ Data Consistency: All services executed successfully")
        print(f"   Total execution time: {execution_time:.2f}s")
        print(f"   Trending keywords: {len(market_keywords)}")
        print(f"   Data freshness: {market_data.get('data_freshness')}")


async def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Helios Phase 1 Integration Tests")
    print("=" * 60)
    
    # Create test instance
    test_instance = TestFullPipeline()
    await test_instance.setup_services()
    
    # Run all tests
    tests = [
        test_instance.test_market_research_real_time_data(),
        test_instance.test_copyright_review_comprehensive(),
        test_instance.test_supplier_vetting_operational(),
        test_instance.test_ethical_code_generation(),
        test_instance.test_full_phase1_execution(),
        test_instance.test_data_freshness_and_consistency()
    ]
    
    results = []
    for test in tests:
        try:
            await test
            results.append(("✅ PASS", test.__name__))
        except Exception as e:
            results.append(("❌ FAIL", f"{test.__name__}: {str(e)}"))
    
    # Print results summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for status, _ in results if status == "✅ PASS")
    total = len(results)
    
    for status, test_name in results:
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! Phase 1 is ready for production.")
    else:
        print("⚠️ Some tests failed. Please review and fix issues before production deployment.")
    
    return passed == total


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)
