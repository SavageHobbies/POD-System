import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from helios.agents.ceo import HeliosCEO, Priority, QualityGateStatus, TrendDecision


class TestHeliosCEO:
    """Test cases for the HeliosCEO agent"""
    
    @pytest.fixture
    def ceo_agent(self):
        """Create a CEO agent instance for testing"""
        return HeliosCEO(min_opportunity=7.0, min_confidence=0.7)
    
    @pytest.fixture
    def sample_trend_data(self):
        """Sample trend data for testing"""
        return {
            "trend_name": "Test Trend",
            "keywords": ["test", "trend", "example"],
            "opportunity_score": 8.5,
            "velocity": "high",
            "urgency_level": "high",
            "ethical_status": "approved",
            "confidence_level": 0.85,
            "mcp_model_used": "gemini-1.5-pro"
        }
    
    def test_initialization(self, ceo_agent):
        """Test CEO agent initialization"""
        assert ceo_agent.min_opportunity == 7.0
        assert ceo_agent.min_audience_confidence == 0.7
        assert ceo_agent.min_profit_margin == 0.35
        assert ceo_agent.max_execution_time == 300
        assert len(ceo_agent.quality_gates) == 5
    
    def test_priority_determination(self, ceo_agent):
        """Test priority determination logic"""
        # High priority: high urgency + high opportunity
        priority = ceo_agent._determine_priority("high", 9.0, "high")
        assert priority == Priority.HIGH
        
        # Medium priority: medium urgency + medium opportunity
        priority = ceo_agent._determine_priority("medium", 7.5, "medium")
        assert priority == Priority.MEDIUM
        
        # Low priority: low urgency + low opportunity
        priority = ceo_agent._determine_priority("low", 6.0, "low")
        assert priority == Priority.LOW
    
    @pytest.mark.asyncio
    async def test_validate_trend_success(self, ceo_agent, sample_trend_data):
        """Test successful trend validation"""
        with patch.object(ceo_agent, '_run_quality_gates') as mock_gates:
            mock_gates.return_value = {
                "ethical_approval": Mock(
                    status=QualityGateStatus.PASSED,
                    score=1.0,
                    threshold=1.0,
                    details="Passed ethical screening",
                    execution_time_ms=100
                ),
                "audience_confidence": Mock(
                    status=QualityGateStatus.PASSED,
                    score=0.85,
                    threshold=0.7,
                    details="High audience confidence",
                    execution_time_ms=150
                ),
                "trend_opportunity": Mock(
                    status=QualityGateStatus.PASSED,
                    score=8.5,
                    threshold=7.0,
                    details="High opportunity score",
                    execution_time_ms=200
                )
            }
            
            result = await ceo_agent.validate_trend(sample_trend_data)
            
            assert result.approved is True
            assert result.trend_name == "Test Trend"
            assert result.opportunity_score == 8.5
            assert result.priority == Priority.HIGH
            assert len(result.quality_gates) == 3
    
    @pytest.mark.asyncio
    async def test_validate_trend_failure_low_opportunity(self, ceo_agent, sample_trend_data):
        """Test trend validation failure due to low opportunity score"""
        sample_trend_data["opportunity_score"] = 5.0
        
        with patch.object(ceo_agent, '_run_quality_gates') as mock_gates:
            mock_gates.return_value = {
                "trend_opportunity": Mock(
                    status=QualityGateStatus.FAILED,
                    score=5.0,
                    threshold=7.0,
                    details="Opportunity score too low",
                    execution_time_ms=100
                )
            }
            
            result = await ceo_agent.validate_trend(sample_trend_data)
            
            assert result.approved is False
            assert result.opportunity_score == 5.0
    
    @pytest.mark.asyncio
    async def test_validate_trend_failure_ethical_concerns(self, ceo_agent, sample_trend_data):
        """Test trend validation failure due to ethical concerns"""
        with patch.object(ceo_agent, '_run_quality_gates') as mock_gates:
            mock_gates.return_value = {
                "ethical_approval": Mock(
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    threshold=1.0,
                    details="Failed ethical screening",
                    execution_time_ms=100
                )
            }
            
            result = await ceo_agent.validate_trend(sample_trend_data)
            
            assert result.approved is False
    
    @pytest.mark.asyncio
    async def test_batch_validate_trends(self, ceo_agent):
        """Test batch trend validation"""
        trends = [
            {"trend_name": "Trend 1", "opportunity_score": 8.0, "confidence_level": 0.8},
            {"trend_name": "Trend 2", "opportunity_score": 6.0, "confidence_level": 0.6},
            {"trend_name": "Trend 3", "opportunity_score": 9.0, "confidence_level": 0.9}
        ]
        
        with patch.object(ceo_agent, 'validate_trend') as mock_validate:
            mock_validate.side_effect = [
                Mock(approved=True, trend_name="Trend 1"),
                Mock(approved=False, trend_name="Trend 2"),
                Mock(approved=True, trend_name="Trend 3")
            ]
            
            results = await ceo_agent.batch_validate_trends(trends)
            
            assert len(results) == 3
            assert results[0].approved is True
            assert results[1].approved is False
            assert results[2].approved is True
    
    @pytest.mark.asyncio
    async def test_coordinate_parallel_execution(self, ceo_agent):
        """Test parallel execution coordination"""
        agents = ["audience_analyst", "product_strategist"]
        
        result = await ceo_agent.coordinate_parallel_execution("analysis", agents)
        
        assert result["stage"] == "analysis"
        assert result["agents"] == agents
        assert result["status"] == "coordinated"
    
    def test_create_failed_gate_result(self, ceo_agent):
        """Test creation of failed gate result"""
        result = ceo_agent._create_failed_gate_result("test_gate", "Test error")
        
        assert result.gate_name == "test_gate"
        assert result.status == QualityGateStatus.FAILED
        assert result.score == 0.0
        assert "Test error" in result.details
    
    def test_generate_optimization_recommendations(self, ceo_agent):
        """Test optimization recommendation generation"""
        quality_gates = {
            "audience_confidence": Mock(
                status=QualityGateStatus.PASSED,
                score=0.8,
                threshold=0.7
            ),
            "trend_opportunity": Mock(
                status=QualityGateStatus.PASSED,
                score=8.5,
                threshold=7.0
            )
        }
        
        trend_data = {"trend_name": "Test Trend"}
        priority = Priority.HIGH
        
        recommendations = ceo_agent._generate_optimization_recommendations(
            quality_gates, trend_data, priority
        )
        
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


if __name__ == "__main__":
    pytest.main([__file__])
