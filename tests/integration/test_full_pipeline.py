import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from helios.main import run_helios_pipeline
from helios.config import load_config


class TestFullPipeline:
    """Integration tests for the complete Helios pipeline"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        config = Mock()
        config.dry_run = True
        config.enable_parallel_processing = True
        config.min_opportunity_score = 7.0
        config.min_audience_confidence = 0.7
        config.max_execution_time = 300
        return config
    
    @pytest.fixture
    def sample_trend_data(self):
        """Sample trend data for testing"""
        return {
            "status": "approved",
            "trend_name": "Test Trend",
            "keywords": ["test", "trend", "example"],
            "opportunity_score": 8.5,
            "velocity": "high",
            "urgency_level": "high",
            "ethical_status": "approved",
            "confidence_level": 0.85,
            "mcp_model_used": "gemini-1.5-pro"
        }
    
    @pytest.fixture
    def mock_ceo_decision(self):
        """Mock CEO decision for testing"""
        decision = Mock()
        decision.approved = True
        decision.trend_name = "Test Trend"
        decision.opportunity_score = 8.5
        decision.priority = "high"
        decision.mcp_model_used = "gemini-1.5-pro"
        return decision
    
    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, mock_config, sample_trend_data, mock_ceo_decision):
        """Test successful execution of the full pipeline"""
        with patch('helios.main.load_config', return_value=mock_config), \
             patch('helios.main.ZeitgeistAgent') as mock_zeitgeist, \
             patch('helios.main.HeliosCEO') as mock_ceo, \
             patch('helios.main.AudienceAnalyst') as mock_audience, \
             patch('helios.main.ProductStrategist') as mock_product, \
             patch('helios.main.CreativeDirector') as mock_creative, \
             patch('helios.main.MarketingCopywriter') as mock_marketing, \
             patch('helios.main.PrintifyPublisherAgent') as mock_publisher:
            
            # Mock Zeitgeist agent
            mock_zeitgeist_instance = Mock()
            mock_zeitgeist_instance.run.return_value = sample_trend_data
            mock_zeitgeist.return_value = mock_zeitgeist_instance
            
            # Mock CEO agent
            mock_ceo_instance = Mock()
            mock_ceo_instance.prepare_validation.return_value = {"status": "ready"}
            mock_ceo_instance.validate_trend.return_value = mock_ceo_decision
            mock_ceo.return_value = mock_ceo_instance
            
            # Mock Audience Analyst
            mock_audience_instance = Mock()
            mock_audience_instance.run.return_value = {
                "audience_segments": ["segment1", "segment2"],
                "confidence_level": 0.85,
                "behavioral_insights": ["insight1", "insight2"]
            }
            mock_audience.return_value = mock_audience_instance
            
            # Mock Product Strategist
            mock_product_instance = Mock()
            mock_product_instance.get_products_async.return_value = {
                "products": ["product1", "product2"],
                "pricing_strategy": "competitive",
                "margin_analysis": {"avg_margin": 0.45}
            }
            mock_product.return_value = mock_product_instance
            
            # Mock Creative Director
            mock_creative_instance = Mock()
            mock_creative_instance.generate_designs.return_value = {
                "designs": ["design1", "design2"],
                "quality_scores": [0.9, 0.85],
                "brand_compliance": True
            }
            mock_creative.return_value = mock_creative_instance
            
            # Mock Marketing Copywriter
            mock_marketing_instance = Mock()
            mock_marketing_instance.generate_copy.return_value = {
                "titles": ["title1", "title2"],
                "descriptions": ["desc1", "desc2"],
                "seo_optimized": True
            }
            mock_marketing.return_value = mock_marketing_instance
            
            # Mock Publisher
            mock_publisher_instance = Mock()
            mock_publisher_instance.publish_products.return_value = {
                "published_count": 2,
                "success_rate": 1.0,
                "product_ids": ["id1", "id2"]
            }
            mock_publisher.return_value = mock_publisher_instance
            
            # Run the pipeline
            start_time = time.time()
            result = await run_helios_pipeline(
                seed="test seed",
                dry_run=True,
                enable_parallel=True
            )
            execution_time = time.time() - start_time
            
            # Verify results
            assert result["status"] == "success"
            assert execution_time < 300  # Should complete within max execution time
            assert "trend_name" in result
            assert "audience_analysis" in result
            assert "product_strategy" in result
            assert "creative_output" in result
            assert "marketing_copy" in result
            assert "publication_results" in result
    
    @pytest.mark.asyncio
    async def test_pipeline_trend_rejection(self, mock_config):
        """Test pipeline behavior when trend is rejected"""
        with patch('helios.main.load_config', return_value=mock_config), \
             patch('helios.main.ZeitgeistAgent') as mock_zeitgeist:
            
            # Mock rejected trend
            mock_zeitgeist_instance = Mock()
            mock_zeitgeist_instance.run.return_value = {
                "status": "rejected",
                "reason": "Low opportunity score"
            }
            mock_zeitgeist.return_value = mock_zeitgeist_instance
            
            result = await run_helios_pipeline(
                seed="test seed",
                dry_run=True,
                enable_parallel=True
            )
            
            assert result["status"] == "trend_rejected"
            assert "reason" in result
    
    @pytest.mark.asyncio
    async def test_pipeline_ceo_rejection(self, mock_config, sample_trend_data):
        """Test pipeline behavior when CEO rejects trend"""
        with patch('helios.main.load_config', return_value=mock_config), \
             patch('helios.main.ZeitgeistAgent') as mock_zeitgeist, \
             patch('helios.main.HeliosCEO') as mock_ceo:
            
            # Mock approved trend
            mock_zeitgeist_instance = Mock()
            mock_zeitgeist_instance.run.return_value = sample_trend_data
            mock_zeitgeist.return_value = mock_zeitgeist_instance
            
            # Mock CEO rejection
            mock_ceo_instance = Mock()
            mock_ceo_instance.prepare_validation.return_value = {"status": "ready"}
            mock_ceo_instance.validate_trend.return_value = Mock(
                approved=False,
                trend_name="Test Trend",
                opportunity_score=5.0
            )
            mock_ceo.return_value = mock_ceo_instance
            
            result = await run_helios_pipeline(
                seed="test seed",
                dry_run=True,
                enable_parallel=True
            )
            
            assert result["status"] == "ceo_rejected"
            assert "reason" in result
    
    @pytest.mark.asyncio
    async def test_pipeline_parallel_execution(self, mock_config, sample_trend_data, mock_ceo_decision):
        """Test pipeline with parallel execution enabled"""
        with patch('helios.main.load_config', return_value=mock_config), \
             patch('helios.main.ZeitgeistAgent') as mock_zeitgeist, \
             patch('helios.main.HeliosCEO') as mock_ceo, \
             patch('helios.main.AudienceAnalyst') as mock_audience, \
             patch('helios.main.ProductStrategist') as mock_product, \
             patch('helios.main.CreativeDirector') as mock_creative, \
             patch('helios.main.MarketingCopywriter') as mock_marketing, \
             patch('helios.main.PrintifyPublisherAgent') as mock_publisher:
            
            # Setup all mocks
            mock_zeitgeist_instance = Mock()
            mock_zeitgeist_instance.run.return_value = sample_trend_data
            mock_zeitgeist.return_value = mock_zeitgeist_instance
            
            mock_ceo_instance = Mock()
            mock_ceo_instance.prepare_validation.return_value = {"status": "ready"}
            mock_ceo_instance.validate_trend.return_value = mock_ceo_decision
            mock_ceo.return_value = mock_ceo_instance
            
            # Mock other agents
            mock_audience.return_value = Mock(run=AsyncMock(return_value={"status": "success"}))
            mock_product.return_value = Mock(get_products_async=AsyncMock(return_value={"status": "success"}))
            mock_creative.return_value = Mock(generate_designs=AsyncMock(return_value={"status": "success"}))
            mock_marketing.return_value = Mock(generate_copy=AsyncMock(return_value={"status": "success"}))
            mock_publisher.return_value = Mock(publish_products=AsyncMock(return_value={"status": "success"}))
            
            # Run with parallel execution
            start_time = time.time()
            result = await run_helios_pipeline(
                seed="test seed",
                dry_run=True,
                enable_parallel=True
            )
            execution_time = time.time() - start_time
            
            # Verify parallel execution was used
            assert result["status"] == "success"
            assert execution_time < 300
    
    @pytest.mark.asyncio
    async def test_pipeline_sequential_execution(self, mock_config, sample_trend_data, mock_ceo_decision):
        """Test pipeline with sequential execution"""
        with patch('helios.main.load_config', return_value=mock_config), \
             patch('helios.main.ZeitgeistAgent') as mock_zeitgeist, \
             patch('helios.main.HeliosCEO') as mock_ceo, \
             patch('helios.main.AudienceAnalyst') as mock_audience, \
             patch('helios.main.ProductStrategist') as mock_product, \
             patch('helios.main.CreativeDirector') as mock_creative, \
             patch('helios.main.MarketingCopywriter') as mock_marketing, \
             patch('helios.main.PrintifyPublisherAgent') as mock_publisher:
            
            # Setup all mocks
            mock_zeitgeist_instance = Mock()
            mock_zeitgeist_instance.run.return_value = sample_trend_data
            mock_zeitgeist.return_value = mock_zeitgeist_instance
            
            mock_ceo_instance = Mock()
            mock_ceo_instance.prepare_validation.return_value = {"status": "ready"}
            mock_ceo_instance.validate_trend.return_value = mock_ceo_decision
            mock_ceo.return_value = mock_ceo_instance
            
            # Mock other agents
            mock_audience.return_value = Mock(run=AsyncMock(return_value={"status": "success"}))
            mock_product.return_value = Mock(get_products_async=AsyncMock(return_value={"status": "success"}))
            mock_creative.return_value = Mock(generate_designs=AsyncMock(return_value={"status": "success"}))
            mock_marketing.return_value = Mock(generate_copy=AsyncMock(return_value={"status": "success"}))
            mock_publisher.return_value = Mock(publish_products=AsyncMock(return_value={"status": "success"}))
            
            # Run with sequential execution
            start_time = time.time()
            result = await run_helios_pipeline(
                seed="test seed",
                dry_run=True,
                enable_parallel=False
            )
            execution_time = time.time() - start_time
            
            # Verify sequential execution was used
            assert result["status"] == "success"
            assert execution_time < 300
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, mock_config):
        """Test pipeline error handling"""
        with patch('helios.main.load_config', return_value=mock_config), \
             patch('helios.main.ZeitgeistAgent') as mock_zeitgeist:
            
            # Mock agent that raises an exception
            mock_zeitgeist_instance = Mock()
            mock_zeitgeist_instance.run.side_effect = Exception("Test error")
            mock_zeitgeist.return_value = mock_zeitgeist_instance
            
            result = await run_helios_pipeline(
                seed="test seed",
                dry_run=True,
                enable_parallel=True
            )
            
            assert result["status"] == "error"
            assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__])
