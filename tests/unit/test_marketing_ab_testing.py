"""Unit tests for A/B testing framework and adaptive learning system"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch
import sys
import os

# Add the helios package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from helios.agents.marketing import (
    ABTestingFramework, 
    AdaptiveLearningSystem,
    ABTestVariant,
    ABTestResult,
    ABTestExperiment,
    LearningParameter,
    AdaptiveLearningConfig
)


class TestABTestingFramework:
    """Test A/B testing framework functionality"""
    
    @pytest.fixture
    def ab_framework(self):
        """Create A/B testing framework instance"""
        return ABTestingFramework()
    
    @pytest.fixture
    def sample_experiment_config(self):
        """Sample experiment configuration"""
        return {
            "name": "Headline A/B Test",
            "description": "Testing different headline styles",
            "variants": [
                {
                    "name": "Control - Professional",
                    "content": {"headline_style": "professional", "tone": "formal"}
                },
                {
                    "name": "Variant A - Emotional",
                    "content": {"headline_style": "emotional", "tone": "passionate"}
                },
                {
                    "name": "Variant B - Urgent",
                    "content": {"headline_style": "urgent", "tone": "time-sensitive"}
                }
            ],
            "primary_metric": "conversion_rate",
            "confidence_level": 0.95,
            "minimum_sample_size": 100
        }
    
    @pytest.mark.asyncio
    async def test_create_experiment(self, ab_framework, sample_experiment_config):
        """Test experiment creation"""
        experiment = await ab_framework.create_experiment(sample_experiment_config)
        
        assert experiment is not None
        assert experiment.experiment_name == "Headline A/B Test"
        assert len(experiment.variants) == 3
        assert experiment.primary_metric == "conversion_rate"
        assert experiment.confidence_level == 0.95
        
        # Check variants
        control_variant = next(v for v in experiment.variants if v.is_control)
        assert control_variant.variant_name == "Control - Professional"
        assert control_variant.traffic_allocation == pytest.approx(1/3, rel=1e-2)
    
    @pytest.mark.asyncio
    async def test_record_interactions(self, ab_framework, sample_experiment_config):
        """Test recording user interactions"""
        experiment = await ab_framework.create_experiment(sample_experiment_config)
        experiment_id = experiment.experiment_id
        variant_id = experiment.variants[0].variant_id
        
        # Record various interactions
        await ab_framework.record_interaction(experiment_id, variant_id, "impression")
        await ab_framework.record_interaction(experiment_id, variant_id, "click")
        await ab_framework.record_interaction(experiment_id, variant_id, "conversion", 25.0)
        
        # Check results
        results = ab_framework.results[experiment_id][variant_id]
        assert results.impressions == 1
        assert results.clicks == 1
        assert results.conversions == 1
        assert results.revenue == 25.0
        assert results.ctr == 1.0
        assert results.conversion_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_get_winning_variant_insufficient_data(self, ab_framework, sample_experiment_config):
        """Test winning variant determination with insufficient data"""
        experiment = await ab_framework.create_experiment(sample_experiment_config)
        experiment_id = experiment.experiment_id
        
        # Add some interactions but not enough for statistical significance
        for variant in experiment.variants:
            variant_id = variant.variant_id
            await ab_framework.record_interaction(experiment_id, variant_id, "impression", 50)
            await ab_framework.record_interaction(experiment_id, variant_id, "click", 5)
        
        winning_variant = await ab_framework.get_winning_variant(experiment_id)
        assert winning_variant is None  # Not enough data yet
    
    @pytest.mark.asyncio
    async def test_get_experiment_summary(self, ab_framework, sample_experiment_config):
        """Test experiment summary generation"""
        experiment = await ab_framework.create_experiment(sample_experiment_config)
        experiment_id = experiment.experiment_id
        
        # Add some test data
        for variant in experiment.variants:
            variant_id = variant.variant_id
            await ab_framework.record_interaction(experiment_id, variant_id, "impression", 100)
            await ab_framework.record_interaction(experiment_id, variant_id, "click", 10)
        
        summary = await ab_framework.get_experiment_summary(experiment_id)
        
        assert summary["experiment_id"] == experiment_id
        assert summary["experiment_name"] == "Headline A/B Test"
        assert summary["total_impressions"] == 300
        assert len(summary["variants"]) == 3
        
        # Check variant data
        for variant_summary in summary["variants"]:
            assert variant_summary["impressions"] == 100
            assert variant_summary["clicks"] == 10
            assert variant_summary["ctr"] == 0.1


class TestAdaptiveLearningSystem:
    """Test adaptive learning system functionality"""
    
    @pytest.fixture
    def learning_system(self):
        """Create adaptive learning system instance"""
        config = AdaptiveLearningConfig(
            learning_rate=0.01,
            momentum=0.9,
            adaptation_frequency=10,
            history_window=100
        )
        return AdaptiveLearningSystem(config)
    
    @pytest.fixture
    def sample_performance_data(self):
        """Sample performance data"""
        return {
            "performance_score": 0.75,
            "metrics": {
                "impression": 100,
                "click": 15,
                "conversion": 3,
                "revenue": 45.0
            }
        }
    
    def test_parameter_initialization(self, learning_system):
        """Test parameter initialization"""
        params = learning_system.get_optimized_parameters()
        
        expected_params = [
            "content_creativity", "keyword_density", "emotional_intensity",
            "urgency_level", "social_proof_weight", "price_sensitivity",
            "audience_targeting", "platform_optimization"
        ]
        
        for param_name in expected_params:
            assert param_name in params
            assert 0.0 <= params[param_name] <= 1.0
    
    @pytest.mark.asyncio
    async def test_record_performance(self, learning_system, sample_performance_data):
        """Test performance recording"""
        initial_count = learning_system.interaction_count
        
        await learning_system.record_performance(sample_performance_data)
        
        assert learning_system.interaction_count == initial_count + 1
        assert len(learning_system.performance_history) == 1
        
        # Check recorded data
        recorded = learning_system.performance_history[0]
        assert recorded["performance_score"] == 0.75
        assert "timestamp" in recorded
        assert "parameters" in recorded
    
    @pytest.mark.asyncio
    async def test_parameter_adaptation(self, learning_system):
        """Test parameter adaptation after sufficient data"""
        # Add enough performance data to trigger adaptation
        for i in range(15):
            performance_data = {
                "performance_score": 0.5 + (i * 0.02),  # Gradually improving
                "metrics": {"impression": 100 + i}
            }
            await learning_system.record_performance(performance_data)
        
        # Check if parameters were adapted
        initial_params = learning_system.get_optimized_parameters()
        trends = learning_system.get_parameter_trends()
        
        # Should have some parameter trends
        assert len(trends) > 0
        assert all(trend in ["increasing", "decreasing", "stable"] for trend in trends.values())
    
    def test_parameter_update(self):
        """Test individual parameter update logic"""
        param = LearningParameter(
            name="test_param",
            current_value=0.5,
            min_value=0.0,
            max_value=1.0,
            learning_rate=0.01,
            momentum=0.9
        )
        
        initial_value = param.current_value
        
        # Update parameter
        param.update(gradient=0.1, performance_score=0.8)
        
        # Value should change
        assert param.current_value != initial_value
        assert param.current_value >= param.min_value
        assert param.current_value <= param.max_value
        
        # Check history
        assert len(param.history) > 0
        assert param.history[-1] == param.current_value
    
    @pytest.mark.asyncio
    async def test_get_learning_summary(self, learning_system):
        """Test learning summary generation"""
        # Add some performance data
        for i in range(5):
            await learning_system.record_performance({
                "performance_score": 0.6 + (i * 0.05),
                "metrics": {"impression": 100 + i * 10}
            })
        
        summary = await learning_system.get_learning_summary()
        
        assert "total_interactions" in summary
        assert "current_parameters" in summary
        assert "parameter_trends" in summary
        assert "recent_performance" in summary
        
        assert summary["total_interactions"] == 5
        assert len(summary["recent_performance"]) == 5


class TestIntegration:
    """Test integration between A/B testing and adaptive learning"""
    
    @pytest.fixture
    def marketing_copywriter(self):
        """Create marketing copywriter with both systems"""
        from helios.agents.marketing import MarketingCopywriter
        return MarketingCopywriter()
    
    @pytest.fixture
    def sample_creative_batch(self):
        """Sample creative batch data"""
        return {
            "product_name": "Test Product",
            "product_description": "A test product for testing",
            "target_audience": "testers",
            "brand_voice": "professional"
        }
    
    @pytest.fixture
    def sample_ab_test_config(self):
        """Sample A/B test configuration"""
        return {
            "name": "Content Style Test",
            "description": "Testing different content styles",
            "variants": [
                {
                    "name": "Control - Professional",
                    "content": {"headline_style": "professional", "tone": "formal"}
                },
                {
                    "name": "Variant A - Creative",
                    "content": {"headline_style": "creative", "tone": "playful"}
                }
            ],
            "primary_metric": "conversion_rate",
            "minimum_sample_size": 50
        }
    
    @pytest.mark.asyncio
    async def test_ab_test_creation_and_execution(self, marketing_copywriter, 
                                                sample_creative_batch, 
                                                sample_ab_test_config):
        """Test A/B test creation and execution"""
        # Run A/B test
        ab_test_result = await marketing_copywriter.run_ab_test(
            sample_creative_batch, 
            sample_ab_test_config
        )
        
        assert ab_test_result["status"] == "success"
        assert "experiment_id" in ab_test_result
        assert "content_variations" in ab_test_result
        assert len(ab_test_result["content_variations"]) == 2
        
        # Check that each variant has content and parameters
        for variation in ab_test_result["content_variations"]:
            assert "variant_id" in variation
            assert "content" in variation
            assert "parameters_used" in variation
    
    @pytest.mark.asyncio
    async def test_performance_recording_and_learning(self, marketing_copywriter):
        """Test performance recording and learning integration"""
        # Create a mock experiment ID and variant ID
        experiment_id = "test_experiment_123"
        variant_id = "test_variant_456"
        
        # Record performance data
        interaction_data = {
            "impression": 100,
            "click": 15,
            "conversion": 3,
            "revenue": 45.0,
            "engagement": 0.15
        }
        
        result = await marketing_copywriter.record_ab_test_results(
            experiment_id, 
            variant_id, 
            interaction_data
        )
        
        assert result["status"] == "success"
        assert "performance_score" in result
        assert result["performance_score"] > 0.0
    
    @pytest.mark.asyncio
    async def test_learning_insights_generation(self, marketing_copywriter):
        """Test learning insights generation"""
        # Add some performance data to trigger learning
        for i in range(20):
            interaction_data = {
                "impression": 100 + i,
                "click": 10 + i,
                "conversion": 2 + (i // 5),
                "revenue": 30.0 + (i * 2),
                "engagement": 0.1 + (i * 0.01)
            }
            
            await marketing_copywriter.record_ab_test_results(
                "test_exp", 
                "test_var", 
                interaction_data
            )
        
        # Get learning insights
        insights = await marketing_copywriter.get_learning_insights()
        
        assert insights["status"] == "success"
        assert "optimized_parameters" in insights
        assert "actionable_insights" in insights
        assert len(insights["actionable_insights"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
