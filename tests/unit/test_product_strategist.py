"""
Unit tests for Product Strategist Agent
Tests product selection, strategy formulation, and psychological enhancement
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from helios.agents.product import ProductStrategistAgent
from helios.models.product_data import ProductData, ProductVariant, PricingStrategy
from helios.models.audience_data import AudienceProfile
from helios.models.trend_data import TrendData


class TestProductStrategistAgent:
    """Test suite for Product Strategist Agent"""
    
    @pytest.fixture
    def mock_firestore_client(self):
        """Mock Firestore client"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_document = Mock()
        
        # Mock catalog data
        mock_catalog_data = {
            'products': [
                {
                    'id': 'prod_001',
                    'name': 'Classic T-Shirt',
                    'category': 'clothing',
                    'base_price': 15.00,
                    'print_areas': ['front', 'back'],
                    'variants': ['S', 'M', 'L', 'XL'],
                    'performance_score': 8.5,
                    'last_updated': datetime.now().isoformat()
                },
                {
                    'id': 'prod_002',
                    'name': 'Premium Hoodie',
                    'category': 'clothing',
                    'base_price': 35.00,
                    'print_areas': ['front', 'back', 'sleeve'],
                    'variants': ['S', 'M', 'L', 'XL', '2XL'],
                    'performance_score': 9.2,
                    'last_updated': datetime.now().isoformat()
                }
            ]
        }
        
        mock_document.get.return_value.to_dict.return_value = mock_catalog_data
        mock_collection.document.return_value = mock_document
        mock_client.collection.return_value = mock_collection
        
        return mock_client
    
    @pytest.fixture
    def mock_bigquery_client(self):
        """Mock BigQuery client"""
        mock_client = Mock()
        mock_query_job = Mock()
        
        # Mock performance data
        mock_performance_data = pd.DataFrame({
            'product_id': ['prod_001', 'prod_002'],
            'avg_order_value': [45.00, 78.50],
            'conversion_rate': [0.045, 0.062],
            'profit_margin': [0.38, 0.42],
            'trend_relevance': [0.72, 0.89]
        })
        
        mock_query_job.to_dataframe.return_value = mock_performance_data
        mock_client.query.return_value = mock_query_job
        
        return mock_client
    
    @pytest.fixture
    def mock_vertex_ai_client(self):
        """Mock Vertex AI client"""
        mock_client = Mock()
        mock_model = Mock()
        
        # Mock brand messaging response
        mock_response = Mock()
        mock_response.text = "Empower your individuality with our premium designs"
        mock_model.generate_content.return_value = mock_response
        
        mock_client.get_model.return_value = mock_model
        
        return mock_client
    
    @pytest.fixture
    def product_strategist(self, mock_firestore_client, mock_bigquery_client, mock_vertex_ai_client):
        """Product Strategist Agent instance with mocked dependencies"""
        with patch('helios.agents.product.FirestoreClient', return_value=mock_firestore_client), \
             patch('helios.agents.product.BigQueryClient', return_value=mock_bigquery_client), \
             patch('helios.agents.product.VertexAIClient', return_value=mock_vertex_ai_client):
            
            agent = ProductStrategistAgent()
            return agent
    
    @pytest.fixture
    def sample_trend(self):
        """Sample trend data for testing"""
        return TrendData(
            name="Sustainable Fashion Movement",
            description="Growing trend towards eco-friendly clothing",
            opportunity_score=8.7,
            urgency=7.5,
            category="fashion",
            keywords=["sustainable", "eco-friendly", "organic", "ethical"],
            emotional_drivers=["environmental consciousness", "social responsibility"],
            cultural_context="Global climate awareness",
            discovered_at=datetime.now()
        )
    
    @pytest.fixture
    def sample_audience(self):
        """Sample audience profile for testing"""
        return AudienceProfile(
            primary_segment="Eco-conscious Millennials",
            age_range="25-40",
            income_level="middle_upper",
            interests=["sustainability", "fashion", "social_impact"],
            values=["environmental protection", "ethical consumption"],
            confidence_score=0.85,
            market_size="large",
            purchase_intent="high"
        )
    
    def test_initialization(self, product_strategist):
        """Test agent initialization"""
        assert product_strategist is not None
        assert hasattr(product_strategist, 'firestore_client')
        assert hasattr(product_strategist, 'bigquery_client')
        assert hasattr(product_strategist, 'vertex_ai_client')
    
    def test_get_catalog_cache(self, product_strategist, mock_firestore_client):
        """Test retrieving catalog cache from Firestore"""
        catalog = product_strategist.get_catalog_cache()
        
        assert catalog is not None
        assert 'products' in catalog
        assert len(catalog['products']) == 2
        assert catalog['products'][0]['id'] == 'prod_001'
        
        # Verify Firestore was called
        mock_firestore_client.collection.assert_called_with('printify_catalog')
    
    def test_get_performance_data(self, product_strategist, mock_bigquery_client):
        """Test retrieving performance data from BigQuery"""
        performance_data = product_strategist.get_performance_data()
        
        assert performance_data is not None
        assert isinstance(performance_data, pd.DataFrame)
        assert len(performance_data) == 2
        assert 'profit_margin' in performance_data.columns
        
        # Verify BigQuery was called
        mock_bigquery_client.query.assert_called()
    
    def test_calculate_margin_prediction(self, product_strategist):
        """Test ML-powered margin prediction"""
        base_price = 25.00
        target_margin = 0.35
        
        predicted_price = product_strategist.calculate_margin_prediction(base_price, target_margin)
        
        assert predicted_price > base_price
        assert predicted_price == pytest.approx(38.46, rel=0.01)  # 25 / (1 - 0.35)
    
    def test_select_products(self, product_strategist, sample_trend, sample_audience):
        """Test product selection based on trend and audience"""
        selected_products = product_strategist.select_products(sample_trend, sample_audience)
        
        assert selected_products is not None
        assert len(selected_products) > 0
        
        # Verify products meet criteria
        for product in selected_products:
            assert product.performance_score >= 7.0
            assert product.category in ['clothing', 'fashion']
    
    def test_optimize_pricing_strategy(self, product_strategist, sample_trend):
        """Test pricing strategy optimization"""
        base_price = 20.00
        target_margin = 0.35
        
        pricing_strategy = product_strategist.optimize_pricing_strategy(
            base_price, target_margin, sample_trend
        )
        
        assert pricing_strategy is not None
        assert pricing_strategy.base_price == base_price
        assert pricing_strategy.target_margin == target_margin
        assert pricing_strategy.final_price > base_price
        
        # Verify margin calculation
        calculated_margin = (pricing_strategy.final_price - base_price) / pricing_strategy.final_price
        assert calculated_margin >= target_margin
    
    def test_generate_brand_messaging(self, product_strategist, sample_trend, sample_audience):
        """Test brand messaging generation using Vertex AI"""
        messaging = product_strategist.generate_brand_messaging(sample_trend, sample_audience)
        
        assert messaging is not None
        assert isinstance(messaging, str)
        assert len(messaging) > 0
        
        # Verify Vertex AI was called
        product_strategist.vertex_ai_client.get_model.assert_called()
    
    def test_plan_collection_strategy(self, product_strategist, sample_trend):
        """Test collection strategy planning"""
        collection_plan = product_strategist.plan_collection_strategy(sample_trend)
        
        assert collection_plan is not None
        assert 'sequence' in collection_plan
        assert 'timing' in collection_plan
        assert 'themes' in collection_plan
        
        # Verify sequence planning
        assert len(collection_plan['sequence']) > 0
        assert collection_plan['timing']['launch_interval'] > 0
    
    def test_identify_emotional_hooks(self, product_strategist, sample_trend, sample_audience):
        """Test emotional hook identification"""
        emotional_hooks = product_strategist.identify_emotional_hooks(sample_trend, sample_audience)
        
        assert emotional_hooks is not None
        assert isinstance(emotional_hooks, list)
        assert len(emotional_hooks) > 0
        
        # Verify emotional relevance
        for hook in emotional_hooks:
            assert 'emotion' in hook
            assert 'trigger' in hook
            assert 'intensity' in hook
    
    def test_validate_product_strategy(self, product_strategist):
        """Test product strategy validation"""
        strategy = {
            'products': ['prod_001', 'prod_002'],
            'pricing': {'base_price': 25.00, 'target_margin': 0.35},
            'collection_plan': {'sequence': [1, 2], 'timing': {'launch_interval': 7}},
            'emotional_hooks': [{'emotion': 'pride', 'trigger': 'sustainability'}]
        }
        
        validation_result = product_strategist.validate_product_strategy(strategy)
        
        assert validation_result is not None
        assert 'is_valid' in validation_result
        assert 'issues' in validation_result
        assert 'recommendations' in validation_result
    
    def test_generate_marketing_angles(self, product_strategist, sample_trend, sample_audience):
        """Test marketing angle generation"""
        marketing_angles = product_strategist.generate_marketing_angles(sample_trend, sample_audience)
        
        assert marketing_angles is not None
        assert isinstance(marketing_angles, list)
        assert len(marketing_angles) > 0
        
        # Verify angle components
        for angle in marketing_angles:
            assert 'theme' in angle
            assert 'message' in angle
            assert 'target_emotion' in angle
            assert 'urgency_factor' in angle
    
    def test_calculate_roi_prediction(self, product_strategist, sample_trend):
        """Test ROI prediction calculation"""
        investment = 1000.00
        expected_revenue = 2500.00
        
        roi_prediction = product_strategist.calculate_roi_prediction(investment, expected_revenue)
        
        assert roi_prediction is not None
        assert 'roi_percentage' in roi_prediction
        assert 'payback_period' in roi_prediction
        assert 'risk_level' in roi_prediction
        
        # Verify ROI calculation
        expected_roi = ((expected_revenue - investment) / investment) * 100
        assert roi_prediction['roi_percentage'] == pytest.approx(expected_roi, rel=0.01)
    
    def test_error_handling_catalog_failure(self, product_strategist, mock_firestore_client):
        """Test error handling when catalog retrieval fails"""
        # Mock Firestore error
        mock_firestore_client.collection.side_effect = Exception("Firestore connection failed")
        
        with pytest.raises(Exception):
            product_strategist.get_catalog_cache()
    
    def test_error_handling_performance_data_failure(self, product_strategist, mock_bigquery_client):
        """Test error handling when performance data retrieval fails"""
        # Mock BigQuery error
        mock_bigquery_client.query.side_effect = Exception("BigQuery query failed")
        
        with pytest.raises(Exception):
            product_strategist.get_performance_data()
    
    def test_fallback_to_cached_data(self, product_strategist):
        """Test fallback to cached data when fresh data unavailable"""
        # Mock fresh data failure
        with patch.object(product_strategist, 'get_catalog_cache', side_effect=Exception("Fresh data failed")):
            cached_catalog = product_strategist.get_catalog_cache_fallback()
            
            assert cached_catalog is not None
            # Should return last known good data
    
    def test_performance_optimization(self, product_strategist):
        """Test performance optimization features"""
        # Test parallel processing
        start_time = datetime.now()
        
        with product_strategist.parallel_processing():
            # Simulate parallel operations
            pass
        
        execution_time = (datetime.now() - start_time).total_seconds()
        assert execution_time < 5.0  # Should complete within 5 seconds
    
    def test_quality_gate_enforcement(self, product_strategist):
        """Test quality gate enforcement"""
        strategy = {
            'products': ['prod_001'],
            'pricing': {'base_price': 25.00, 'target_margin': 0.35},
            'collection_plan': {'sequence': [1], 'timing': {'launch_interval': 7}},
            'emotional_hooks': [{'emotion': 'pride', 'trigger': 'sustainability'}]
        }
        
        quality_result = product_strategist.enforce_quality_gates(strategy)
        
        assert quality_result is not None
        assert 'passed' in quality_result
        assert 'score' in quality_result
        assert 'issues' in quality_result
        
        # Verify quality thresholds
        if quality_result['passed']:
            assert quality_result['score'] >= 7.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
