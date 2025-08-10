"""
Integration test for full Helios workflow
Tests complete trend-to-product pipeline with all agents
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import json
import os

from helios.agents.ceo import HeliosCEOAgent
from helios.agents.zeitgeist import ZeitgeistFinderAgent
from helios.agents.audience import AudienceAnalystAgent
from helios.agents.product import ProductStrategistAgent
from helios.agents.creative import CreativeDirectorAgent
from helios.agents.marketing import MarketingCopywriterAgent
from helios.agents.ethics import EthicalGuardianAgent
from helios.agents.publish import PrintifyPublisherAgent
from helios.agents.performance import PerformanceAnalyticsAgent

from helios.models.trend_data import TrendData
from helios.models.audience_data import AudienceProfile
from helios.models.product_data import ProductData
from helios.models.analytics_data import AnalyticsData

from helios.services.google_cloud.firestore_client import FirestoreClient
from helios.services.google_cloud.storage_client import StorageClient
from helios.services.google_cloud.vertex_ai_client import VertexAIClient
from helios.services.external_apis.printify_client import PrintifyAPIClient


class TestFullWorkflow:
    """Test suite for complete Helios workflow"""
    
    @pytest.fixture
    def mock_google_cloud_services(self):
        """Mock all Google Cloud services"""
        mock_firestore = Mock()
        mock_storage = Mock()
        mock_vertex_ai = Mock()
        mock_pubsub = Mock()
        mock_cloud_tasks = Mock()
        
        # Mock Firestore collections
        mock_trends_collection = Mock()
        mock_audience_collection = Mock()
        mock_products_collection = Mock()
        mock_analytics_collection = Mock()
        
        mock_firestore.collection.side_effect = lambda name: {
            'trend_discoveries': mock_trends_collection,
            'audience_profiles': mock_audience_collection,
            'product_strategies': mock_products_collection,
            'analytics_data': mock_analytics_collection
        }.get(name, Mock())
        
        # Mock Storage operations
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_storage.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.upload_from_string.return_value = None
        mock_blob.public_url = "https://storage.googleapis.com/test-bucket/test-image.png"
        
        # Mock Vertex AI
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Generated content"
        mock_model.generate_content.return_value = mock_response
        mock_vertex_ai.get_model.return_value = mock_model
        
        # Mock Pub/Sub
        mock_topic = Mock()
        mock_pubsub.topic.return_value = mock_topic
        mock_topic.publish.return_value = Mock()
        
        # Mock Cloud Tasks
        mock_queue = Mock()
        mock_cloud_tasks.queue.return_value = mock_queue
        mock_queue.create_task.return_value = Mock()
        
        return {
            'firestore': mock_firestore,
            'storage': mock_storage,
            'vertex_ai': mock_vertex_ai,
            'pubsub': mock_pubsub,
            'cloud_tasks': mock_cloud_tasks
        }
    
    @pytest.fixture
    def mock_external_apis(self):
        """Mock external API services"""
        mock_printify = Mock()
        mock_etsy = Mock()
        mock_mcp = Mock()
        
        # Mock Printify API
        mock_printify.upload_image.return_value = {'id': 'img_001', 'url': 'https://printify.com/img_001.jpg'}
        mock_printify.create_product.return_value = {'id': 'prod_001', 'status': 'draft'}
        mock_printify.publish_product.return_value = {'id': 'prod_001', 'status': 'published'}
        
        # Mock Etsy API
        mock_etsy.create_listing.return_value = {'listing_id': 'etsy_001', 'status': 'active'}
        
        # Mock MCP tools
        mock_mcp.google_trends.return_value = {'trends': ['sustainable fashion', 'eco-friendly']}
        mock_mcp.social_scanner.return_value = {'mentions': 1500, 'sentiment': 'positive'}
        mock_mcp.news_analyzer.return_value = {'articles': 25, 'relevance': 0.85}
        
        return {
            'printify': mock_printify,
            'etsy': mock_etsy,
            'mcp': mock_mcp
        }
    
    @pytest.fixture
    def sample_trend_data(self):
        """Sample trend data for testing"""
        return TrendData(
            name="Sustainable Fashion Movement",
            description="Growing trend towards eco-friendly clothing and accessories",
            opportunity_score=8.7,
            urgency=7.5,
            category="fashion",
            keywords=["sustainable", "eco-friendly", "organic", "ethical", "fashion"],
            emotional_drivers=["environmental consciousness", "social responsibility", "individual expression"],
            cultural_context="Global climate awareness and social media influence",
            discovered_at=datetime.now(),
            source="google_trends",
            volume_trend="rising",
            competition_level="medium"
        )
    
    @pytest.fixture
    def sample_audience_profile(self):
        """Sample audience profile for testing"""
        return AudienceProfile(
            primary_segment="Eco-conscious Millennials",
            age_range="25-40",
            income_level="middle_upper",
            interests=["sustainability", "fashion", "social_impact", "technology"],
            values=["environmental protection", "ethical consumption", "social responsibility"],
            confidence_score=0.85,
            market_size="large",
            purchase_intent="high",
            preferred_channels=["instagram", "tiktok", "etsy"],
            content_preferences=["visual", "storytelling", "authentic"],
            price_sensitivity="medium"
        )
    
    @pytest.fixture
    def sample_product_strategy(self):
        """Sample product strategy for testing"""
        return {
            'products': [
                {
                    'id': 'prod_001',
                    'name': 'Sustainable Fashion T-Shirt',
                    'category': 'clothing',
                    'base_price': 25.00,
                    'target_margin': 0.35,
                    'print_areas': ['front', 'back'],
                    'variants': ['S', 'M', 'L', 'XL']
                }
            ],
            'pricing': {
                'base_price': 25.00,
                'target_margin': 0.35,
                'final_price': 38.46,
                'profit_per_unit': 13.46
            },
            'collection_plan': {
                'sequence': [1],
                'timing': {'launch_interval': 7},
                'themes': ['sustainability', 'eco-consciousness']
            },
            'emotional_hooks': [
                {
                    'emotion': 'pride',
                    'trigger': 'environmental impact',
                    'intensity': 8.5
                }
            ]
        }
    
    @pytest.fixture
    def sample_design_output(self):
        """Sample design output for testing"""
        return {
            'design_concept': {
                'theme': 'sustainability',
                'style': 'modern',
                'color_scheme': ['green', 'blue', 'white'],
                'visual_elements': ['leaf', 'recycle', 'nature'],
                'brand_alignment': 8.7
            },
            'image_data': b"fake_image_data",
            'image_url': "https://storage.googleapis.com/test-bucket/sustainable-tshirt.png",
            'safety_score': 9.8,
            'quality_score': 9.2,
            'print_optimized': True
        }
    
    @pytest.fixture
    def sample_marketing_copy(self):
        """Sample marketing copy for testing"""
        return {
            'product_title': "Eco-Conscious Sustainable Fashion T-Shirt",
            'product_description': "Make a statement for the planet with our premium eco-friendly t-shirt. Crafted with sustainable materials and featuring inspiring environmental designs.",
            'emotional_benefits': [
                "Feel proud of your environmental impact",
                "Join a community of conscious consumers",
                "Express your values through fashion"
            ],
            'social_proof': "Join 10,000+ eco-conscious fashion lovers",
            'urgency_message': "Limited edition - Only 100 pieces available",
            'call_to_action': "Shop Now - Make a Difference",
            'seo_keywords': ["sustainable fashion", "eco-friendly t-shirt", "environmental clothing"]
        }
    
    @pytest.fixture
    def helios_ceo(self, mock_google_cloud_services, mock_external_apis):
        """Helios CEO Agent instance with mocked dependencies"""
        with patch('helios.agents.ceo.FirestoreClient', return_value=mock_google_cloud_services['firestore']), \
             patch('helios.agents.ceo.StorageClient', return_value=mock_google_cloud_services['storage']), \
             patch('helios.agents.ceo.VertexAIClient', return_value=mock_google_cloud_services['vertex_ai']), \
             patch('helios.agents.ceo.PubSubClient', return_value=mock_google_cloud_services['pubsub']), \
             patch('helios.agents.ceo.CloudTasksClient', return_value=mock_google_cloud_services['cloud_tasks']):
            
            agent = HeliosCEOAgent()
            return agent
    
    @pytest.fixture
    def zeitgeist_finder(self, mock_google_cloud_services, mock_external_apis):
        """Zeitgeist Finder Agent instance with mocked dependencies"""
        with patch('helios.agents.zeitgeist.FirestoreClient', return_value=mock_google_cloud_services['firestore']), \
             patch('helios.agents.zeitgeist.MCPClient', return_value=mock_external_apis['mcp']), \
             patch('helios.agents.zeitgeist.VertexAIClient', return_value=mock_google_cloud_services['vertex_ai']):
            
            agent = ZeitgeistFinderAgent()
            return agent
    
    @pytest.fixture
    def audience_analyst(self, mock_google_cloud_services, mock_external_apis):
        """Audience Analyst Agent instance with mocked dependencies"""
        with patch('helios.agents.audience.FirestoreClient', return_value=mock_google_cloud_services['firestore']), \
             patch('helios.agents.audience.VertexAIClient', return_value=mock_google_cloud_services['vertex_ai']), \
             patch('helios.agents.audience.MCPClient', return_value=mock_external_apis['mcp']):
            
            agent = AudienceAnalystAgent()
            return agent
    
    @pytest.fixture
    def product_strategist(self, mock_google_cloud_services):
        """Product Strategist Agent instance with mocked dependencies"""
        with patch('helios.agents.product.FirestoreClient', return_value=mock_google_cloud_services['firestore']), \
             patch('helios.agents.product.VertexAIClient', return_value=mock_google_cloud_services['vertex_ai']):
            
            agent = ProductStrategistAgent()
            return agent
    
    @pytest.fixture
    def creative_director(self, mock_google_cloud_services):
        """Creative Director Agent instance with mocked dependencies"""
        with patch('helios.agents.creative.VertexAIClient', return_value=mock_google_cloud_services['vertex_ai']), \
             patch('helios.agents.creative.StorageClient', return_value=mock_google_cloud_services['storage']):
            
            agent = CreativeDirectorAgent()
            return agent
    
    @pytest.fixture
    def marketing_copywriter(self, mock_google_cloud_services):
        """Marketing Copywriter Agent instance with mocked dependencies"""
        with patch('helios.agents.marketing.VertexAIClient', return_value=mock_google_cloud_services['vertex_ai']):
            
            agent = MarketingCopywriterAgent()
            return agent
    
    @pytest.fixture
    def ethical_guardian(self, mock_google_cloud_services):
        """Ethical Guardian Agent instance with mocked dependencies"""
        with patch('helios.agents.ethics.VertexAIClient', return_value=mock_google_cloud_services['vertex_ai']):
            
            agent = EthicalGuardianAgent()
            return agent
    
    @pytest.fixture
    def printify_publisher(self, mock_google_cloud_services, mock_external_apis):
        """Printify Publisher Agent instance with mocked dependencies"""
        with patch('helios.agents.publish.FirestoreClient', return_value=mock_google_cloud_services['firestore']), \
             patch('helios.agents.publish.StorageClient', return_value=mock_google_cloud_services['storage']), \
             patch('helios.agents.publish.PrintifyAPIClient', return_value=mock_external_apis['printify']):
            
            agent = PrintifyPublisherAgent()
            return agent
    
    @pytest.fixture
    def performance_analytics(self, mock_google_cloud_services):
        """Performance Analytics Agent instance with mocked dependencies"""
        with patch('helios.agents.performance.FirestoreClient', return_value=mock_google_cloud_services['firestore']), \
             patch('helios.agents.performance.StorageClient', return_value=mock_google_cloud_services['storage']):
            
            agent = PerformanceAnalyticsAgent()
            return agent
    
    @pytest.mark.asyncio
    async def test_complete_workflow_execution(self, helios_ceo, zeitgeist_finder, audience_analyst, 
                                             product_strategist, creative_director, marketing_copywriter,
                                             ethical_guardian, printify_publisher, performance_analytics,
                                             sample_trend_data, sample_audience_profile, 
                                             sample_product_strategy, sample_design_output, 
                                             sample_marketing_copy):
        """Test complete workflow from trend discovery to product publication"""
        
        # Step 1: Trend Discovery
        print("Step 1: Discovering trends...")
        discovered_trends = await zeitgeist_finder.discover_trends()
        
        assert discovered_trends is not None
        assert len(discovered_trends) > 0
        assert discovered_trends[0].opportunity_score >= 7.0
        
        # Step 2: Audience Analysis
        print("Step 2: Analyzing audience...")
        audience_profile = await audience_analyst.analyze_audience(sample_trend_data)
        
        assert audience_profile is not None
        assert audience_profile.confidence_score >= 7.0
        assert audience_profile.market_size in ["small", "medium", "large"]
        
        # Step 3: Product Strategy
        print("Step 3: Developing product strategy...")
        product_strategy = await product_strategist.develop_strategy(sample_trend_data, audience_profile)
        
        assert product_strategy is not None
        assert 'products' in product_strategy
        assert 'pricing' in product_strategy
        assert product_strategy['pricing']['target_margin'] >= 0.35
        
        # Step 4: Creative Design
        print("Step 4: Generating creative designs...")
        design_output = await creative_director.generate_designs(
            product_strategy['products'][0], audience_profile
        )
        
        assert design_output is not None
        assert 'image_data' in design_output
        assert 'design_concept' in design_output
        assert design_output['safety_score'] >= 8.0
        assert design_output['quality_score'] >= 8.0
        
        # Step 5: Marketing Copy
        print("Step 5: Creating marketing copy...")
        marketing_copy = await marketing_copywriter.generate_copy(
            product_strategy['products'][0], audience_profile, sample_trend_data
        )
        
        assert marketing_copy is not None
        assert 'product_title' in marketing_copy
        assert 'product_description' in marketing_copy
        assert 'emotional_benefits' in marketing_copy
        
        # Step 6: Ethical Review
        print("Step 6: Conducting ethical review...")
        ethical_review = await ethical_guardian.review_content(
            marketing_copy, design_output, audience_profile
        )
        
        assert ethical_review is not None
        assert 'is_approved' in ethical_review
        assert 'safety_score' in ethical_review
        
        # Step 7: Product Publication
        print("Step 7: Publishing product...")
        if ethical_review['is_approved']:
            publication_result = await printify_publisher.publish_product(
                product_strategy, design_output, marketing_copy
            )
            
            assert publication_result is not None
            assert 'status' in publication_result
            assert 'product_id' in publication_result
        
        # Step 8: Performance Analytics
        print("Step 8: Analyzing performance...")
        analytics_result = await performance_analytics.analyze_performance(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        assert analytics_result is not None
        assert 'execution_time' in analytics_result
        assert 'success_rate' in analytics_result
        assert 'roi_calculation' in analytics_result
        
        # Verify performance targets
        assert analytics_result['execution_time'] < 300  # < 5 minutes
        assert analytics_result['success_rate'] > 0.85  # > 85%
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, helios_ceo, zeitgeist_finder, audience_analyst):
        """Test workflow error handling and recovery"""
        
        # Mock trend discovery failure
        with patch.object(zeitgeist_finder, 'discover_trends', side_effect=Exception("API failure")):
            try:
                trends = await zeitgeist_finder.discover_trends()
                assert False, "Should have raised exception"
            except Exception as e:
                assert "API failure" in str(e)
        
        # Mock audience analysis failure
        with patch.object(audience_analyst, 'analyze_audience', side_effect=Exception("Data unavailable")):
            try:
                audience = await audience_analyst.analyze_audience(sample_trend_data)
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Data unavailable" in str(e)
    
    @pytest.mark.asyncio
    async def test_workflow_quality_gates(self, helios_ceo, ethical_guardian, sample_design_output, sample_marketing_copy):
        """Test quality gate enforcement throughout workflow"""
        
        # Test content safety gate
        safety_result = await ethical_guardian.review_content(
            sample_marketing_copy, sample_design_output, sample_audience_profile
        )
        
        assert safety_result['is_approved'] == True
        assert safety_result['safety_score'] >= 8.0
        
        # Test quality threshold enforcement
        if safety_result['safety_score'] < 7.0:
            assert safety_result['is_approved'] == False
            assert 'blocked_reasons' in safety_result
    
    @pytest.mark.asyncio
    async def test_workflow_performance_optimization(self, helios_ceo, zeitgeist_finder, audience_analyst):
        """Test workflow performance optimization features"""
        
        start_time = datetime.now()
        
        # Execute workflow steps in parallel where possible
        trend_task = asyncio.create_task(zeitgeist_finder.discover_trends())
        audience_task = asyncio.create_task(audience_analyst.analyze_audience(sample_trend_data))
        
        # Wait for both to complete
        trends, audience = await asyncio.gather(trend_task, audience_task)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Verify parallel execution improved performance
        assert execution_time < 60  # Should complete within 1 minute
        assert trends is not None
        assert audience is not None
    
    @pytest.mark.asyncio
    async def test_workflow_data_persistence(self, helios_ceo, mock_google_cloud_services, sample_trend_data):
        """Test data persistence throughout workflow"""
        
        # Mock Firestore operations
        mock_firestore = mock_google_cloud_services['firestore']
        mock_collection = Mock()
        mock_document = Mock()
        mock_firestore.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_document
        
        # Test trend data persistence
        await zeitgeist_finder.save_trend_discovery(sample_trend_data)
        
        # Verify Firestore was called
        mock_firestore.collection.assert_called_with('trend_discoveries')
        mock_collection.document.assert_called()
    
    @pytest.mark.asyncio
    async def test_workflow_monitoring_and_alerting(self, helios_ceo, performance_analytics):
        """Test workflow monitoring and alerting capabilities"""
        
        # Test performance monitoring
        performance_metrics = await performance_analytics.collect_metrics()
        
        assert performance_metrics is not None
        assert 'execution_time' in performance_metrics
        assert 'success_rate' in performance_metrics
        assert 'error_rate' in performance_metrics
        
        # Test alert generation for performance issues
        if performance_metrics['success_rate'] < 0.85:
            alerts = await performance_analytics.generate_alerts(performance_metrics)
            assert len(alerts) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_scalability(self, helios_ceo, zeitgeist_finder, audience_analyst):
        """Test workflow scalability with multiple concurrent operations"""
        
        # Test handling multiple trends simultaneously
        trend_batch = [sample_trend_data] * 5  # 5 concurrent trends
        
        # Process trends in parallel
        audience_tasks = [
            audience_analyst.analyze_audience(trend) 
            for trend in trend_batch
        ]
        
        audience_results = await asyncio.gather(*audience_tasks)
        
        assert len(audience_results) == 5
        assert all(result is not None for result in audience_results)
        
        # Verify all meet quality thresholds
        for result in audience_results:
            assert result.confidence_score >= 7.0
    
    @pytest.mark.asyncio
    async def test_workflow_integration_with_external_services(self, helios_ceo, printify_publisher, mock_external_apis):
        """Test integration with external services (Printify, Etsy)"""
        
        # Test Printify integration
        printify_client = mock_external_apis['printify']
        
        # Mock successful product creation
        product_data = {
            'name': 'Test Product',
            'description': 'Test Description',
            'price': 25.00,
            'image_url': 'https://example.com/image.jpg'
        }
        
        publication_result = await printify_publisher.publish_product(
            {'products': [product_data]}, 
            {'image_url': 'https://example.com/image.jpg'}, 
            {'product_title': 'Test Product'}
        )
        
        assert publication_result is not None
        assert 'status' in publication_result
        
        # Verify Printify API was called
        printify_client.create_product.assert_called()
    
    def test_workflow_configuration_loading(self, helios_ceo):
        """Test workflow configuration loading and validation"""
        
        # Test configuration loading
        config = helios_ceo.load_configuration()
        
        assert config is not None
        assert 'performance_targets' in config
        assert 'quality_gates' in config
        assert 'agent_configs' in config
        
        # Verify performance targets
        performance_targets = config['performance_targets']
        assert performance_targets['execution_time'] < 300  # < 5 minutes
        assert performance_targets['success_rate'] > 0.85  # > 85%
        assert performance_targets['roi_target'] > 3.0  # > 300%
    
    def test_workflow_environment_validation(self, helios_ceo):
        """Test workflow environment validation"""
        
        # Test required environment variables
        required_vars = [
            'GOOGLE_CLOUD_PROJECT',
            'PRINTIFY_API_TOKEN',
            'GOOGLE_SHEETS_TRACKING_ID'
        ]
        
        for var in required_vars:
            assert os.getenv(var) is not None, f"Missing required environment variable: {var}"
    
    @pytest.mark.asyncio
    async def test_workflow_end_to_end_validation(self, helios_ceo, zeitgeist_finder, audience_analyst,
                                                product_strategist, creative_director, marketing_copywriter,
                                                ethical_guardian, printify_publisher, performance_analytics):
        """Test complete end-to-end workflow validation"""
        
        print("Starting end-to-end workflow validation...")
        
        # Execute complete workflow
        workflow_result = await helios_ceo.execute_workflow()
        
        assert workflow_result is not None
        assert 'status' in workflow_result
        assert 'execution_time' in workflow_result
        assert 'products_created' in workflow_result
        assert 'quality_score' in workflow_result
        
        # Verify workflow success
        assert workflow_result['status'] == 'completed'
        assert workflow_result['execution_time'] < 300  # < 5 minutes
        assert workflow_result['quality_score'] >= 7.0
        assert workflow_result['products_created'] > 0
        
        print(f"Workflow completed successfully in {workflow_result['execution_time']} seconds")
        print(f"Quality score: {workflow_result['quality_score']}")
        print(f"Products created: {workflow_result['products_created']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
