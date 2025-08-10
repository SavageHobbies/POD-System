"""
Unit tests for AudienceAnalystAgent
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime

from helios.agents.audience import AudienceAnalystAgent
from helios.models.audience_data import AudienceData, DemographicSegment, PsychographicProfile
from helios.services.google_cloud.vertex_ai_client import VertexAIClient
from helios.services.google_cloud.firestore_client import FirestoreClient


class TestAudienceAnalystAgent:
    """Test cases for AudienceAnalystAgent"""
    
    @pytest.fixture
    def mock_vertex_client(self):
        """Mock Vertex AI client"""
        client = Mock(spec=VertexAIClient)
        client.generate_text = AsyncMock()
        client.analyze_sentiment = AsyncMock()
        client.cluster_data = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_firestore_client(self):
        """Mock Firestore client"""
        client = Mock(spec=FirestoreClient)
        client.get_document = AsyncMock()
        client.set_document = AsyncMock()
        client.query_collection = AsyncMock()
        return client
    
    @pytest.fixture
    def agent(self, mock_vertex_client, mock_firestore_client):
        """Create agent instance with mocked dependencies"""
        return AudienceAnalystAgent(
            vertex_client=mock_vertex_client,
            firestore_client=mock_firestore_client,
            config={
                'min_confidence_score': 7.0,
                'rapid_mode_threshold': 8.5,
                'cache_ttl_hours': 24
            }
        )
    
    @pytest.mark.asyncio
    async def test_analyze_audience_success(self, agent, mock_vertex_client):
        """Test successful audience analysis"""
        # Mock Vertex AI response
        mock_vertex_client.generate_text.return_value = {
            'text': 'Young professionals aged 25-35, tech-savvy, value creativity and innovation',
            'confidence': 0.88
        }
        
        mock_vertex_client.cluster_data.return_value = {
            'segments': [
                {'age_group': '25-35', 'confidence': 0.85},
                {'interests': 'technology', 'confidence': 0.90},
                {'values': 'creativity', 'confidence': 0.82}
            ]
        }
        
        trend_query = "AI art generator"
        
        audience_data = await agent.analyze_audience(trend_query)
        
        assert audience_data is not None
        assert audience_data.trend_query == trend_query
        assert audience_data.confidence_score >= 7.0
        assert len(audience_data.demographic_segments) > 0
        assert len(audience_data.psychographic_profiles) > 0
        
        mock_vertex_client.generate_text.assert_called_once()
        mock_vertex_client.cluster_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_audience_rapid_mode(self, agent, mock_vertex_client, mock_firestore_client):
        """Test rapid mode audience analysis"""
        # Mock high urgency trend
        mock_vertex_client.generate_text.return_value = {
            'text': 'High urgency trend analysis',
            'confidence': 0.92
        }
        
        # Mock cached persona data
        mock_firestore_client.get_document.return_value = {
            'demographics': {'age_group': '25-35', 'location': 'US'},
            'psychographics': {'interests': ['technology', 'art'], 'values': ['creativity']},
            'cached_at': datetime.now().isoformat()
        }
        
        trend_query = "Viral meme trend"
        urgency_score = 9.0  # High urgency
        
        audience_data = await agent.analyze_audience(trend_query, urgency_score=urgency_score)
        
        assert audience_data is not None
        assert audience_data.analysis_mode == "rapid"
        assert audience_data.execution_time < 30  # Should be fast
        
        # Should use cached data for rapid mode
        mock_firestore_client.get_document.assert_called()
    
    @pytest.mark.asyncio
    async def test_analyze_audience_low_confidence(self, agent, mock_vertex_client):
        """Test audience analysis with low confidence"""
        mock_vertex_client.generate_text.return_value = {
            'text': 'Unclear audience profile',
            'confidence': 0.45  # Below threshold
        }
        
        trend_query = "Obscure trend"
        
        audience_data = await agent.analyze_audience(trend_query)
        
        # Should return None for low confidence
        assert audience_data is None
    
    @pytest.mark.asyncio
    async def test_create_demographic_segments(self, agent, mock_vertex_client):
        """Test demographic segmentation"""
        mock_vertex_client.cluster_data.return_value = {
            'segments': [
                {'age_group': '18-24', 'confidence': 0.85, 'size': 'large'},
                {'age_group': '25-34', 'confidence': 0.78, 'size': 'medium'},
                {'location': 'US', 'confidence': 0.90, 'size': 'large'}
            ]
        }
        
        segments = await agent.create_demographic_segments("AI art trend")
        
        assert len(segments) == 3
        assert all(isinstance(seg, DemographicSegment) for seg in segments)
        assert any(seg.age_group == '18-24' for seg in segments)
        assert any(seg.location == 'US' for seg in segments)
    
    @pytest.mark.asyncio
    async def test_create_psychographic_profiles(self, agent, mock_vertex_client):
        """Test psychographic profiling"""
        mock_vertex_client.generate_text.return_value = {
            'text': 'Creative individuals who value self-expression and innovation',
            'confidence': 0.87
        }
        
        profiles = await agent.create_psychographic_profiles("Creative trend")
        
        assert len(profiles) > 0
        assert all(isinstance(profile, PsychographicProfile) for profile in profiles)
        assert any('creative' in profile.personality_traits.lower() for profile in profiles)
    
    @pytest.mark.asyncio
    async def test_calculate_audience_size(self, agent):
        """Test audience size calculation"""
        segments = [
            DemographicSegment(age_group='18-24', confidence=0.85, size='large'),
            DemographicSegment(age_group='25-34', confidence=0.78, size='medium'),
            DemographicSegment(location='US', confidence=0.90, size='large')
        ]
        
        total_size = agent.calculate_audience_size(segments)
        
        # Should calculate based on segment sizes and confidence
        assert total_size > 0
        assert isinstance(total_size, int)
    
    @pytest.mark.asyncio
    async def test_validate_audience_data(self, agent):
        """Test audience data validation"""
        # Valid audience data
        valid_data = AudienceData(
            trend_query="Test trend",
            confidence_score=8.5,
            demographic_segments=[
                DemographicSegment(age_group='25-35', confidence=0.85, size='large')
            ],
            psychographic_profiles=[
                PsychographicProfile(personality_traits=['creative'], confidence=0.82)
            ],
            timestamp=datetime.now()
        )
        
        assert agent.validate_audience_data(valid_data) is True
        
        # Invalid data (low confidence)
        invalid_data = AudienceData(
            trend_query="Test trend",
            confidence_score=4.2,  # Below threshold
            demographic_segments=[],
            psychographic_profiles=[],
            timestamp=datetime.now()
        )
        
        assert agent.validate_audience_data(invalid_data) is False
    
    @pytest.mark.asyncio
    async def test_cache_audience_data(self, agent, mock_firestore_client):
        """Test audience data caching"""
        audience_data = AudienceData(
            trend_query="Cached trend",
            confidence_score=8.0,
            demographic_segments=[],
            psychographic_profiles=[],
            timestamp=datetime.now()
        )
        
        await agent.cache_audience_data(audience_data)
        
        mock_firestore_client.set_document.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_cached_audience_data(self, agent, mock_firestore_client):
        """Test retrieving cached audience data"""
        cached_data = {
            'trend_query': 'Cached trend',
            'confidence_score': 8.0,
            'demographic_segments': [],
            'psychographic_profiles': [],
            'timestamp': datetime.now().isoformat()
        }
        
        mock_firestore_client.get_document.return_value = cached_data
        
        result = await agent.get_cached_audience_data("Cached trend")
        
        assert result is not None
        assert result.trend_query == "Cached trend"
        mock_firestore_client.get_document.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_vertex_failure(self, agent, mock_vertex_client):
        """Test error handling when Vertex AI fails"""
        mock_vertex_client.generate_text.side_effect = Exception("Vertex AI error")
        
        # Should handle error gracefully
        audience_data = await agent.analyze_audience("Test trend")
        
        assert audience_data is None
    
    @pytest.mark.asyncio
    async def test_batch_audience_analysis(self, agent, mock_vertex_client):
        """Test batch processing of multiple trends"""
        mock_vertex_client.generate_text.return_value = {
            'text': 'Standard audience profile',
            'confidence': 0.85
        }
        
        trends = ["AI art", "Sustainable fashion", "Tech gadgets"]
        
        results = await agent.analyze_audiences_batch(trends)
        
        assert len(results) == 3
        assert all(result is not None for result in results)
        assert all(result.confidence_score >= 7.0 for result in results)


if __name__ == '__main__':
    pytest.main([__file__])
