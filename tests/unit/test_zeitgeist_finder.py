"""
Unit tests for ZeitgeistFinderAgent
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime, timedelta

from helios.agents.zeitgeist import ZeitgeistFinderAgent
from helios.models.trend_data import TrendData, TrendSource
from helios.services.mcp_integration.mcp_client import MCPClient


class TestZeitgeistFinderAgent:
    """Test cases for ZeitgeistFinderAgent"""
    
    @pytest.fixture
    def mock_mcp_client(self):
        """Mock MCP client"""
        client = Mock(spec=MCPClient)
        client.get_google_trends = AsyncMock()
        client.scan_social_media = AsyncMock()
        client.analyze_news = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_vertex_client(self):
        """Mock Vertex AI client"""
        client = Mock()
        client.generate_text = AsyncMock()
        client.analyze_sentiment = AsyncMock()
        return client
    
    @pytest.fixture
    def agent(self, mock_mcp_client, mock_vertex_client):
        """Create agent instance with mocked dependencies"""
        return ZeitgeistFinderAgent(
            mcp_client=mock_mcp_client,
            vertex_client=mock_vertex_client,
            config={
                'min_trend_score': 7.0,
                'max_trends_per_batch': 10,
                'trend_freshness_hours': 24
            }
        )
    
    @pytest.mark.asyncio
    async def test_discover_trends_success(self, agent, mock_mcp_client):
        """Test successful trend discovery"""
        # Mock MCP responses
        mock_mcp_client.get_google_trends.return_value = [
            {
                'query': 'AI art generator',
                'score': 8.5,
                'rising': True,
                'region': 'US',
                'time_range': '1d'
            }
        ]
        
        mock_mcp_client.scan_social_media.return_value = [
            {
                'platform': 'reddit',
                'topic': 'AI art generator',
                'engagement': 1500,
                'sentiment': 'positive',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        # Execute trend discovery
        trends = await agent.discover_trends()
        
        # Verify results
        assert len(trends) > 0
        assert trends[0].query == 'AI art generator'
        assert trends[0].score >= 7.0
        assert trends[0].source == TrendSource.GOOGLE_TRENDS
        
        # Verify MCP calls
        mock_mcp_client.get_google_trends.assert_called_once()
        mock_mcp_client.scan_social_media.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discover_trends_no_results(self, agent, mock_mcp_client):
        """Test trend discovery with no results"""
        # Mock empty responses
        mock_mcp_client.get_google_trends.return_value = []
        mock_mcp_client.scan_social_media.return_value = []
        
        trends = await agent.discover_trends()
        
        assert len(trends) == 0
    
    @pytest.mark.asyncio
    async def test_discover_trends_filtering(self, agent, mock_mcp_client):
        """Test trend filtering by score"""
        # Mock mixed quality trends
        mock_mcp_client.get_google_trends.return_value = [
            {'query': 'High score trend', 'score': 8.5, 'rising': True, 'region': 'US', 'time_range': '1d'},
            {'query': 'Low score trend', 'score': 4.2, 'rising': True, 'region': 'US', 'time_range': '1d'},
            {'query': 'Medium score trend', 'score': 7.8, 'rising': True, 'region': 'US', 'time_range': '1d'}
        ]
        
        mock_mcp_client.scan_social_media.return_value = []
        
        trends = await agent.discover_trends()
        
        # Should only include trends with score >= 7.0
        assert len(trends) == 2
        assert all(trend.score >= 7.0 for trend in trends)
        assert 'Low score trend' not in [t.query for t in trends]
    
    @pytest.mark.asyncio
    async def test_analyze_trend_psychology(self, agent, mock_vertex_client):
        """Test psychological analysis of trends"""
        mock_vertex_client.generate_text.return_value = {
            'text': 'This trend appeals to creativity and self-expression',
            'confidence': 0.85
        }
        
        trend_data = TrendData(
            query='AI art generator',
            score=8.5,
            source=TrendSource.GOOGLE_TRENDS,
            timestamp=datetime.now()
        )
        
        analysis = await agent.analyze_trend_psychology(trend_data)
        
        assert analysis is not None
        assert 'creativity' in analysis.lower()
        assert 'self-expression' in analysis.lower()
        
        mock_vertex_client.generate_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_prioritize_trends(self, agent):
        """Test trend prioritization logic"""
        trends = [
            TrendData(query='Trend A', score=8.5, source=TrendSource.GOOGLE_TRENDS, timestamp=datetime.now()),
            TrendData(query='Trend B', score=9.2, source=TrendSource.SOCIAL_MEDIA, timestamp=datetime.now()),
            TrendData(query='Trend C', score=7.8, source=TrendSource.NEWS, timestamp=datetime.now())
        ]
        
        prioritized = agent.prioritize_trends(trends)
        
        # Should be sorted by score descending
        assert prioritized[0].score == 9.2
        assert prioritized[1].score == 8.5
        assert prioritized[2].score == 7.8
    
    @pytest.mark.asyncio
    async def test_validate_trend_freshness(self, agent):
        """Test trend freshness validation"""
        # Recent trend
        recent_trend = TrendData(
            query='Recent trend',
            score=8.0,
            source=TrendSource.GOOGLE_TRENDS,
            timestamp=datetime.now()
        )
        
        # Old trend
        old_trend = TrendData(
            query='Old trend',
            score=8.0,
            source=TrendSource.GOOGLE_TRENDS,
            timestamp=datetime.now() - timedelta(hours=48)
        )
        
        assert agent.validate_trend_freshness(recent_trend) is True
        assert agent.validate_trend_freshness(old_trend) is False
    
    @pytest.mark.asyncio
    async def test_error_handling_mcp_failure(self, agent, mock_mcp_client):
        """Test error handling when MCP client fails"""
        mock_mcp_client.get_google_trends.side_effect = Exception("MCP connection failed")
        
        # Should handle error gracefully
        trends = await agent.discover_trends()
        
        assert len(trends) == 0
        # Should log error (we can't easily test logging in unit tests)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, agent, mock_mcp_client):
        """Test batch processing of multiple trend sources"""
        # Mock multiple sources
        mock_mcp_client.get_google_trends.return_value = [
            {'query': 'Google trend', 'score': 8.0, 'rising': True, 'region': 'US', 'time_range': '1d'}
        ]
        
        mock_mcp_client.scan_social_media.return_value = [
            {'platform': 'reddit', 'topic': 'Social trend', 'engagement': 1000, 'sentiment': 'positive', 'timestamp': datetime.now().isoformat()}
        ]
        
        mock_mcp_client.analyze_news.return_value = [
            {'headline': 'News trend', 'relevance': 8.5, 'sentiment': 'neutral', 'timestamp': datetime.now().isoformat()}
        ]
        
        trends = await agent.discover_trends()
        
        # Should combine all sources
        assert len(trends) >= 3
        sources = [t.source for t in trends]
        assert TrendSource.GOOGLE_TRENDS in sources
        assert TrendSource.SOCIAL_MEDIA in sources
        assert TrendSource.NEWS in sources


if __name__ == '__main__':
    pytest.main([__file__])
