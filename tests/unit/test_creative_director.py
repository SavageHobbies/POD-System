"""
Unit tests for Creative Director Agent
Tests image generation, design strategy, and quality assurance
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
import asyncio
from PIL import Image
import io

from helios.agents.creative import CreativeDirectorAgent
from helios.models.product_data import ProductData
from helios.models.audience_data import AudienceProfile
from helios.services.google_cloud.vertex_ai_client import VertexAIClient
from helios.services.google_cloud.storage_client import StorageClient


class TestCreativeDirectorAgent:
    """Test suite for Creative Director Agent"""
    
    @pytest.fixture
    def mock_vertex_ai_client(self):
        """Mock Vertex AI client"""
        mock_client = Mock()
        mock_model = Mock()
        
        # Mock image generation response
        mock_response = Mock()
        mock_response.image = Mock()
        mock_response.image.data = b"fake_image_data"
        mock_model.generate_content.return_value = mock_response
        
        mock_client.get_model.return_value = mock_model
        mock_client.generate_image.return_value = mock_response
        
        return mock_client
    
    @pytest.fixture
    def mock_storage_client(self):
        """Mock Cloud Storage client"""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.upload_from_string.return_value = None
        
        return mock_client
    
    @pytest.fixture
    def mock_vision_client(self):
        """Mock Cloud Vision client"""
        mock_client = Mock()
        
        # Mock safe search response
        mock_safe_search = Mock()
        mock_safe_search.safe_search_annotation.adult = "VERY_UNLIKELY"
        mock_safe_search.safe_search_annotation.violence = "VERY_UNLIKELY"
        mock_safe_search.safe_search_annotation.racy = "VERY_UNLIKELY"
        
        mock_client.annotate_image.return_value = [mock_safe_search]
        
        return mock_client
    
    @pytest.fixture
    def creative_director(self, mock_vertex_ai_client, mock_storage_client, mock_vision_client):
        """Creative Director Agent instance with mocked dependencies"""
        with patch('helios.agents.creative.VertexAIClient', return_value=mock_vertex_ai_client), \
             patch('helios.agents.creative.StorageClient', return_value=mock_storage_client), \
             patch('helios.agents.creative.VisionClient', return_value=mock_vision_client):
            
            agent = CreativeDirectorAgent()
            return agent
    
    @pytest.fixture
    def sample_product(self):
        """Sample product data for testing"""
        return ProductData(
            id="prod_001",
            name="Sustainable Fashion T-Shirt",
            category="clothing",
            description="Eco-friendly fashion statement",
            base_price=25.00,
            target_margin=0.35,
            print_areas=["front", "back"],
            variants=["S", "M", "L", "XL"]
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
    
    def test_initialization(self, creative_director):
        """Test agent initialization"""
        assert creative_director is not None
        assert hasattr(creative_director, 'vertex_ai_client')
        assert hasattr(creative_director, 'storage_client')
        assert hasattr(creative_director, 'vision_client')
        assert creative_director.max_concurrent_generations == 5
    
    def test_generate_design_concept(self, creative_director, sample_product, sample_audience):
        """Test design concept generation"""
        concept = creative_director.generate_design_concept(sample_product, sample_audience)
        
        assert concept is not None
        assert 'theme' in concept
        assert 'style' in concept
        assert 'color_scheme' in concept
        assert 'visual_elements' in concept
        assert 'brand_alignment' in concept
        
        # Verify concept aligns with product and audience
        assert 'sustainability' in concept['theme'].lower()
        assert concept['brand_alignment'] >= 7.0
    
    def test_generate_image_prompt(self, creative_director, sample_product, sample_audience):
        """Test image prompt generation"""
        prompt = creative_director.generate_image_prompt(sample_product, sample_audience)
        
        assert prompt is not None
        assert isinstance(prompt, str)
        assert len(prompt) > 50  # Should be detailed
        
        # Verify prompt includes key elements
        assert 'sustainable' in prompt.lower()
        assert 'fashion' in prompt.lower()
        assert 'eco-friendly' in prompt.lower()
        assert '1024x1024' in prompt
    
    def test_generate_image_primary_model(self, creative_director, mock_vertex_ai_client):
        """Test image generation using primary Imagen-3 model"""
        prompt = "A sustainable fashion t-shirt design with eco-friendly elements"
        
        image_data = creative_director.generate_image(prompt, model="imagen-3")
        
        assert image_data is not None
        assert isinstance(image_data, bytes)
        
        # Verify Vertex AI was called with correct model
        mock_vertex_ai_client.get_model.assert_called_with("imagen-3")
    
    def test_generate_image_backup_model(self, creative_director, mock_vertex_ai_client):
        """Test image generation using backup Imagen-2 model"""
        prompt = "A sustainable fashion t-shirt design with eco-friendly elements"
        
        # Mock primary model failure
        mock_vertex_ai_client.get_model.side_effect = Exception("Primary model unavailable")
        
        image_data = creative_director.generate_image(prompt, model="imagen-2")
        
        assert image_data is not None
        assert isinstance(image_data, bytes)
        
        # Verify backup model was used
        mock_vertex_ai_client.get_model.assert_called_with("imagen-2")
    
    def test_batch_image_generation(self, creative_director):
        """Test concurrent batch image generation"""
        prompt = "A sustainable fashion t-shirt design"
        count = 3
        
        # Mock async generation
        with patch.object(creative_director, 'generate_image_async', return_value=b"fake_image"):
            images = creative_director.generate_batch_images(prompt, count)
        
        assert images is not None
        assert len(images) == count
        assert all(isinstance(img, bytes) for img in images)
    
    @pytest.mark.asyncio
    async def test_generate_image_async(self, creative_director):
        """Test asynchronous image generation"""
        prompt = "A sustainable fashion t-shirt design"
        
        # Mock async response
        with patch.object(creative_director.vertex_ai_client, 'generate_image_async', 
                         new_callable=AsyncMock, return_value=b"fake_image_data"):
            image_data = await creative_director.generate_image_async(prompt)
        
        assert image_data is not None
        assert isinstance(image_data, bytes)
    
    def test_validate_image_safety(self, creative_director, mock_vision_client):
        """Test image safety validation using Cloud Vision"""
        # Create a fake image
        fake_image = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        fake_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        safety_result = creative_director.validate_image_safety(img_byte_arr)
        
        assert safety_result is not None
        assert 'is_safe' in safety_result
        assert 'safety_scores' in safety_result
        assert 'blocked_reasons' in safety_result
        
        # Verify Vision API was called
        mock_vision_client.annotate_image.assert_called()
    
    def test_validate_image_quality(self, creative_director):
        """Test image quality validation"""
        # Create a high-quality image
        high_quality_image = Image.new('RGB', (1024, 1024), color='blue')
        img_byte_arr = io.BytesIO()
        high_quality_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        quality_result = creative_director.validate_image_quality(img_byte_arr)
        
        assert quality_result is not None
        assert 'is_high_quality' in quality_result
        assert 'resolution' in quality_result
        assert 'format' in quality_result
        assert 'size' in quality_result
        
        # Verify quality metrics
        assert quality_result['resolution'] == (1024, 1024)
        assert quality_result['format'] == 'PNG'
    
    def test_upload_to_storage(self, creative_director, mock_storage_client):
        """Test image upload to Cloud Storage"""
        image_data = b"fake_image_data"
        filename = "sustainable_tshirt_design_001.png"
        
        url = creative_director.upload_to_storage(image_data, filename)
        
        assert url is not None
        assert isinstance(url, str)
        assert filename in url
        
        # Verify Storage client was called
        mock_storage_client.bucket.assert_called()
    
    def test_generate_design_variations(self, creative_director, sample_product, sample_audience):
        """Test generation of multiple design variations"""
        variations = creative_director.generate_design_variations(sample_product, sample_audience, count=3)
        
        assert variations is not None
        assert len(variations) == 3
        
        for variation in variations:
            assert 'design_concept' in variation
            assert 'image_prompt' in variation
            assert 'target_audience' in variation
            assert 'brand_alignment' in variation
    
    def test_apply_brand_guidelines(self, creative_director, sample_product):
        """Test application of brand guidelines to designs"""
        design_concept = {
            'theme': 'sustainability',
            'style': 'modern',
            'color_scheme': ['green', 'blue'],
            'visual_elements': ['leaf', 'recycle']
        }
        
        branded_concept = creative_director.apply_brand_guidelines(design_concept, sample_product)
        
        assert branded_concept is not None
        assert 'brand_colors' in branded_concept
        assert 'brand_fonts' in branded_concept
        assert 'brand_elements' in branded_concept
        assert 'compliance_score' in branded_concept
    
    def test_optimize_for_print(self, creative_director, sample_product):
        """Test print optimization for designs"""
        design_specs = {
            'resolution': (1024, 1024),
            'format': 'PNG',
            'color_mode': 'RGB',
            'print_areas': ['front', 'back']
        }
        
        optimized_specs = creative_director.optimize_for_print(design_specs, sample_product)
        
        assert optimized_specs is not None
        assert 'print_resolution' in optimized_specs
        assert 'color_profile' in optimized_specs
        assert 'file_format' in optimized_specs
        assert 'print_areas' in optimized_specs
        
        # Verify print optimization
        assert optimized_specs['print_resolution'] >= 300  # DPI for print
        assert optimized_specs['file_format'] in ['PNG', 'TIFF', 'PDF']
    
    def test_create_design_metadata(self, creative_director, sample_product, sample_audience):
        """Test creation of design metadata"""
        design_concept = {
            'theme': 'sustainability',
            'style': 'modern',
            'color_scheme': ['green', 'blue']
        }
        
        metadata = creative_director.create_design_metadata(design_concept, sample_product, sample_audience)
        
        assert metadata is not None
        assert 'product_id' in metadata
        assert 'audience_segment' in metadata
        assert 'design_theme' in metadata
        assert 'created_at' in metadata
        assert 'version' in metadata
        assert 'tags' in metadata
    
    def test_error_handling_generation_failure(self, creative_director, mock_vertex_ai_client):
        """Test error handling when image generation fails"""
        # Mock generation failure
        mock_vertex_ai_client.generate_image.side_effect = Exception("Generation failed")
        
        with pytest.raises(Exception):
            creative_director.generate_image("test prompt")
    
    def test_error_handling_storage_failure(self, creative_director, mock_storage_client):
        """Test error handling when storage upload fails"""
        # Mock storage failure
        mock_storage_client.bucket.side_effect = Exception("Storage connection failed")
        
        with pytest.raises(Exception):
            creative_director.upload_to_storage(b"fake_data", "test.png")
    
    def test_fallback_generation_strategy(self, creative_director):
        """Test fallback generation strategy when primary method fails"""
        prompt = "A sustainable fashion t-shirt design"
        
        # Mock primary generation failure
        with patch.object(creative_director, 'generate_image', side_effect=Exception("Primary failed")):
            # Should fall back to alternative method
            fallback_image = creative_director.generate_image_fallback(prompt)
            
            assert fallback_image is not None
            # Could be a template or placeholder image
    
    def test_quality_gate_enforcement(self, creative_director):
        """Test quality gate enforcement for designs"""
        design_result = {
            'image_data': b"fake_image",
            'concept': {'theme': 'sustainability', 'brand_alignment': 8.5},
            'safety_score': 9.8,
            'quality_score': 9.2
        }
        
        quality_result = creative_director.enforce_quality_gates(design_result)
        
        assert quality_result is not None
        assert 'passed' in quality_result
        assert 'score' in quality_result
        assert 'issues' in quality_result
        
        # Verify quality thresholds
        if quality_result['passed']:
            assert quality_result['score'] >= 7.0
    
    def test_performance_optimization(self, creative_director):
        """Test performance optimization features"""
        # Test concurrent processing limits
        start_time = datetime.now()
        
        with creative_director.concurrent_processing():
            # Simulate concurrent operations
            pass
        
        execution_time = (datetime.now() - start_time).total_seconds()
        assert execution_time < 5.0  # Should complete within 5 seconds
    
    def test_cache_management(self, creative_director):
        """Test design cache management"""
        # Test cache hit
        cache_key = "sustainable_tshirt_design"
        cached_design = creative_director.get_cached_design(cache_key)
        
        # Initially should be None
        assert cached_design is None
        
        # Test cache storage
        design_data = {'concept': 'sustainability', 'image': b'fake_image'}
        creative_director.cache_design(cache_key, design_data)
        
        # Now should retrieve cached design
        retrieved_design = creative_director.get_cached_design(cache_key)
        assert retrieved_design is not None
        assert retrieved_design['concept'] == 'sustainability'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
