"""
Google Cloud Vertex AI Client for Helios Autonomous Store
Provides integration with Gemini AI models for text and image generation
"""

import os
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import time

import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import aiplatform
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError

from loguru import logger


@dataclass
class GeminiModelConfig:
    """Configuration for Gemini AI models"""
    model_name: str
    max_tokens: int
    temperature: float
    top_p: float = 0.8
    top_k: int = 40
    use_cases: List[str] = None


class VertexAIClient:
    """
    Google Cloud Vertex AI client for Helios Autonomous Store
    Handles text generation, analysis, and image generation using Gemini models
    """
    
    def __init__(self, project_id: str = None, location: str = "us-central1"):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.initialized = False
        
        # Model configurations based on requirements
        self.models = {
            "gemini_pro": GeminiModelConfig(
                model_name="gemini-1.5-pro",
                max_tokens=8192,
                temperature=0.7,
                use_cases=["Complex analysis", "Strategy formulation"]
            ),
            "gemini_flash": GeminiModelConfig(
                model_name="gemini-1.5-flash",
                max_tokens=8192,
                temperature=0.8,
                use_cases=["Rapid processing", "Copy generation"]
            ),
            "gemini_ultra": GeminiModelConfig(
                model_name="gemini-1.0-ultra",
                max_tokens=32768,
                temperature=0.5,
                use_cases=["CEO orchestration", "Critical decisions"]
            )
        }
        
        # Image generation model
        self.image_model = "imagen-3"
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Vertex AI client"""
        try:
            # Set environment variables
            os.environ["GOOGLE_CLOUD_PROJECT"] = self.project_id
            os.environ["GOOGLE_CLOUD_LOCATION"] = self.location
            
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            
            # Initialize AI Platform
            aiplatform.init(project=self.project_id, location=self.location)
            
            self.initialized = True
            logger.info(f"✅ Vertex AI client initialized for project: {self.project_id}")
            
        except DefaultCredentialsError:
            logger.error("❌ Google Cloud credentials not found. Please set up authentication.")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vertex AI client: {e}")
            raise
    
    def get_model(self, model_type: str = "gemini_pro") -> GenerativeModel:
        """Get a Gemini model instance"""
        if not self.initialized:
            raise RuntimeError("Vertex AI client not initialized")
        
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = self.models[model_type]
        return GenerativeModel(
            model_name=config.model_name,
            generation_config={
                "max_output_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k
            }
        )
    
    async def generate_text(
        self,
        prompt: str,
        model_type: str = "gemini_pro",
        system_prompt: str = None,
        context: List[Dict] = None
    ) -> str:
        """
        Generate text using Gemini AI
        
        Args:
            prompt: The main prompt for text generation
            model_type: Type of Gemini model to use
            system_prompt: Optional system prompt for context
            context: Optional conversation context
            
        Returns:
            Dictionary containing generated text and metadata
        """
        try:
            model = self.get_model(model_type)
            # Concatenate system/context into a single prompt for simplicity
            full_prompt_parts: List[str] = []
            if system_prompt:
                full_prompt_parts.append(system_prompt)
            if context:
                try:
                    for entry in context:
                        if isinstance(entry, dict):
                            part = entry.get("parts") or entry.get("content") or ""
                            if isinstance(part, list):
                                full_prompt_parts.extend([str(p) for p in part])
                            else:
                                full_prompt_parts.append(str(part))
                        else:
                            full_prompt_parts.append(str(entry))
                except Exception:
                    pass
            full_prompt_parts.append(prompt)
            full_prompt = "\n\n".join([p for p in full_prompt_parts if p])

            response = await asyncio.to_thread(model.generate_content, full_prompt)
            text = getattr(response, "text", "") or ""
            logger.info(f"✅ Text generated using {model_type}: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"❌ Text generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_used": model_type,
                "timestamp": time.time()
            }
    
    async def analyze_trend(
        self,
        trend_data: Dict[str, Any],
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Analyze trend data using Gemini AI
        
        Args:
            trend_data: Dictionary containing trend information
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis results with insights and recommendations
        """
        # Use Gemini Pro for complex trend analysis
        model_type = "gemini_pro"
        
        prompt = f"""
        Analyze the following trend data and provide comprehensive insights:
        
        Trend: {trend_data.get('trend_name', 'Unknown')}
        Keywords: {trend_data.get('keywords', [])}
        Volume: {trend_data.get('volume', 'Unknown')}
        Growth: {trend_data.get('growth_rate', 'Unknown')}
        
        Please provide:
        1. Opportunity assessment (1-10 scale)
        2. Target audience analysis
        3. Product category recommendations
        4. Marketing angle suggestions
        5. Risk factors
        6. Timeline recommendations
        
        Analysis type: {analysis_type}
        """
        
        return await self.generate_text(prompt, model_type)
    
    async def generate_marketing_copy(
        self,
        product_info: Dict[str, Any],
        target_audience: Dict[str, Any],
        tone: str = "persuasive"
    ) -> Dict[str, Any]:
        """
        Generate marketing copy using Gemini Flash for speed
        
        Args:
            product_info: Product details
            target_audience: Target audience information
            tone: Desired tone for the copy
            
        Returns:
            Generated marketing copy and variations
        """
        model_type = "gemini_flash"
        
        prompt = f"""
        Generate compelling marketing copy for this product:
        
        Product: {product_info.get('name', 'Unknown')}
        Category: {product_info.get('category', 'Unknown')}
        Target Audience: {target_audience.get('demographic_cluster', 'Unknown')}
        Tone: {tone}
        
        Generate:
        1. Product title (SEO optimized)
        2. Short description (2-3 sentences)
        3. Long description (5-7 sentences)
        4. 5-7 bullet points highlighting benefits
        5. Call-to-action variations
        
        Make it engaging, persuasive, and optimized for e-commerce conversion.
        """
        
        return await self.generate_text(prompt, model_type)
    
    async def generate_product_strategy(
        self,
        trend_analysis: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate product strategy using Gemini Ultra for complex decision making
        
        Args:
            trend_analysis: Results from trend analysis
            market_data: Market and competitor information
            
        Returns:
            Comprehensive product strategy
        """
        model_type = "gemini_ultra"
        
        prompt = f"""
        Based on the following data, generate a comprehensive product strategy:
        
        Trend Analysis:
        {json.dumps(trend_analysis, indent=2)}
        
        Market Data:
        {json.dumps(market_data, indent=2)}
        
        Provide:
        1. Product positioning strategy
        2. Pricing strategy with margin recommendations
        3. Launch timeline
        4. Marketing channel prioritization
        5. Risk mitigation strategies
        6. Success metrics and KPIs
        7. Competitive advantage analysis
        
        This is for a print-on-demand e-commerce business. Be strategic and data-driven.
        """
        
        return await self.generate_text(prompt, model_type)
    
    async def generate_image(
        self,
        prompt: str,
        output_path: Path = None,
        resolution: str = "1024x1024",
        quality: str = "high"
    ) -> Dict[str, Any]:
        """
        Generate image using Vertex AI Imagen
        
        Args:
            prompt: Text description for image generation
            output_path: Path to save the generated image
            resolution: Image resolution
            quality: Image quality setting
            
        Returns:
            Image generation results
        """
        try:
            # Note: Image generation with Imagen requires specific setup
            # This is a placeholder for the actual implementation
            logger.info(f"🖼️ Image generation requested: {prompt}")
            
            # For now, return a placeholder
            result = {
                "status": "not_implemented",
                "message": "Image generation with Imagen requires additional setup",
                "prompt": prompt,
                "resolution": resolution,
                "quality": quality,
                "timestamp": time.time()
            }
            
            logger.warning("⚠️ Image generation not yet implemented - requires Imagen API access")
            return result
            
        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def batch_generate_text(
        self,
        prompts: List[str],
        model_type: str = "gemini_flash"
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts in batch
        
        Args:
            prompts: List of prompts to process
            model_type: Type of Gemini model to use
            
        Returns:
            List of generation results
        """
        tasks = []
        for prompt in prompts:
            task = self.generate_text(prompt, model_type)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "status": "error",
                    "error": str(result),
                    "prompt_index": i,
                    "timestamp": time.time()
                })
            else:
                processed_results.append({
                    "status": "success",
                    "text": result,
                    "prompt_index": i
                })
        
        logger.info(f"✅ Batch text generation completed: {len(processed_results)} prompts")
        return processed_results
    
    def get_model_info(self, model_type: str = None) -> Dict[str, Any]:
        """Get information about available models"""
        if model_type:
            if model_type not in self.models:
                return {"error": f"Unknown model type: {model_type}"}
            return {
                "model_type": model_type,
                **self.models[model_type].__dict__
            }
        
        return {
            "available_models": list(self.models.keys()),
            "models": self.models,
            "image_model": self.image_model,
            "project_id": self.project_id,
            "location": self.location
        }


# Convenience functions for common operations
async def analyze_trend_with_ai(trend_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for trend analysis"""
    client = VertexAIClient()
    return await client.analyze_trend(trend_data)


async def generate_marketing_copy_ai(product_info: Dict[str, Any], audience: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for marketing copy generation"""
    client = VertexAIClient()
    return await client.generate_marketing_copy(product_info, audience)


async def generate_product_strategy_ai(trend_analysis: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for product strategy generation"""
    client = VertexAIClient()
    return await client.generate_product_strategy(trend_analysis, market_data)
