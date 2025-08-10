from __future__ import annotations

from typing import Any, Dict, List, Optional
from ..config import load_config
from ..mcp_client import MCPClient
from ..services.google_cloud.vertex_ai_client import VertexAIClient
from ..services.google_cloud.sheets_client import GoogleSheetsClient
from ..services.google_cloud.drive_client import GoogleDriveClient
import asyncio
from datetime import datetime
import json
import random
import math
from dataclasses import dataclass, field
from enum import Enum
import statistics
from scipy import stats
import numpy as np


class MarketingCopywriter:
    """Marketing copy generation using Google MCP and Gemini 2.0 Flash for fast creative marketing."""

    def __init__(self) -> None:
        self.config = load_config()
        self.mcp_client = MCPClient.from_env(self.config.google_mcp_url, self.config.google_mcp_auth_token)
        
        # Initialize Google Cloud services
        self.vertex_ai_client = None
        self.sheets_client = None
        self.drive_client = None
        
        if self.config.google_cloud_project:
            try:
                self.vertex_ai_client = VertexAIClient(
                    project_id=self.config.google_cloud_project,
                    location=self.config.google_cloud_location
                )
                self.sheets_client = GoogleSheetsClient()
                self.drive_client = GoogleDriveClient()
            except Exception as e:
                print(f"Warning: Could not initialize Google Cloud services: {e}")
        
        # Initialize A/B testing and adaptive learning systems
        self.ab_testing = ABTestingFramework(self.config)
        self.adaptive_learning = AdaptiveLearningSystem()

    async def run(self, creative_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Generate marketing copy using Google MCP marketing_ai"""
        
        try:
            if self.mcp_client:
                # Use Google MCP marketing_ai (Gemini 2.0 Flash for creative marketing)
                product_info = self._extract_product_info(creative_batch)
                
                mcp_response = await self.mcp_client.marketing_ai(product_info)
                
                if "response" in mcp_response and mcp_response["response"]:
                    ai_copy = mcp_response["response"]
                    model_used = mcp_response.get("model", "gemini-2.0-flash-exp")
                    
                    # Parse AI response for marketing copy
                    marketing_copy = self._parse_marketing_ai_response(ai_copy, creative_batch)
                    
                    return {
                        "status": "success",
                        "marketing_copy": marketing_copy,
                        "mcp_model_used": model_used,
                        "execution_time_ms": mcp_response.get("execution_ms", 0),
                        "ai_raw_response": ai_copy
                    }
            
            # Fallback to basic marketing copy if MCP not available
            return self._generate_basic_marketing_copy(creative_batch)
            
        except Exception as e:
            print(f"Error in marketing copy generation: {e}")
            return self._generate_basic_marketing_copy(creative_batch)

    async def generate_seo_optimized_copy(self, product_data: Dict[str, Any], target_keywords: List[str]) -> Dict[str, Any]:
        """Generate SEO-optimized marketing copy with keyword integration"""
        try:
            # Create SEO-focused prompt
            seo_prompt = self._create_seo_prompt(product_data, target_keywords)
            
            if self.vertex_ai_client:
                # Use Vertex AI for SEO optimization
                response = await self.vertex_ai_client.generate_text(
                    model=self.config.gemini_flash_model,
                    prompt=seo_prompt,
                    max_tokens=1000,
                    temperature=0.7
                )
                
                seo_copy = self._parse_seo_response(response, product_data, target_keywords)
                return {
                    "status": "success",
                    "seo_copy": seo_copy,
                    "model_used": self.config.gemini_flash_model,
                    "keywords_integrated": target_keywords
                }
            else:
                # Fallback to basic SEO copy
                return self._generate_basic_seo_copy(product_data, target_keywords)
                
        except Exception as e:
            print(f"Error in SEO copy generation: {e}")
            return self._generate_basic_seo_copy(product_data, target_keywords)

    async def generate_social_media_campaign(self, product_data: Dict[str, Any], platform: str = "instagram") -> Dict[str, Any]:
        """Generate platform-specific social media campaign content"""
        try:
            platform_prompts = {
                "instagram": "Create engaging Instagram post content with hashtags",
                "facebook": "Create Facebook post content for business page",
                "twitter": "Create Twitter post content within character limit",
                "tiktok": "Create TikTok video description and hashtags",
                "pinterest": "Create Pinterest pin description and board suggestions"
            }
            
            prompt = f"{platform_prompts.get(platform, 'Create social media content')} for: {product_data.get('name', 'Product')}"
            
            if self.vertex_ai_client:
                response = await self.vertex_ai_client.generate_text(
                    model=self.config.gemini_flash_model,
                    prompt=prompt,
                    max_tokens=500,
                    temperature=0.8
                )
                
                campaign_content = self._parse_social_media_response(response, platform, product_data)
                return {
                    "status": "success",
                    "platform": platform,
                    "campaign_content": campaign_content,
                    "model_used": self.config.gemini_flash_model
                }
            else:
                return self._generate_basic_social_campaign(product_data, platform)
                
        except Exception as e:
            print(f"Error in social media campaign generation: {e}")
            return self._generate_basic_social_campaign(product_data, platform)

    async def generate_email_campaign(self, product_data: Dict[str, Any], campaign_type: str = "product_launch") -> Dict[str, Any]:
        """Generate email campaign content for different campaign types"""
        try:
            campaign_prompts = {
                "product_launch": "Create an exciting product launch email campaign",
                "seasonal_promotion": "Create a seasonal promotion email campaign",
                "abandoned_cart": "Create an abandoned cart recovery email",
                "customer_retention": "Create a customer retention email campaign",
                "newsletter": "Create a newsletter-style product showcase"
            }
            
            prompt = f"{campaign_prompts.get(campaign_type, 'Create an email campaign')} for: {product_data.get('name', 'Product')}"
            
            if self.vertex_ai_client:
                response = await self.vertex_ai_client.generate_text(
                    model=self.config.gemini_flash_model,
                    prompt=prompt,
                    max_tokens=800,
                    temperature=0.7
                )
                
                email_content = self._parse_email_response(response, campaign_type, product_data)
                return {
                    "status": "success",
                    "campaign_type": campaign_type,
                    "email_content": email_content,
                    "model_used": self.config.gemini_flash_model
                }
            else:
                return self._generate_basic_email_campaign(product_data, campaign_type)
                
        except Exception as e:
            print(f"Error in email campaign generation: {e}")
            return self._generate_basic_email_campaign(product_data, campaign_type)

    async def generate_ad_copy(self, product_data: Dict[str, Any], ad_platform: str = "google_ads") -> Dict[str, Any]:
        """Generate platform-specific advertising copy"""
        try:
            platform_requirements = {
                "google_ads": "Create Google Ads copy with headlines and descriptions",
                "facebook_ads": "Create Facebook Ads copy with primary text and headlines",
                "instagram_ads": "Create Instagram Ads copy for feed and stories",
                "tiktok_ads": "Create TikTok Ads copy for in-feed ads"
            }
            
            prompt = f"{platform_requirements.get(ad_platform, 'Create advertising copy')} for: {product_data.get('name', 'Product')}"
            
            if self.vertex_ai_client:
                response = await self.vertex_ai_client.generate_text(
                    model=self.config.gemini_flash_model,
                    prompt=prompt,
                    max_tokens=600,
                    temperature=0.8
                )
                
                ad_copy = self._parse_ad_response(response, ad_platform, product_data)
                return {
                    "status": "success",
                    "ad_platform": ad_platform,
                    "ad_copy": ad_copy,
                    "model_used": self.config.gemini_flash_model
                }
            else:
                return self._generate_basic_ad_copy(product_data, ad_platform)
                
        except Exception as e:
            print(f"Error in ad copy generation: {e}")
            return self._generate_basic_ad_copy(product_data, ad_platform)

    async def analyze_competitor_copy(self, competitor_urls: List[str], product_category: str) -> Dict[str, Any]:
        """Analyze competitor marketing copy for insights"""
        try:
            analysis_prompt = f"""
            Analyze marketing copy from competitors in the {product_category} category.
            Focus on:
            1. Common messaging themes
            2. Unique selling propositions
            3. Tone and voice
            4. Call-to-action strategies
            5. Keyword usage patterns
            
            Competitor URLs: {', '.join(competitor_urls)}
            """
            
            if self.vertex_ai_client:
                response = await self.vertex_ai_client.generate_text(
                    model=self.config.gemini_pro_model,
                    prompt=analysis_prompt,
                    max_tokens=1200,
                    temperature=0.6
                )
                
                competitor_analysis = self._parse_competitor_analysis(response, competitor_urls)
                return {
                    "status": "success",
                    "competitor_analysis": competitor_analysis,
                    "model_used": self.config.gemini_pro_model,
                    "competitors_analyzed": len(competitor_urls)
                }
            else:
                return self._generate_basic_competitor_analysis(competitor_urls, product_category)
                
        except Exception as e:
            print(f"Error in competitor copy analysis: {e}")
            return self._generate_basic_competitor_analysis(competitor_urls, product_category)

    async def generate_brand_voice_guide(self, brand_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Generate brand voice and messaging guidelines"""
        try:
            brand_prompt = f"""
            Create comprehensive brand voice and messaging guidelines based on:
            Brand personality: {brand_attributes.get('personality', 'Professional')}
            Target audience: {brand_attributes.get('target_audience', 'General')}
            Industry: {brand_attributes.get('industry', 'E-commerce')}
            Values: {brand_attributes.get('values', 'Quality, Innovation')}
            
            Include:
            1. Tone of voice
            2. Messaging pillars
            3. Do's and don'ts
            4. Example copy
            5. Brand vocabulary
            """
            
            if self.vertex_ai_client:
                response = await self.vertex_ai_client.generate_text(
                    model=self.config.gemini_pro_model,
                    prompt=brand_prompt,
                    max_tokens=1500,
                    temperature=0.7
                )
                
                brand_guide = self._parse_brand_voice_guide(response, brand_attributes)
                return {
                    "status": "success",
                    "brand_voice_guide": brand_guide,
                    "model_used": self.config.gemini_pro_model
                }
            else:
                return self._generate_basic_brand_guide(brand_attributes)
                
        except Exception as e:
            print(f"Error in brand voice guide generation: {e}")
            return self._generate_basic_brand_guide(brand_attributes)

    async def track_marketing_performance(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track and analyze marketing campaign performance"""
        try:
            # Store campaign data in Google Sheets if available
            if self.sheets_client and self.config.gsheet_id:
                await self._store_campaign_data(campaign_data)
            
            # Generate performance insights
            performance_insights = self._analyze_campaign_performance(campaign_data)
            
            return {
                "status": "success",
                "performance_insights": performance_insights,
                "campaign_tracked": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"Error in marketing performance tracking: {e}")
            return {
                "status": "error",
                "error": str(e),
                "campaign_tracked": False
            }

    async def prepare_batch(self, trend_data: Dict[str, Any], products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare batch marketing generation for multiple products"""
        try:
            batch_config = {
                "trend_name": trend_data.get("trend_name", "Unknown Trend"),
                "keywords": trend_data.get("keywords", []),
                "products": products,
                "batch_size": len(products),
                "marketing_types": ["seo", "social", "email", "ads"],
                "platforms": ["instagram", "facebook", "twitter", "tiktok", "pinterest"],
                "ad_platforms": ["google_ads", "facebook_ads", "instagram_ads", "tiktok_ads"]
            }
            
            return {
                "status": "success",
                "batch_config": batch_config,
                "prepared_for": len(products),
                "marketing_types": batch_config["marketing_types"],
                "platforms": batch_config["platforms"]
            }
            
        except Exception as e:
            print(f"Error preparing marketing batch: {e}")
            return {
                "status": "error",
                "error": str(e),
                "batch_config": {}
            }

    async def run_batch(self, designs: List[Dict[str, Any]], batch_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate marketing copy for a batch of designs"""
        try:
            if not designs:
                return {
                    "status": "error",
                    "error": "No designs provided for batch processing"
                }
            
            batch_results = []
            trend_name = batch_config.get("trend_name", "Unknown Trend")
            keywords = batch_config.get("keywords", [])
            
            for design in designs:
                # Create creative batch for this design
                creative_batch = {
                    "trend_name": trend_name,
                    "keywords": keywords,
                    "products": [design.get("product", {})],
                    "designs": [design]
                }
                
                # Generate marketing copy for this design
                marketing_result = await self.run(creative_batch)
                batch_results.append({
                    "design_id": design.get("id", "unknown"),
                    "product_name": design.get("product", {}).get("name", "Unknown Product"),
                    "marketing_copy": marketing_result.get("marketing_copy", {}),
                    "status": marketing_result.get("status", "unknown")
                })
            
            # Generate comprehensive campaign materials
            campaign_materials = await self._generate_campaign_materials(batch_config, designs)
            
            return {
                "status": "success",
                "batch_results": batch_results,
                "campaign_materials": campaign_materials,
                "total_designs_processed": len(designs),
                "successful_generations": len([r for r in batch_results if r["status"] == "success"]),
                "batch_config": batch_config
            }
            
        except Exception as e:
            print(f"Error in batch marketing generation: {e}")
            return {
                "status": "error",
                "error": str(e),
                "batch_results": [],
                "campaign_materials": {}
            }

    async def _generate_campaign_materials(self, batch_config: Dict[str, Any], designs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive campaign materials for the batch"""
        try:
            trend_name = batch_config.get("trend_name", "Unknown Trend")
            keywords = batch_config.get("keywords", [])
            
            campaign_materials = {
                "trend_campaign": {},
                "social_media_campaigns": {},
                "email_campaigns": {},
                "ad_campaigns": {},
                "brand_voice_guide": {}
            }
            
            # Generate trend-based campaign
            if self.vertex_ai_client:
                trend_prompt = f"""
                Create a comprehensive marketing campaign for the trend: {trend_name}
                
                Keywords: {', '.join(keywords)}
                Number of designs: {len(designs)}
                
                Include:
                1. Campaign theme and messaging
                2. Target audience personas
                3. Key selling points
                4. Campaign timeline
                5. Success metrics
                """
                
                response = await self.vertex_ai_client.generate_text(
                    model=self.config.gemini_pro_model,
                    prompt=trend_prompt,
                    max_tokens=1200,
                    temperature=0.7
                )
                
                campaign_materials["trend_campaign"] = {
                    "campaign_theme": response[:200] + "..." if len(response) > 200 else response,
                    "full_campaign": response,
                    "model_used": self.config.gemini_pro_model
                }
            
            # Generate platform-specific campaigns
            platforms = batch_config.get("platforms", [])
            for platform in platforms:
                campaign_materials["social_media_campaigns"][platform] = await self.generate_social_media_campaign(
                    {"name": trend_name, "keywords": keywords}, platform
                )
            
            # Generate email campaigns
            campaign_types = ["product_launch", "seasonal_promotion", "newsletter"]
            for campaign_type in campaign_types:
                campaign_materials["email_campaigns"][campaign_type] = await self.generate_email_campaign(
                    {"name": trend_name, "keywords": keywords}, campaign_type
                )
            
            # Generate ad campaigns
            ad_platforms = batch_config.get("ad_platforms", [])
            for ad_platform in ad_platforms:
                campaign_materials["ad_campaigns"][ad_platform] = await self.generate_ad_copy(
                    {"name": trend_name, "keywords": keywords}, ad_platform
                )
            
            # Generate brand voice guide
            brand_attributes = {
                "personality": "Trend-conscious and innovative",
                "target_audience": "Fashion-forward consumers",
                "industry": "Print-on-demand fashion",
                "values": "Quality, Innovation, Trend-awareness"
            }
            campaign_materials["brand_voice_guide"] = await self.generate_brand_voice_guide(brand_attributes)
            
            return campaign_materials
            
        except Exception as e:
            print(f"Error generating campaign materials: {e}")
            return {
                "trend_campaign": {},
                "social_media_campaigns": {},
                "email_campaigns": {},
                "ad_campaigns": {},
                "brand_voice_guide": {},
                "error": str(e)
            }

    # Helper methods for parsing responses
    def _create_seo_prompt(self, product_data: Dict[str, Any], target_keywords: List[str]) -> str:
        """Create SEO-focused prompt for copy generation"""
        return f"""
        Create SEO-optimized marketing copy for: {product_data.get('name', 'Product')}
        
        Target keywords: {', '.join(target_keywords)}
        Product description: {product_data.get('description', '')}
        Target audience: {product_data.get('target_audience', 'General consumers')}
        
        Include:
        1. SEO-optimized title (50-60 characters)
        2. Meta description (150-160 characters)
        3. Product description with keyword integration
        4. Alt text suggestions for images
        5. Internal linking suggestions
        
        Ensure natural keyword usage and engaging copy.
        """

    def _parse_seo_response(self, response: str, product_data: Dict[str, Any], target_keywords: List[str]) -> Dict[str, Any]:
        """Parse SEO response into structured format"""
        # Basic parsing - in production you'd want more sophisticated NLP
        lines = response.split('\n')
        
        seo_copy = {
            "seo_title": "",
            "meta_description": "",
            "product_description": "",
            "alt_text_suggestions": [],
            "internal_links": [],
            "keyword_density": {}
        }
        
        current_section = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "title:" in line.lower():
                seo_copy["seo_title"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif "meta" in line.lower() and "description" in line.lower():
                seo_copy["meta_description"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif "description:" in line.lower():
                seo_copy["product_description"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif "alt text" in line.lower():
                alt_text = line.split(":", 1)[1].strip() if ":" in line else line
                if alt_text:
                    seo_copy["alt_text_suggestions"].append(alt_text)
        
        # Calculate keyword density
        for keyword in target_keywords:
            count = response.lower().count(keyword.lower())
            seo_copy["keyword_density"][keyword] = count
        
        return seo_copy

    def _parse_social_media_response(self, response: str, platform: str, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse social media response into structured format"""
        lines = response.split('\n')
        
        campaign_content = {
            "post_caption": "",
            "hashtags": [],
            "mentions": [],
            "call_to_action": "",
            "platform_specific": {}
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "caption:" in line.lower():
                campaign_content["post_caption"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif "hashtags:" in line.lower():
                hashtags_text = line.split(":", 1)[1].strip() if ":" in line else line
                campaign_content["hashtags"] = [tag.strip() for tag in hashtags_text.split(",")]
            elif "cta:" in line.lower() or "call to action:" in line.lower():
                campaign_content["call_to_action"] = line.split(":", 1)[1].strip() if ":" in line else line
        
        return campaign_content

    def _parse_email_response(self, response: str, campaign_type: str, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse email response into structured format"""
        lines = response.split('\n')
        
        email_content = {
            "subject_line": "",
            "preheader": "",
            "email_body": "",
            "call_to_action": "",
            "personalization_tokens": []
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "subject:" in line.lower():
                email_content["subject_line"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif "preheader:" in line.lower():
                email_content["preheader"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif "body:" in line.lower():
                email_content["email_body"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif "cta:" in line.lower():
                email_content["call_to_action"] = line.split(":", 1)[1].strip() if ":" in line else line
        
        return email_content

    def _parse_ad_response(self, response: str, ad_platform: str, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse ad response into structured format"""
        lines = response.split('\n')
        
        ad_copy = {
            "headlines": [],
            "descriptions": [],
            "call_to_action": "",
            "display_url": "",
            "platform_specific": {}
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "headline:" in line.lower():
                headline = line.split(":", 1)[1].strip() if ":" in line else line
                if headline:
                    ad_copy["headlines"].append(headline)
            elif "description:" in line.lower():
                description = line.split(":", 1)[1].strip() if ":" in line else line
                if description:
                    ad_copy["descriptions"].append(description)
            elif "cta:" in line.lower():
                ad_copy["call_to_action"] = line.split(":", 1)[1].strip() if ":" in line else line
        
        return ad_copy

    def _parse_competitor_analysis(self, response: str, competitor_urls: List[str]) -> Dict[str, Any]:
        """Parse competitor analysis response into structured format"""
        lines = response.split('\n')
        
        analysis = {
            "common_themes": [],
            "unique_propositions": [],
            "tone_analysis": "",
            "cta_strategies": [],
            "keyword_patterns": [],
            "recommendations": []
        }
        
        current_section = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "themes:" in line.lower():
                current_section = "themes"
            elif "propositions:" in line.lower():
                current_section = "propositions"
            elif "tone:" in line.lower():
                analysis["tone_analysis"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif "cta:" in line.lower():
                current_section = "cta"
            elif "keywords:" in line.lower():
                current_section = "keywords"
            elif "recommendations:" in line.lower():
                current_section = "recommendations"
            elif line.startswith("-") or line.startswith("•"):
                item = line[1:].strip()
                if current_section == "themes":
                    analysis["common_themes"].append(item)
                elif current_section == "propositions":
                    analysis["unique_propositions"].append(item)
                elif current_section == "cta":
                    analysis["cta_strategies"].append(item)
                elif current_section == "keywords":
                    analysis["keyword_patterns"].append(item)
                elif current_section == "recommendations":
                    analysis["recommendations"].append(item)
        
        return analysis

    def _parse_brand_voice_guide(self, response: str, brand_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Parse brand voice guide response into structured format"""
        lines = response.split('\n')
        
        brand_guide = {
            "tone_of_voice": "",
            "messaging_pillars": [],
            "dos_and_donts": {"dos": [], "donts": []},
            "example_copy": [],
            "brand_vocabulary": []
        }
        
        current_section = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "tone:" in line.lower():
                brand_guide["tone_of_voice"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif "pillars:" in line.lower():
                current_section = "pillars"
            elif "do's:" in line.lower() or "dos:" in line.lower():
                current_section = "dos"
            elif "don'ts:" in line.lower() or "donts:" in line.lower():
                current_section = "donts"
            elif "examples:" in line.lower():
                current_section = "examples"
            elif "vocabulary:" in line.lower():
                current_section = "vocabulary"
            elif line.startswith("-") or line.startswith("•"):
                item = line[1:].strip()
                if current_section == "pillars":
                    brand_guide["messaging_pillars"].append(item)
                elif current_section == "dos":
                    brand_guide["dos_and_donts"]["dos"].append(item)
                elif current_section == "donts":
                    brand_guide["dos_and_donts"]["donts"].append(item)
                elif current_section == "examples":
                    brand_guide["example_copy"].append(item)
                elif current_section == "vocabulary":
                    brand_guide["brand_vocabulary"].append(item)
        
        return brand_guide

    # Fallback methods
    def _generate_basic_seo_copy(self, product_data: Dict[str, Any], target_keywords: List[str]) -> Dict[str, Any]:
        """Generate basic SEO copy as fallback"""
        product_name = product_data.get('name', 'Product')
        keywords_str = ' '.join(target_keywords[:3])
        
        return {
            "seo_title": f"{product_name} - {keywords_str}",
            "meta_description": f"Discover amazing {product_name}. {keywords_str}. High quality and trendy design.",
            "product_description": f"Premium {product_name} featuring {keywords_str}. Perfect for trend-conscious consumers.",
            "alt_text_suggestions": [f"{product_name} design", f"{keywords_str} product"],
            "internal_links": [],
            "keyword_density": {kw: 1 for kw in target_keywords[:3]}
        }

    def _generate_basic_social_campaign(self, product_data: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Generate basic social media campaign as fallback"""
        product_name = product_data.get('name', 'Product')
        
        return {
            "post_caption": f"Check out this amazing {product_name}! 🎨✨ Perfect for anyone who loves trendy designs.",
            "hashtags": ["trendy", "design", "fashion", "style"],
            "mentions": [],
            "call_to_action": "Shop Now",
            "platform_specific": {}
        }

    def _generate_basic_email_campaign(self, product_data: Dict[str, Any], campaign_type: str) -> Dict[str, Any]:
        """Generate basic email campaign as fallback"""
        product_name = product_data.get('name', 'Product')
        
        return {
            "subject_line": f"New {product_name} Just Dropped!",
            "preheader": f"Discover the latest {product_name} design",
            "email_body": f"We're excited to introduce our new {product_name}! This trendy design is perfect for fashion-forward consumers.",
            "call_to_action": "Shop Now",
            "personalization_tokens": []
        }

    def _generate_basic_ad_copy(self, product_data: Dict[str, Any], ad_platform: str) -> Dict[str, Any]:
        """Generate basic ad copy as fallback"""
        product_name = product_data.get('name', 'Product')
        
        return {
            "headlines": [f"Amazing {product_name}", f"Trendy {product_name} Design"],
            "descriptions": [f"Discover this unique {product_name} perfect for trend-conscious consumers."],
            "call_to_action": "Shop Now",
            "display_url": "shop.example.com",
            "platform_specific": {}
        }

    def _generate_basic_competitor_analysis(self, competitor_urls: List[str], product_category: str) -> Dict[str, Any]:
        """Generate basic competitor analysis as fallback"""
        return {
            "common_themes": ["Quality", "Innovation", "Style"],
            "unique_propositions": ["Unique designs", "Trend-focused", "Premium quality"],
            "tone_analysis": "Professional and trendy",
            "cta_strategies": ["Shop Now", "Learn More", "Get Started"],
            "keyword_patterns": ["trendy", "design", "fashion"],
            "recommendations": ["Focus on unique value proposition", "Use trending keywords", "Maintain professional tone"]
        }

    def _generate_basic_brand_guide(self, brand_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic brand guide as fallback"""
        return {
            "tone_of_voice": "Professional yet approachable",
            "messaging_pillars": ["Quality", "Innovation", "Trend-conscious"],
            "dos_and_donts": {
                "dos": ["Be professional", "Focus on quality", "Stay trendy"],
                "donts": ["Use slang", "Be overly casual", "Ignore trends"]
            },
            "example_copy": ["Premium quality designs", "Trend-forward products", "Innovative solutions"],
            "brand_vocabulary": ["premium", "trendy", "innovative", "quality"]
        }

    # Performance tracking methods
    async def _store_campaign_data(self, campaign_data: Dict[str, Any]) -> None:
        """Store campaign data in Google Sheets"""
        try:
            if not self.sheets_client or not self.config.gsheet_id:
                return
                
            # Prepare data for sheets
            sheet_data = [
                [
                    datetime.utcnow().isoformat(),
                    campaign_data.get('campaign_name', 'Unknown'),
                    campaign_data.get('platform', 'Unknown'),
                    campaign_data.get('impressions', 0),
                    campaign_data.get('clicks', 0),
                    campaign_data.get('conversions', 0),
                    campaign_data.get('spend', 0),
                    campaign_data.get('roas', 0)
                ]
            ]
            
            # Append to marketing performance sheet
            await self.sheets_client.append_rows(
                spreadsheet_id=self.config.gsheet_id,
                range_name="Marketing_Performance",
                values=sheet_data
            )
            
        except Exception as e:
            print(f"Error storing campaign data: {e}")

    def _analyze_campaign_performance(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze campaign performance metrics"""
        impressions = campaign_data.get('impressions', 0)
        clicks = campaign_data.get('clicks', 0)
        conversions = campaign_data.get('conversions', 0)
        spend = campaign_data.get('spend', 0)
        
        # Calculate key metrics
        ctr = (clicks / impressions * 100) if impressions > 0 else 0
        conversion_rate = (conversions / clicks * 100) if clicks > 0 else 0
        cpc = (spend / clicks) if clicks > 0 else 0
        cpa = (spend / conversions) if conversions > 0 else 0
        
        # Performance insights
        insights = {
            "ctr": round(ctr, 2),
            "conversion_rate": round(conversion_rate, 2),
            "cpc": round(cpc, 2),
            "cpa": round(cpa, 2),
            "performance_score": self._calculate_performance_score(campaign_data),
            "recommendations": self._generate_performance_recommendations(campaign_data)
        }
        
        return insights

    def _calculate_performance_score(self, campaign_data: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)"""
        score = 0
        
        # CTR scoring (0-25 points)
        ctr = campaign_data.get('ctr', 0)
        if ctr >= 2.0:
            score += 25
        elif ctr >= 1.5:
            score += 20
        elif ctr >= 1.0:
            score += 15
        elif ctr >= 0.5:
            score += 10
        else:
            score += 5
        
        # Conversion rate scoring (0-25 points)
        conv_rate = campaign_data.get('conversion_rate', 0)
        if conv_rate >= 3.0:
            score += 25
        elif conv_rate >= 2.0:
            score += 20
        elif conv_rate >= 1.0:
            score += 15
        elif conv_rate >= 0.5:
            score += 10
        else:
            score += 5
        
        # ROAS scoring (0-25 points)
        roas = campaign_data.get('roas', 0)
        if roas >= 4.0:
            score += 25
        elif roas >= 3.0:
            score += 20
        elif roas >= 2.0:
            score += 15
        elif roas >= 1.5:
            score += 10
        else:
            score += 5
        
        # Cost efficiency scoring (0-25 points)
        cpc = campaign_data.get('cpc', 0)
        if cpc <= 0.50:
            score += 25
        elif cpc <= 1.00:
            score += 20
        elif cpc <= 1.50:
            score += 15
        elif cpc <= 2.00:
            score += 10
        else:
            score += 5
        
        return min(score, 100)

    def _generate_performance_recommendations(self, campaign_data: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        ctr = campaign_data.get('ctr', 0)
        if ctr < 1.0:
            recommendations.append("Improve ad creative and messaging to increase click-through rate")
        
        conv_rate = campaign_data.get('conversion_rate', 0)
        if conv_rate < 1.0:
            recommendations.append("Optimize landing page and user experience to improve conversion rate")
        
        roas = campaign_data.get('roas', 0)
        if roas < 2.0:
            recommendations.append("Review pricing strategy and target audience to improve ROAS")
        
        cpc = campaign_data.get('cpc', 0)
        if cpc > 2.0:
            recommendations.append("Optimize keyword targeting and bidding strategy to reduce cost per click")
        
        if not recommendations:
            recommendations.append("Campaign performing well - continue monitoring and optimize for scale")
        
        return recommendations

    def _extract_product_info(self, creative_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Extract product information for marketing AI"""
        products = creative_batch.get("products", [])
        if not products:
            return {"name": "Unknown Product", "style": "Generic", "target_audience": "General"}
        
        # Use first product as primary
        primary_product = products[0]
        
        return {
            "name": primary_product.get("name", "Design Product"),
            "style": primary_product.get("style", "Modern"),
            "target_audience": primary_product.get("target_audience", "Trend-conscious consumers"),
            "keywords": primary_product.get("keywords", []),
            "design_type": primary_product.get("design_type", "Graphic Design"),
            "product_category": primary_product.get("category", "Apparel")
        }

    def _parse_marketing_ai_response(self, ai_text: str, creative_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AI response for marketing copy components"""
        # Default marketing copy structure
        marketing_copy = {
            "title": "",
            "description": "",
            "tags": [],
            "social_media_copy": "",
            "email_subject": "",
            "call_to_action": ""
        }
        
        # Simple parsing - in production you'd want more sophisticated NLP
        lines = ai_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to identify different copy components
            if line.lower().startswith(("title:", "name:", "product:")):
                marketing_copy["title"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif line.lower().startswith(("description:", "desc:", "about:")):
                marketing_copy["description"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif line.lower().startswith(("tags:", "keywords:", "hashtags:")):
                tags_text = line.split(":", 1)[1].strip() if ":" in line else line
                marketing_copy["tags"] = [tag.strip() for tag in tags_text.split(",")]
            elif line.lower().startswith(("social:", "social media:", "instagram:")):
                marketing_copy["social_media_copy"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif line.lower().startswith(("email:", "subject:", "email subject:")):
                marketing_copy["email_subject"] = line.split(":", 1)[1].strip() if ":" in line else line
            elif line.lower().startswith(("cta:", "call to action:", "action:")):
                marketing_copy["call_to_action"] = line.split(":", 1)[1].strip() if ":" in line else line
        
        # Fill in missing components with defaults
        if not marketing_copy["title"]:
            marketing_copy["title"] = creative_batch.get("trend_name", "Trendy Design")
        
        if not marketing_copy["description"]:
            marketing_copy["description"] = f"Unique {creative_batch.get('trend_name', 'design')} perfect for trend-conscious consumers."
        
        if not marketing_copy["tags"]:
            marketing_copy["tags"] = creative_batch.get("keywords", ["trendy", "design", "fashion"])
        
        if not marketing_copy["social_media_copy"]:
            marketing_copy["social_media_copy"] = f"Check out this amazing {creative_batch.get('trend_name', 'design')}! 🎨✨"
        
        if not marketing_copy["email_subject"]:
            marketing_copy["email_subject"] = f"New {creative_batch.get('trend_name', 'Design')} Just Dropped!"
        
        if not marketing_copy["call_to_action"]:
            marketing_copy["call_to_action"] = "Shop Now"
        
        return marketing_copy

    def _generate_basic_marketing_copy(self, creative_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic marketing copy as fallback"""
        trend_name = creative_batch.get("trend_name", "Trendy Design")
        keywords = creative_batch.get("keywords", ["trendy", "design", "fashion"])
        
        return {
            "status": "success",
            "marketing_copy": {
                "title": f"Amazing {trend_name} Design",
                "description": f"Unique {trend_name} perfect for trend-conscious consumers. High-quality design that stands out.",
                "tags": keywords[:5] + ["trendy", "design", "fashion"],
                "social_media_copy": f"Check out this amazing {trend_name}! 🎨✨ Perfect for anyone who loves trendy designs.",
                "email_subject": f"New {trend_name} Just Dropped!",
                "call_to_action": "Shop Now"
            },
            "mcp_model_used": "fallback",
            "execution_time_ms": 0,
            "ai_raw_response": "Basic marketing copy generated as fallback"
        }

    async def run_ab_test(self, creative_batch: Dict[str, Any], test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run A/B test on marketing content variations"""
        try:
            # Create A/B test experiment
            experiment = await self.ab_testing.create_experiment(test_config)
            
            # Generate content variations for each variant
            content_variations = []
            for variant in experiment.variants:
                # Apply adaptive learning parameters to content generation
                optimized_params = self.adaptive_learning.get_optimized_parameters()
                
                # Generate content with optimized parameters
                variant_content = await self._generate_optimized_content(
                    creative_batch, 
                    variant.content_variations,
                    optimized_params
                )
                
                content_variations.append({
                    "variant_id": variant.variant_id,
                    "content": variant_content,
                    "parameters_used": optimized_params
                })
            
            return {
                "status": "success",
                "experiment_id": experiment.experiment_id,
                "experiment_name": experiment.experiment_name,
                "content_variations": content_variations,
                "test_config": test_config,
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"Error running A/B test: {e}")
            return {
                "status": "error",
                "error": str(e),
                "created_at": datetime.utcnow().isoformat()
            }
    
    async def _generate_optimized_content(self, creative_batch: Dict[str, Any], 
                                        variations: Dict[str, Any], 
                                        optimized_params: Dict[str, float]) -> Dict[str, Any]:
        """Generate content using optimized learning parameters"""
        try:
            # Apply optimized parameters to content generation
            enhanced_batch = self._apply_learning_parameters(creative_batch, optimized_params)
            
            # Generate marketing copy with enhanced parameters
            marketing_copy = await self.run(enhanced_batch)
            
            # Apply content variations
            if variations:
                marketing_copy = self._apply_content_variations(marketing_copy, variations)
            
            return marketing_copy
            
        except Exception as e:
            print(f"Error generating optimized content: {e}")
            return creative_batch
    
    def _apply_learning_parameters(self, creative_batch: Dict[str, Any], 
                                 optimized_params: Dict[str, float]) -> Dict[str, Any]:
        """Apply learned optimization parameters to creative batch"""
        try:
            enhanced_batch = creative_batch.copy()
            
            # Apply content creativity parameter
            if "content_creativity" in optimized_params:
                creativity_factor = optimized_params["content_creativity"]
                enhanced_batch["creativity_level"] = creativity_factor
                enhanced_batch["content_style"] = "creative" if creativity_factor > 0.7 else "professional"
            
            # Apply emotional intensity parameter
            if "emotional_intensity" in optimized_params:
                emotional_factor = optimized_params["emotional_intensity"]
                enhanced_batch["emotional_tone"] = "high" if emotional_factor > 0.7 else "moderate" if emotional_factor > 0.4 else "low"
            
            # Apply urgency level parameter
            if "urgency_level" in optimized_params:
                urgency_factor = optimized_params["urgency_level"]
                enhanced_batch["urgency_tone"] = "high" if urgency_factor > 0.7 else "moderate" if urgency_factor > 0.4 else "low"
            
            # Apply social proof weight parameter
            if "social_proof_weight" in optimized_params:
                social_proof_factor = optimized_params["social_proof_weight"]
                enhanced_batch["social_proof_elements"] = "prominent" if social_proof_factor > 0.6 else "moderate" if social_proof_factor > 0.3 else "minimal"
            
            # Apply audience targeting parameter
            if "audience_targeting" in optimized_params:
                targeting_factor = optimized_params["audience_targeting"]
                enhanced_batch["targeting_precision"] = "high" if targeting_factor > 0.7 else "moderate" if targeting_factor > 0.4 else "basic"
            
            return enhanced_batch
            
        except Exception as e:
            print(f"Error applying learning parameters: {e}")
            return creative_batch
    
    def _apply_content_variations(self, marketing_copy: Dict[str, Any], 
                                variations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply content variations to marketing copy"""
        try:
            enhanced_copy = marketing_copy.copy()
            
            # Apply headline variations
            if "headline_style" in variations:
                enhanced_copy["headline_variation"] = variations["headline_style"]
            
            # Apply copy length variations
            if "copy_length" in variations:
                enhanced_copy["copy_length"] = variations["copy_length"]
            
            # Apply tone variations
            if "tone" in variations:
                enhanced_copy["tone_variation"] = variations["tone"]
            
            # Apply CTA variations
            if "cta_style" in variations:
                enhanced_copy["cta_variation"] = variations["cta_style"]
            
            return enhanced_copy
            
        except Exception as e:
            print(f"Error applying content variations: {e}")
            return marketing_copy
    
    async def record_ab_test_results(self, experiment_id: str, variant_id: str, 
                                   interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Record A/B test interaction results"""
        try:
            # Record interaction in A/B testing framework
            for interaction_type, value in interaction_data.items():
                if interaction_type in ["impression", "click", "conversion", "revenue", "engagement"]:
                    await self.ab_testing.record_interaction(
                        experiment_id, 
                        variant_id, 
                        interaction_type, 
                        value if isinstance(value, (int, float)) else 1.0
                    )
            
            # Record performance for adaptive learning
            performance_score = self._calculate_interaction_performance_score(interaction_data)
            await self.adaptive_learning.record_performance({
                "performance_score": performance_score,
                "metrics": interaction_data
            })
            
            return {
                "status": "success",
                "experiment_id": experiment_id,
                "variant_id": variant_id,
                "performance_score": performance_score,
                "recorded_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"Error recording A/B test results: {e}")
            return {
                "status": "error",
                "error": str(e),
                "recorded_at": datetime.utcnow().isoformat()
            }
    
    def _calculate_interaction_performance_score(self, interaction_data: Dict[str, Any]) -> float:
        """Calculate performance score from interaction data"""
        try:
            score = 0.0
            weights = {
                "impression": 0.1,
                "click": 0.3,
                "conversion": 0.5,
                "revenue": 0.8,
                "engagement": 0.2
            }
            
            for metric, weight in weights.items():
                if metric in interaction_data:
                    value = interaction_data[metric]
                    if isinstance(value, (int, float)) and value > 0:
                        score += weight * min(value / 100, 1.0)  # Normalize to 0-1
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            print(f"Error calculating performance score: {e}")
            return 0.0
    
    async def get_ab_test_winner(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get the winning variant from A/B test"""
        try:
            winning_variant = await self.ab_testing.get_winning_variant(experiment_id)
            
            if winning_variant:
                # Get experiment summary
                experiment_summary = await self.ab_testing.get_experiment_summary(experiment_id)
                
                return {
                    "status": "success",
                    "winning_variant": {
                        "variant_id": winning_variant.variant_id,
                        "variant_name": winning_variant.variant_name,
                        "performance_metrics": {
                            "impressions": winning_variant.impressions,
                            "clicks": winning_variant.clicks,
                            "conversions": winning_variant.conversions,
                            "ctr": winning_variant.ctr,
                            "conversion_rate": winning_variant.conversion_rate,
                            "revenue": winning_variant.revenue,
                            "roi": winning_variant.roi
                        }
                    },
                    "experiment_summary": experiment_summary,
                    "determined_at": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "status": "no_winner",
                    "message": "No statistically significant winner yet",
                    "experiment_id": experiment_id,
                    "checked_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            print(f"Error getting A/B test winner: {e}")
            return {
                "status": "error",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from adaptive learning system"""
        try:
            learning_summary = await self.adaptive_learning.get_learning_summary()
            optimized_params = self.adaptive_learning.get_optimized_parameters()
            parameter_trends = self.adaptive_learning.get_parameter_trends()
            
            # Generate actionable insights
            insights = self._generate_learning_insights(optimized_params, parameter_trends)
            
            return {
                "status": "success",
                "learning_summary": learning_summary,
                "optimized_parameters": optimized_params,
                "parameter_trends": parameter_trends,
                "actionable_insights": insights,
                "retrieved_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting learning insights: {e}")
            return {
                "status": "error",
                "error": str(e),
                "retrieved_at": datetime.utcnow().isoformat()
            }
    
    def _generate_learning_insights(self, optimized_params: Dict[str, float], 
                                  parameter_trends: Dict[str, str]) -> List[str]:
        """Generate actionable insights from learning data"""
        insights = []
        
        try:
            # Content creativity insights
            if "content_creativity" in optimized_params:
                creativity = optimized_params["content_creativity"]
                if creativity > 0.8:
                    insights.append("High creativity content is performing well - consider increasing creative elements")
                elif creativity < 0.4:
                    insights.append("Low creativity content underperforming - try more creative approaches")
            
            # Emotional intensity insights
            if "emotional_intensity" in optimized_params:
                emotional = optimized_params["emotional_intensity"]
                if emotional > 0.8:
                    insights.append("High emotional content driving engagement - leverage emotional triggers")
                elif emotional < 0.3:
                    insights.append("Low emotional content may be too clinical - add emotional elements")
            
            # Urgency insights
            if "urgency_level" in optimized_params:
                urgency = optimized_params["urgency_level"]
                if urgency > 0.7:
                    insights.append("Urgency tactics effective - consider time-limited offers")
                elif urgency < 0.3:
                    insights.append("Low urgency may reduce conversions - add scarcity elements")
            
            # Social proof insights
            if "social_proof_weight" in optimized_params:
                social_proof = optimized_params["social_proof_weight"]
                if social_proof > 0.6:
                    insights.append("Social proof highly effective - emphasize testimonials and reviews")
                elif social_proof < 0.3:
                    insights.append("Social proof underutilized - add customer success stories")
            
            # Trend-based insights
            for param_name, trend in parameter_trends.items():
                if trend == "increasing":
                    insights.append(f"{param_name.replace('_', ' ').title()} parameter trending up - performance improving")
                elif trend == "decreasing":
                    insights.append(f"{param_name.replace('_', ' ').title()} parameter trending down - may need adjustment")
            
            return insights[:10]  # Limit to top 10 insights
            
        except Exception as e:
            print(f"Error generating learning insights: {e}")
            return ["Unable to generate insights at this time"]


@dataclass
class ABTestVariant:
    """A/B test variant configuration"""
    variant_id: str
    variant_name: str
    content_variations: Dict[str, Any]
    traffic_allocation: float  # Percentage of traffic (0.0 to 1.0)
    is_control: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not 0.0 <= self.traffic_allocation <= 1.0:
            raise ValueError("Traffic allocation must be between 0.0 and 1.0")


@dataclass
class ABTestResult:
    """A/B test result data"""
    variant_id: str
    variant_name: str
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    revenue: float = 0.0
    engagement_rate: float = 0.0
    conversion_rate: float = 0.0
    ctr: float = 0.0
    cpc: float = 0.0
    cpa: float = 0.0
    roi: float = 0.0
    
    def calculate_metrics(self) -> None:
        """Calculate derived metrics"""
        if self.impressions > 0:
            self.ctr = self.clicks / self.impressions
        if self.clicks > 0:
            self.cpc = self.revenue / self.clicks
        if self.conversions > 0:
            self.cpa = self.revenue / self.conversions
        if self.revenue > 0:
            self.roi = (self.revenue - self.cpc * self.clicks) / (self.cpc * self.clicks) * 100


@dataclass
class ABTestExperiment:
    """A/B test experiment configuration"""
    experiment_id: str
    experiment_name: str
    description: str
    variants: List[ABTestVariant]
    start_date: datetime
    end_date: Optional[datetime] = None
    status: str = "active"  # active, paused, completed, cancelled
    primary_metric: str = "conversion_rate"
    confidence_level: float = 0.95
    minimum_sample_size: int = 1000
    traffic_split: str = "equal"  # equal, weighted, dynamic
    
    def __post_init__(self):
        if len(self.variants) < 2:
            raise ValueError("A/B test must have at least 2 variants")
        
        # Validate traffic allocation
        total_allocation = sum(v.traffic_allocation for v in self.variants)
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(f"Total traffic allocation must equal 1.0, got {total_allocation}")


class ABTestingFramework:
    """A/B testing framework for marketing content optimization"""
    
    def __init__(self, config: Any = None) -> None:
        self.config = config or load_config()
        self.experiments: Dict[str, ABTestExperiment] = {}
        self.results: Dict[str, Dict[str, ABTestResult]] = {}
        
    async def create_experiment(self, experiment_config: Dict[str, Any]) -> ABTestExperiment:
        """Create a new A/B test experiment"""
        try:
            # Generate unique experiment ID
            experiment_id = f"ab_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            
            # Create variants
            variants = []
            for i, variant_config in enumerate(experiment_config["variants"]):
                is_control = i == 0  # First variant is control
                traffic_allocation = 1.0 / len(experiment_config["variants"])  # Equal split by default
                
                variant = ABTestVariant(
                    variant_id=f"{experiment_id}_variant_{i}",
                    variant_name=variant_config["name"],
                    content_variations=variant_config["content"],
                    traffic_allocation=traffic_allocation,
                    is_control=is_control
                )
                variants.append(variant)
            
            # Create experiment
            experiment = ABTestExperiment(
                experiment_id=experiment_id,
                experiment_name=experiment_config["name"],
                description=experiment_config.get("description", ""),
                variants=variants,
                start_date=datetime.utcnow(),
                primary_metric=experiment_config.get("primary_metric", "conversion_rate"),
                confidence_level=experiment_config.get("confidence_level", 0.95),
                minimum_sample_size=experiment_config.get("minimum_sample_size", 1000)
            )
            
            self.experiments[experiment_id] = experiment
            self.results[experiment_id] = {}
            
            # Initialize results for each variant
            for variant in variants:
                self.results[experiment_id][variant.variant_id] = ABTestResult(
                    variant_id=variant.variant_id,
                    variant_name=variant.variant_name
                )
            
            return experiment
            
        except Exception as e:
            print(f"Error creating A/B test experiment: {e}")
            raise
    
    async def record_interaction(self, experiment_id: str, variant_id: str, 
                               interaction_type: str, value: float = 1.0) -> None:
        """Record user interaction with a test variant"""
        try:
            if experiment_id not in self.results or variant_id not in self.results[experiment_id]:
                print(f"Invalid experiment or variant ID: {experiment_id}, {variant_id}")
                return
            
            result = self.results[experiment_id][variant_id]
            
            if interaction_type == "impression":
                result.impressions += 1
            elif interaction_type == "click":
                result.clicks += 1
            elif interaction_type == "conversion":
                result.conversions += 1
            elif interaction_type == "revenue":
                result.revenue += value
            elif interaction_type == "engagement":
                result.engagement_rate = (result.engagement_rate + value) / 2
            
            # Recalculate metrics
            result.calculate_metrics()
            
        except Exception as e:
            print(f"Error recording interaction: {e}")
    
    async def get_winning_variant(self, experiment_id: str) -> Optional[ABTestResult]:
        """Determine the winning variant based on statistical significance"""
        try:
            if experiment_id not in self.experiments or experiment_id not in self.results:
                return None
            
            experiment = self.experiments[experiment_id]
            results = self.results[experiment_id]
            
            # Check if we have enough data
            total_impressions = sum(r.impressions for r in results.values())
            if total_impressions < experiment.minimum_sample_size:
                return None
            
            # Get control variant
            control_variant = next((v for v in experiment.variants if v.is_control), None)
            if not control_variant:
                return None
            
            control_result = results[control_variant.variant_id]
            
            # Compare each variant against control
            best_variant = None
            best_improvement = 0.0
            
            for variant in experiment.variants:
                if variant.is_control:
                    continue
                
                variant_result = results[variant.variant_id]
                
                # Perform statistical significance test
                if self._is_statistically_significant(control_result, variant_result, experiment.primary_metric):
                    improvement = self._calculate_improvement(control_result, variant_result, experiment.primary_metric)
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_variant = variant_result
            
            return best_variant if best_improvement > 0 else None
            
        except Exception as e:
            print(f"Error determining winning variant: {e}")
            return None
    
    def _is_statistically_significant(self, control: ABTestResult, variant: ABTestResult, 
                                    metric: str) -> bool:
        """Perform statistical significance testing using chi-square test"""
        try:
            if metric == "conversion_rate":
                # Chi-square test for conversion rates
                control_conversions = control.conversions
                control_non_conversions = control.impressions - control.conversions
                variant_conversions = variant.conversions
                variant_non_conversions = variant.impressions - variant.conversions
                
                if control_conversions == 0 or variant_conversions == 0:
                    return False
                
                # Create contingency table
                observed = np.array([[control_conversions, control_non_conversions],
                                   [variant_conversions, variant_non_conversions]])
                
                # Perform chi-square test
                chi2, p_value = stats.chi2_contingency(observed)[:2]
                
                return p_value < (1 - 0.95)  # 95% confidence level
                
            elif metric == "ctr":
                # Z-test for click-through rates
                control_ctr = control.ctr
                variant_ctr = variant.ctr
                
                if control.impressions == 0 or variant.impressions == 0:
                    return False
                
                # Calculate standard error
                control_se = math.sqrt(control_ctr * (1 - control_ctr) / control.impressions)
                variant_se = math.sqrt(variant_ctr * (1 - variant_ctr) / variant.impressions)
                
                # Calculate z-score
                z_score = abs(variant_ctr - control_ctr) / math.sqrt(control_se**2 + variant_se**2)
                
                # Two-tailed test at 95% confidence level
                return z_score > 1.96
                
            return False
            
        except Exception as e:
            print(f"Error in statistical significance testing: {e}")
            return False
    
    def _calculate_improvement(self, control: ABTestResult, variant: ABTestResult, 
                             metric: str) -> float:
        """Calculate improvement percentage of variant over control"""
        try:
            if metric == "conversion_rate":
                if control.conversion_rate == 0:
                    return 0.0
                return ((variant.conversion_rate - control.conversion_rate) / control.conversion_rate) * 100
            elif metric == "ctr":
                if control.ctr == 0:
                    return 0.0
                return ((variant.ctr - control.ctr) / control.ctr) * 100
            elif metric == "revenue":
                if control.revenue == 0:
                    return 0.0
                return ((variant.revenue - control.revenue) / control.revenue) * 100
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating improvement: {e}")
            return 0.0
    
    async def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of A/B test experiment"""
        try:
            if experiment_id not in self.experiments:
                return {}
            
            experiment = self.experiments[experiment_id]
            results = self.results[experiment_id]
            
            summary = {
                "experiment_id": experiment_id,
                "experiment_name": experiment.experiment_name,
                "status": experiment.status,
                "start_date": experiment.start_date.isoformat(),
                "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
                "primary_metric": experiment.primary_metric,
                "confidence_level": experiment.confidence_level,
                "total_impressions": sum(r.impressions for r in results.values()),
                "variants": []
            }
            
            for variant in experiment.variants:
                result = results[variant.variant_id]
                variant_summary = {
                    "variant_id": variant.variant_id,
                    "variant_name": variant.variant_name,
                    "is_control": variant.is_control,
                    "traffic_allocation": variant.traffic_allocation,
                    "impressions": result.impressions,
                    "clicks": result.clicks,
                    "conversions": result.conversions,
                    "ctr": result.ctr,
                    "conversion_rate": result.conversion_rate,
                    "revenue": result.revenue,
                    "roi": result.roi
                }
                summary["variants"].append(variant_summary)
            
            return summary
            
        except Exception as e:
            print(f"Error getting experiment summary: {e}")
            return {}
    
    async def pause_experiment(self, experiment_id: str) -> bool:
        """Pause an active A/B test experiment"""
        try:
            if experiment_id in self.experiments:
                self.experiments[experiment_id].status = "paused"
                return True
            return False
        except Exception as e:
            print(f"Error pausing experiment: {e}")
            return False
    
    async def resume_experiment(self, experiment_id: str) -> bool:
        """Resume a paused A/B test experiment"""
        try:
            if experiment_id in self.experiments:
                self.experiments[experiment_id].status = "active"
                return True
            return False
        except Exception as e:
            print(f"Error resuming experiment: {e}")
            return False
    
    async def end_experiment(self, experiment_id: str) -> bool:
        """End an A/B test experiment and mark as completed"""
        try:
            if experiment_id in self.experiments:
                self.experiments[experiment_id].status = "completed"
                self.experiments[experiment_id].end_date = datetime.utcnow()
                return True
            return False
        except Exception as e:
            print(f"Error ending experiment: {e}")
            return False
    
    async def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Get list of all experiments with basic info"""
        try:
            experiments_list = []
            for experiment_id, experiment in self.experiments.items():
                results = self.results.get(experiment_id, {})
                total_impressions = sum(r.impressions for r in results.values())
                
                experiments_list.append({
                    "experiment_id": experiment_id,
                    "experiment_name": experiment.experiment_name,
                    "status": experiment.status,
                    "start_date": experiment.start_date.isoformat(),
                    "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
                    "total_impressions": total_impressions,
                    "variant_count": len(experiment.variants)
                })
            
            return experiments_list
            
        except Exception as e:
            print(f"Error getting experiments list: {e}")
            return []
    
    async def get_experiment_stats(self, experiment_id: str) -> Dict[str, Any]:
        """Get statistical analysis of experiment results"""
        try:
            if experiment_id not in self.experiments or experiment_id not in self.results:
                return {}
            
            experiment = self.experiments[experiment_id]
            results = self.results[experiment_id]
            
            # Calculate overall statistics
            total_impressions = sum(r.impressions for r in results.values())
            total_clicks = sum(r.clicks for r in results.values())
            total_conversions = sum(r.conversions for r in results.values())
            total_revenue = sum(r.revenue for r in results.values())
            
            overall_ctr = total_clicks / total_impressions if total_impressions > 0 else 0
            overall_conversion_rate = total_conversions / total_impressions if total_impressions > 0 else 0
            
            # Calculate confidence intervals for key metrics
            stats = {
                "experiment_id": experiment_id,
                "total_impressions": total_impressions,
                "total_clicks": total_clicks,
                "total_conversions": total_conversions,
                "total_revenue": total_revenue,
                "overall_ctr": overall_ctr,
                "overall_conversion_rate": overall_conversion_rate,
                "confidence_level": experiment.confidence_level,
                "minimum_sample_size": experiment.minimum_sample_size,
                "sample_size_achieved": total_impressions >= experiment.minimum_sample_size,
                "statistical_power": self._calculate_statistical_power(experiment_id),
                "variant_comparisons": []
            }
            
            # Compare each variant against control
            control_variant = next((v for v in experiment.variants if v.is_control), None)
            if control_variant:
                control_result = results[control_variant.variant_id]
                
                for variant in experiment.variants:
                    if variant.is_control:
                        continue
                    
                    variant_result = results[variant.variant_id]
                    comparison = {
                        "variant_id": variant.variant_id,
                        "variant_name": variant.variant_name,
                        "vs_control": {
                            "ctr_improvement": self._calculate_improvement(control_result, variant_result, "ctr"),
                            "conversion_improvement": self._calculate_improvement(control_result, variant_result, "conversion_rate"),
                            "revenue_improvement": self._calculate_improvement(control_result, variant_result, "revenue"),
                            "statistically_significant": self._is_statistically_significant(control_result, variant_result, experiment.primary_metric)
                        }
                    }
                    stats["variant_comparisons"].append(comparison)
            
            return stats
            
        except Exception as e:
            print(f"Error getting experiment stats: {e}")
            return {}
    
    def _calculate_statistical_power(self, experiment_id: str) -> float:
        """Calculate statistical power of the experiment"""
        try:
            if experiment_id not in self.experiments or experiment_id not in self.results:
                return 0.0
            
            experiment = self.experiments[experiment_id]
            results = self.results[experiment_id]
            
            # Simple power calculation based on sample size and effect size
            total_impressions = sum(r.impressions for r in results.values())
            
            if total_impressions < experiment.minimum_sample_size:
                return total_impressions / experiment.minimum_sample_size
            
            # If we have sufficient sample size, power is high
            return min(0.95, 0.8 + (total_impressions - experiment.minimum_sample_size) / experiment.minimum_sample_size * 0.15)
            
        except Exception as e:
            print(f"Error calculating statistical power: {e}")
            return 0.0
    
    async def clone_experiment(self, experiment_id: str, new_name: str = None) -> Optional[ABTestExperiment]:
        """Clone an existing experiment with a new name"""
        try:
            if experiment_id not in self.experiments:
                return None
            
            original_experiment = self.experiments[experiment_id]
            
            # Generate new experiment ID
            new_experiment_id = f"ab_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            
            # Clone variants
            new_variants = []
            for variant in original_experiment.variants:
                new_variant = ABTestVariant(
                    variant_id=f"{new_experiment_id}_variant_{len(new_variants)}",
                    variant_name=variant.variant_name,
                    content_variations=variant.content_variations.copy(),
                    traffic_allocation=variant.traffic_allocation,
                    is_control=variant.is_control
                )
                new_variants.append(new_variant)
            
            # Create new experiment
            new_experiment = ABTestExperiment(
                experiment_id=new_experiment_id,
                experiment_name=new_name or f"{original_experiment.experiment_name}_clone",
                description=f"Cloned from {original_experiment.experiment_name}",
                variants=new_variants,
                start_date=datetime.utcnow(),
                primary_metric=original_experiment.primary_metric,
                confidence_level=original_experiment.confidence_level,
                minimum_sample_size=original_experiment.minimum_sample_size
            )
            
            self.experiments[new_experiment_id] = new_experiment
            self.results[new_experiment_id] = {}
            
            # Initialize results for each variant
            for variant in new_variants:
                self.results[new_experiment_id][variant.variant_id] = ABTestResult(
                    variant_id=variant.variant_id,
                    variant_name=variant.variant_name
                )
            
            return new_experiment
            
        except Exception as e:
            print(f"Error cloning experiment: {e}")
            return None
    
    async def get_experiment_recommendations(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get recommendations for experiment optimization"""
        try:
            if experiment_id not in self.experiments or experiment_id not in self.results:
                return []
            
            experiment = self.experiments[experiment_id]
            results = self.results[experiment_id]
            
            recommendations = []
            
            # Check sample size
            total_impressions = sum(r.impressions for r in results.values())
            if total_impressions < experiment.minimum_sample_size:
                recommendations.append({
                    "type": "sample_size",
                    "priority": "high",
                    "description": f"Need {experiment.minimum_sample_size - total_impressions} more impressions for statistical significance",
                    "suggested_actions": [
                        "Continue running the experiment",
                        "Increase traffic allocation if possible",
                        "Consider extending experiment duration"
                    ]
                })
            
            # Check for clear winners
            control_variant = next((v for v in experiment.variants if v.is_control), None)
            if control_variant:
                control_result = results[control_variant.variant_id]
                
                for variant in experiment.variants:
                    if variant.is_control:
                        continue
                    
                    variant_result = results[variant.variant_id]
                    
                    # Check if variant is significantly better
                    if self._is_statistically_significant(control_result, variant_result, experiment.primary_metric):
                        improvement = self._calculate_improvement(control_result, variant_result, experiment.primary_metric)
                        
                        if improvement > 0.1:  # 10% improvement
                            recommendations.append({
                                "type": "significant_improvement",
                                "priority": "medium",
                                "variant": variant.variant_name,
                                "description": f"Variant {variant.variant_name} shows {improvement:.1%} improvement",
                                "suggested_actions": [
                                    "Consider ending experiment early",
                                    "Scale up winning variant",
                                    "Analyze what makes this variant successful"
                                ]
                            })
            
            # Check for poor performers
            for variant in experiment.variants:
                if variant.is_control:
                    continue
                
                variant_result = results[variant.variant_id]
                if variant_result.impressions > 100:  # Only check if we have enough data
                    if variant_result.conversion_rate < control_result.conversion_rate * 0.8:
                        recommendations.append({
                            "type": "poor_performance",
                            "priority": "low",
                            "variant": variant.variant_name,
                            "description": f"Variant {variant.variant_name} is underperforming by 20%+",
                            "suggested_actions": [
                                "Consider pausing this variant",
                                "Analyze why it's underperforming",
                                "Test different content variations"
                            ]
                        })
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting experiment recommendations: {e}")
            return []
    
    async def export_experiment_data(self, experiment_id: str) -> Dict[str, Any]:
        """Export experiment data for analysis"""
        try:
            if experiment_id not in self.experiments or experiment_id not in self.results:
                return {"error": "Experiment not found"}
            
            experiment = self.experiments[experiment_id]
            results = self.results[experiment_id]
            
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "experiment": {
                    "experiment_id": experiment.experiment_id,
                    "experiment_name": experiment.experiment_name,
                    "description": experiment.description,
                    "start_date": experiment.start_date.isoformat(),
                    "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
                    "status": experiment.status,
                    "primary_metric": experiment.primary_metric,
                    "confidence_level": experiment.confidence_level,
                    "minimum_sample_size": experiment.minimum_sample_size,
                    "traffic_split": experiment.traffic_split
                },
                "variants": [],
                "results": {},
                "statistics": await self.get_experiment_stats(experiment_id)
            }
            
            # Export variant data
            for variant in experiment.variants:
                variant_data = {
                    "variant_id": variant.variant_id,
                    "variant_name": variant.variant_name,
                    "is_control": variant.is_control,
                    "traffic_allocation": variant.traffic_allocation,
                    "created_at": variant.created_at.isoformat(),
                    "content_variations": variant.content_variations
                }
                export_data["variants"].append(variant_data)
            
            # Export results data
            for variant_id, result in results.items():
                export_data["results"][variant_id] = {
                    "variant_id": result.variant_id,
                    "variant_name": result.variant_name,
                    "impressions": result.impressions,
                    "clicks": result.clicks,
                    "conversions": result.conversions,
                    "revenue": result.revenue,
                    "engagement_rate": result.engagement_rate,
                    "conversion_rate": result.conversion_rate,
                    "ctr": result.ctr,
                    "cpc": result.cpc,
                    "cpa": result.cpa,
                    "roi": result.roi
                }
            
            return export_data
            
        except Exception as e:
            print(f"Error exporting experiment data: {e}")
            return {"error": str(e)}
    
    async def get_experiment_health_score(self, experiment_id: str) -> float:
        """Calculate overall health score for an experiment"""
        try:
            experiment = await self._get_experiment(experiment_id)
            if not experiment:
                return 0.0
            
            # Get experiment results
            results = await self._get_experiment_results(experiment_id)
            if not results:
                return 0.0
            
            # Calculate health factors
            sample_size_score = min(1.0, sum(r.impressions for r in results) / experiment.minimum_sample_size)
            statistical_power = await self._calculate_statistical_power(experiment_id)
            data_quality = self._calculate_data_quality_score(results)
            execution_health = self._calculate_execution_health(experiment)
            
            # Weighted average
            health_score = (
                sample_size_score * 0.3 +
                statistical_power * 0.3 +
                data_quality * 0.2 +
                execution_health * 0.2
            )
            
            return round(health_score, 3)
            
        except Exception as e:
            print(f"Error calculating experiment health score: {e}")
            return 0.0
    
    def _calculate_data_quality_score(self, results: List[ABTestResult]) -> float:
        """Calculate data quality score based on result consistency"""
        if not results or len(results) < 2:
            return 0.0
        
        # Check for data anomalies
        total_impressions = sum(r.impressions for r in results)
        if total_impressions == 0:
            return 0.0
        
        # Calculate conversion rate consistency
        conversion_rates = [r.conversion_rate for r in results if r.impressions > 0]
        if len(conversion_rates) < 2:
            return 0.5
        
        # Check for reasonable conversion rates (not too extreme)
        avg_rate = statistics.mean(conversion_rates)
        if avg_rate > 0.5 or avg_rate < 0.001:  # Suspicious rates
            return 0.3
        
        # Calculate variance (lower variance = higher quality)
        variance = statistics.variance(conversion_rates)
        max_variance = 0.1  # Reasonable threshold
        quality_score = max(0.0, 1.0 - (variance / max_variance))
        
        return min(1.0, quality_score)
    
    def _calculate_execution_health(self, experiment: ABTestExperiment) -> float:
        """Calculate execution health based on experiment status and timing"""
        if experiment.status == "completed":
            return 1.0
        elif experiment.status == "active":
            # Check if experiment has been running too long
            days_running = (datetime.utcnow() - experiment.start_date).days
            if days_running > 30:  # Long-running experiment
                return 0.7
            elif days_running > 7:  # Medium-running experiment
                return 0.9
            else:  # New experiment
                return 0.8
        elif experiment.status == "paused":
            return 0.5
        else:  # cancelled or other status
            return 0.3

    async def get_experiment_recommendations(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get actionable recommendations for experiment optimization"""
        try:
            experiment = await self._get_experiment(experiment_id)
            if not experiment:
                return []
            
            results = await self._get_experiment_results(experiment_id)
            if not results:
                return []
            
            recommendations = []
            
            # Sample size recommendations
            total_impressions = sum(r.impressions for r in results)
            if total_impressions < experiment.minimum_sample_size:
                recommendations.append({
                    "type": "sample_size",
                    "priority": "high",
                    "message": f"Increase sample size. Current: {total_impressions}, Required: {experiment.minimum_sample_size}",
                    "action": "extend_experiment"
                })
            
            # Statistical significance recommendations
            control = next((r for r in results if r.is_control), None)
            if control:
                for variant in results:
                    if not variant.is_control:
                        is_significant = self._is_statistically_significant(control, variant, experiment.primary_metric)
                        if not is_significant:
                            recommendations.append({
                                "type": "statistical_significance",
                                "priority": "medium",
                                "message": f"Variant '{variant.variant_name}' not statistically significant for {experiment.primary_metric}",
                                "action": "continue_monitoring"
                            })
            
            # Performance recommendations
            best_variant = max(results, key=lambda r: getattr(r, experiment.primary_metric, 0))
            if best_variant and not best_variant.is_control:
                improvement = self._calculate_improvement(control, best_variant, experiment.primary_metric)
                if improvement > 0.1:  # 10% improvement
                    recommendations.append({
                        "type": "performance",
                        "priority": "high",
                        "message": f"Variant '{best_variant.variant_name}' shows {improvement:.1%} improvement",
                        "action": "consider_implementation"
                    })
            
            # Traffic allocation recommendations
            if experiment.traffic_split == "equal":
                # Suggest dynamic allocation for better variants
                recommendations.append({
                    "type": "traffic_optimization",
                    "priority": "low",
                    "message": "Consider dynamic traffic allocation for better performing variants",
                    "action": "optimize_traffic"
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating experiment recommendations: {e}")
            return []

    async def export_experiment_data(self, experiment_id: str) -> Dict[str, Any]:
        """Export comprehensive experiment data for analysis"""
        try:
            experiment = await self._get_experiment(experiment_id)
            if not experiment:
                return {}
            
            results = await self._get_experiment_results(experiment_id)
            interactions = await self._get_experiment_interactions(experiment_id)
            
            export_data = {
                "experiment_info": {
                    "experiment_id": experiment.experiment_id,
                    "experiment_name": experiment.experiment_name,
                    "description": experiment.description,
                    "start_date": experiment.start_date.isoformat(),
                    "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
                    "status": experiment.status,
                    "primary_metric": experiment.primary_metric,
                    "confidence_level": experiment.confidence_level,
                    "minimum_sample_size": experiment.minimum_sample_size,
                    "traffic_split": experiment.traffic_split
                },
                "variants": [
                    {
                        "variant_id": v.variant_id,
                        "variant_name": v.variant_name,
                        "traffic_allocation": v.traffic_allocation,
                        "is_control": v.is_control,
                        "created_at": v.created_at.isoformat()
                    } for v in experiment.variants
                ],
                "results": [
                    {
                        "variant_id": r.variant_id,
                        "variant_name": r.variant_name,
                        "impressions": r.impressions,
                        "clicks": r.clicks,
                        "conversions": r.conversions,
                        "revenue": r.revenue,
                        "engagement_rate": r.engagement_rate,
                        "conversion_rate": r.conversion_rate,
                        "ctr": r.ctr,
                        "cpc": r.cpc,
                        "cpa": r.cpa,
                        "roi": r.roi
                    } for r in results
                ],
                "interactions": interactions,
                "statistical_analysis": {
                    "power_analysis": await self._calculate_statistical_power(experiment_id),
                    "significance_tests": self._run_significance_tests(results),
                    "confidence_intervals": self._calculate_confidence_intervals(results)
                },
                "export_timestamp": datetime.utcnow().isoformat()
            }
            
            return export_data
            
        except Exception as e:
            print(f"Error exporting experiment data: {e}")
            return {}
    
    def _run_significance_tests(self, results: List[ABTestResult]) -> Dict[str, Any]:
        """Run comprehensive statistical significance tests"""
        if len(results) < 2:
            return {}
        
        control = next((r for r in results if r.is_control), results[0])
        variants = [r for r in results if not r.is_control]
        
        significance_tests = {}
        
        for variant in variants:
            # Chi-square test for conversion rates
            if control.conversions > 0 and variant.conversions > 0:
                try:
                    # Create contingency table
                    control_success = control.conversions
                    control_failure = control.impressions - control.conversions
                    variant_success = variant.conversions
                    variant_failure = variant.impressions - variant.conversions
                    
                    chi2_stat, p_value = stats.chi2_contingency([
                        [control_success, control_failure],
                        [variant_success, variant_failure]
                    ])[:2]
                    
                    significance_tests[f"{variant.variant_name}_conversion"] = {
                        "test_type": "chi_square",
                        "statistic": float(chi2_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05
                    }
                except Exception:
                    pass
            
            # T-test for continuous metrics (if applicable)
            if hasattr(variant, 'revenue') and variant.revenue > 0:
                try:
                    # Simulate revenue data (in real implementation, you'd have actual data)
                    control_revenue = [control.revenue] * control.conversions if control.conversions > 0 else [0]
                    variant_revenue = [variant.revenue] * variant.conversions if variant.conversions > 0 else [0]
                    
                    if len(control_revenue) > 1 and len(variant_revenue) > 1:
                        t_stat, p_value = stats.ttest_ind(control_revenue, variant_revenue)
                        significance_tests[f"{variant.variant_name}_revenue"] = {
                            "test_type": "t_test",
                            "statistic": float(t_stat),
                            "p_value": float(p_value),
                            "significant": p_value < 0.05
                        }
                except Exception:
                    pass
        
        return significance_tests
    
    def _calculate_confidence_intervals(self, results: List[ABTestResult]) -> Dict[str, Any]:
        """Calculate confidence intervals for key metrics"""
        confidence_intervals = {}
        
        for result in results:
            if result.impressions > 0:
                # Conversion rate confidence interval
                p_hat = result.conversion_rate
                n = result.impressions
                
                if n * p_hat >= 5 and n * (1 - p_hat) >= 5:  # Normal approximation valid
                    z_score = 1.96  # 95% confidence level
                    margin_of_error = z_score * math.sqrt((p_hat * (1 - p_hat)) / n)
                    
                    confidence_intervals[f"{result.variant_name}_conversion_rate"] = {
                        "lower": max(0.0, p_hat - margin_of_error),
                        "upper": min(1.0, p_hat + margin_of_error),
                        "confidence_level": 0.95
                    }
        
        return confidence_intervals

    async def _get_experiment_interactions(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get detailed interaction data for an experiment"""
        # This would typically query your database
        # For now, return empty list as placeholder
        return []

    async def _get_experiment(self, experiment_id: str) -> Optional[ABTestExperiment]:
        """Get experiment by ID"""
        try:
            if experiment_id in self.experiments:
                return self.experiments[experiment_id]
            return None
        except Exception as e:
            print(f"Error getting experiment: {e}")
            return None
    
    async def _get_experiment_results(self, experiment_id: str) -> List[ABTestResult]:
        """Get experiment results by ID"""
        try:
            if experiment_id in self.results:
                return list(self.results[experiment_id].values())
            return []
        except Exception as e:
            print(f"Error getting experiment results: {e}")
            return []

    async def _calculate_statistical_power(self, experiment_id: str) -> float:
        """Calculate statistical power for an experiment"""
        try:
            experiment = await self._get_experiment(experiment_id)
            if not experiment:
                return 0.0
            
            results = await self._get_experiment_results(experiment_id)
            if len(results) < 2:
                return 0.0
            
            # Calculate effect size (Cohen's d for conversion rates)
            control = next((r for r in results if r.is_control), results[0])
            variants = [r for r in results if not r.is_control]
            
            if not variants:
                return 0.0
            
            # Calculate pooled standard deviation
            total_impressions = sum(r.impressions for r in results)
            if total_impressions == 0:
                return 0.0
            
            # For binary outcomes (conversions), use arcsin transformation
            control_rate = control.conversion_rate
            variant_rate = variants[0].conversion_rate
            
            # Arcsin transformation for proportions
            control_arcsin = 2 * math.asin(math.sqrt(control_rate))
            variant_arcsin = 2 * math.asin(math.sqrt(variant_rate))
            
            effect_size = abs(variant_arcsin - control_arcsin)
            
            # Calculate power using normal approximation
            alpha = 1 - experiment.confidence_level
            z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed test
            
            # Sample size per group (assuming equal allocation)
            n_per_group = total_impressions / len(results)
            
            # Power calculation using normal approximation
            z_beta = (effect_size * math.sqrt(n_per_group/2)) - z_alpha
            power = stats.norm.cdf(z_beta)
            
            return min(1.0, max(0.0, power))
            
        except Exception as e:
            print(f"Error calculating statistical power: {e}")
            return 0.0

    def _calculate_improvement(self, control: ABTestResult, variant: ABTestResult, metric: str) -> float:
        """Calculate improvement percentage for a variant over control"""
        try:
            control_value = getattr(control, metric, 0)
            variant_value = getattr(variant, metric, 0)
            
            if control_value == 0:
                return 0.0 if variant_value == 0 else float('inf')
            
            improvement = (variant_value - control_value) / control_value
            return improvement
            
        except Exception as e:
            print(f"Error calculating improvement: {e}")
            return 0.0


@dataclass
class LearningParameter:
    """Parameter for adaptive learning system"""
    name: str
    current_value: float
    min_value: float
    max_value: float
    learning_rate: float
    momentum: float = 0.9
    history: List[float] = field(default_factory=list)
    
    def update(self, gradient: float, performance_score: float) -> None:
        """Update parameter value using gradient descent with momentum"""
        # Store current value in history
        self.history.append(self.current_value)
        
        # Calculate learning rate adjustment based on performance
        adjusted_lr = self.learning_rate * (1 + performance_score * 0.1)
        
        # Update with momentum
        update = adjusted_lr * gradient + self.momentum * (self.history[-1] - self.history[-2] if len(self.history) > 1 else 0)
        
        # Apply update
        new_value = self.current_value + update
        
        # Clamp to bounds
        self.current_value = max(self.min_value, min(self.max_value, new_value))
    
    def get_trend(self) -> str:
        """Get parameter trend direction"""
        if len(self.history) < 3:
            return "stable"
        
        recent_values = self.history[-3:]
        if recent_values[-1] > recent_values[0]:
            return "increasing"
        elif recent_values[-1] < recent_values[0]:
            return "decreasing"
        else:
            return "stable"


@dataclass
class AdaptiveLearningConfig:
    """Configuration for adaptive learning system"""
    learning_rate: float = 0.01
    momentum: float = 0.9
    performance_threshold: float = 0.7
    adaptation_frequency: int = 100  # Adapt every N interactions
    history_window: int = 1000  # Keep last N performance records
    min_confidence: float = 0.8
    exploration_rate: float = 0.1  # Random exploration probability


class AdaptiveLearningSystem:
    """Adaptive learning system for continuous marketing optimization"""
    
    def __init__(self, config: AdaptiveLearningConfig = None) -> None:
        self.config = config or AdaptiveLearningConfig()
        self.parameters: Dict[str, LearningParameter] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.interaction_count = 0
        self.last_adaptation = 0
        
        # Initialize default parameters
        self._initialize_default_parameters()
    
    def _initialize_default_parameters(self) -> None:
        """Initialize default learning parameters"""
        default_params = {
            "content_creativity": (0.7, 0.3, 1.0),
            "keyword_density": (0.05, 0.01, 0.15),
            "emotional_intensity": (0.6, 0.2, 1.0),
            "urgency_level": (0.5, 0.1, 0.9),
            "social_proof_weight": (0.4, 0.1, 0.8),
            "price_sensitivity": (0.6, 0.3, 0.9),
            "audience_targeting": (0.7, 0.4, 1.0),
            "platform_optimization": (0.5, 0.2, 0.8)
        }
        
        for name, (current, min_val, max_val) in default_params.items():
            self.parameters[name] = LearningParameter(
                name=name,
                current_value=current,
                min_value=min_val,
                max_value=max_val,
                learning_rate=self.config.learning_rate,
                momentum=self.config.momentum
            )
    
    async def record_performance(self, performance_data: Dict[str, Any]) -> None:
        """Record performance data for learning"""
        try:
            # Store performance data
            self.performance_history.append({
                "timestamp": datetime.utcnow(),
                "performance_score": performance_data.get("performance_score", 0.0),
                "metrics": performance_data.get("metrics", {}),
                "parameters": {name: param.current_value for name, param in self.parameters.items()}
            })
            
            # Keep only recent history
            if len(self.performance_history) > self.config.history_window:
                self.performance_history.pop(0)
            
            self.interaction_count += 1
            
            # Check if adaptation is needed
            if self.interaction_count - self.last_adaptation >= self.config.adaptation_frequency:
                await self._adapt_parameters()
                self.last_adaptation = self.interaction_count
                
        except Exception as e:
            print(f"Error recording performance: {e}")
    
    async def _adapt_parameters(self) -> None:
        """Adapt parameters based on performance history"""
        try:
            if len(self.performance_history) < 10:
                return  # Need sufficient data
            
            # Calculate performance trends
            recent_performance = [p["performance_score"] for p in self.performance_history[-10:]]
            performance_trend = self._calculate_trend(recent_performance)
            
            # Calculate parameter gradients
            gradients = self._calculate_parameter_gradients()
            
            # Update parameters
            for param_name, param in self.parameters.items():
                if param_name in gradients:
                    gradient = gradients[param_name]
                    avg_performance = sum(recent_performance) / len(recent_performance)
                    
                    # Apply gradient with performance-based adjustment
                    param.update(gradient, avg_performance)
                    
                    # Add exploration noise
                    if random.random() < self.config.exploration_rate:
                        exploration_noise = random.uniform(-0.1, 0.1)
                        param.current_value = max(param.min_value, 
                                               min(param.max_value, 
                                                   param.current_value + exploration_noise))
            
        except Exception as e:
            print(f"Error adapting parameters: {e}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction and strength"""
        try:
            if len(values) < 2:
                return 0.0
            
            # Simple linear regression slope
            x = list(range(len(values)))
            y = values
            
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            return slope
            
        except Exception as e:
            print(f"Error calculating trend: {e}")
            return 0.0
    
    def _calculate_parameter_gradients(self) -> Dict[str, float]:
        """Calculate gradients for each parameter based on performance correlation"""
        try:
            gradients = {}
            
            for param_name in self.parameters.keys():
                # Get parameter values and performance scores
                param_values = [p["parameters"][param_name] for p in self.performance_history]
                performance_scores = [p["performance_score"] for p in self.performance_history]
                
                if len(param_values) < 5:
                    gradients[param_name] = 0.0
                    continue
                
                # Calculate correlation coefficient
                correlation = self._calculate_correlation(param_values, performance_scores)
                
                # Convert correlation to gradient
                # Positive correlation means increasing parameter improves performance
                gradients[param_name] = correlation * 0.1  # Scale factor
                
            return gradients
            
        except Exception as e:
            print(f"Error calculating parameter gradients: {e}")
            return {}
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        try:
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            sum_y2 = sum(y[i] ** 2 for i in range(n))
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            return 0.0
    
    def get_optimized_parameters(self) -> Dict[str, float]:
        """Get current optimized parameter values"""
        return {name: param.current_value for name, param in self.parameters.items()}
    
    def get_parameter_trends(self) -> Dict[str, str]:
        """Get trend direction for each parameter"""
        return {name: param.get_trend() for name, param in self.parameters.items()}
    
    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning system summary"""
        try:
            summary = {
                "total_interactions": self.interaction_count,
                "last_adaptation": self.last_adaptation,
                "performance_history_size": len(self.performance_history),
                "current_parameters": self.get_optimized_parameters(),
                "parameter_trends": self.get_parameter_trends(),
                "recent_performance": []
            }
            
            # Add recent performance data
            for record in self.performance_history[-10:]:
                summary["recent_performance"].append({
                    "timestamp": record["timestamp"].isoformat(),
                    "performance_score": record["performance_score"]
                })
            
            return summary
            
        except Exception as e:
            print(f"Error getting learning summary: {e}")
            return {}
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for parameter optimization"""
        try:
            if not self.performance_history:
                return []
            
            recommendations = []
            
            # Analyze performance trends
            recent_performance = self.performance_history[-min(10, len(self.performance_history)):]
            avg_performance = sum(p["performance_score"] for p in recent_performance) / len(recent_performance)
            
            # Check if performance is declining
            if len(self.performance_history) >= 5:
                recent_avg = sum(p["performance_score"] for p in recent_performance) / len(recent_performance)
                older_avg = sum(p["performance_score"] for p in self.performance_history[:-5]) / 5
                
                if recent_avg < older_avg * 0.9:  # 10% decline
                    recommendations.append({
                        "type": "performance_decline",
                        "priority": "high",
                        "description": "Performance has declined by 10% or more",
                        "suggested_actions": [
                            "Review recent parameter changes",
                            "Check for external factors affecting performance",
                            "Consider reverting to previous parameters"
                        ]
                    })
            
            # Check parameter effectiveness
            for param_name, param in self.parameters.items():
                if len(param.history) >= 3:
                    recent_values = param.history[-3:]
                    # Get corresponding performance scores
                    recent_performance = []
                    for record in self.performance_history[-3:]:
                        if param_name in record.get("parameters", {}):
                            recent_performance.append(record["performance_score"])
                    
                    if recent_performance:
                        avg_performance_with_change = sum(recent_performance) / len(recent_performance)
                        
                        if avg_performance_with_change < avg_performance * 0.95:
                            recommendations.append({
                                "type": "parameter_ineffective",
                                "priority": "medium",
                                "parameter": param_name,
                                "description": f"Parameter {param_name} changes may be reducing performance",
                                "suggested_actions": [
                                    f"Review effectiveness of {param_name} changes",
                                    "Consider reverting to previous value",
                                    "Test smaller parameter adjustments"
                                ]
                            })
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting optimization recommendations: {e}")
            return []
    
    async def get_performance_insights(self) -> Dict[str, Any]:
        """Get comprehensive performance insights and analysis"""
        try:
            if not self.performance_history:
                return {"error": "No performance data available"}
            
            insights = {
                "overall_performance": {},
                "trends": {},
                "parameter_analysis": {},
                "recommendations": []
            }
            
            # Overall performance metrics
            all_scores = [p["performance_score"] for p in self.performance_history]
            insights["overall_performance"] = {
                "total_records": len(self.performance_history),
                "average_score": sum(all_scores) / len(all_scores),
                "best_score": max(all_scores),
                "worst_score": min(all_scores),
                "score_variance": self._calculate_variance(all_scores),
                "recent_trend": self._calculate_trend_direction(all_scores[-min(10, len(all_scores)):])
            }
            
            # Performance trends over time
            if len(self.performance_history) >= 10:
                recent_scores = all_scores[-10:]
                insights["trends"] = {
                    "short_term_trend": self._calculate_trend_direction(recent_scores),
                    "long_term_trend": self._calculate_trend_direction(all_scores),
                    "seasonality_detected": self._detect_seasonality(all_scores),
                    "performance_stability": self._calculate_stability(all_scores)
                }
            
            # Parameter effectiveness analysis
            for param_name, param in self.parameters.items():
                if len(param.history) >= 3:
                    param_insights = {
                        "total_changes": len(param.history),
                        "value_range": {
                            "min": min(param.history),
                            "max": max(param.history)
                        },
                        "performance_correlation": self._calculate_parameter_correlation(param_name),
                        "optimal_value": self._find_optimal_parameter_value(param_name)
                    }
                    insights["parameter_analysis"][param_name] = param_insights
            
            # Generate recommendations
            insights["recommendations"] = await self.get_optimization_recommendations()
            
            return insights
            
        except Exception as e:
            print(f"Error getting performance insights: {e}")
            return {"error": str(e)}
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        try:
            if len(values) < 2:
                return 0.0
            
            mean = sum(values) / len(values)
            squared_diff_sum = sum((x - mean) ** 2 for x in values)
            return squared_diff_sum / (len(values) - 1)
            
        except Exception as e:
            print(f"Error calculating variance: {e}")
            return 0.0
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        try:
            if len(values) < 2:
                return "insufficient_data"
            
            # Simple linear trend calculation
            x_values = list(range(len(values)))
            y_values = values
            
            n = len(values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            # Calculate slope
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            if abs(slope) < 0.01:
                return "stable"
            elif slope > 0:
                return "increasing"
            else:
                return "decreasing"
                
        except Exception as e:
            print(f"Error calculating trend direction: {e}")
            return "unknown"
    
    def _detect_seasonality(self, values: List[float]) -> bool:
        """Detect if there's seasonality in the performance data"""
        try:
            if len(values) < 12:  # Need at least 12 data points for seasonality detection
                return False
            
            # Simple seasonality detection using autocorrelation
            # This is a simplified approach - in production you might use more sophisticated methods
            
            # Check if there are repeating patterns
            half_length = len(values) // 2
            first_half = values[:half_length]
            second_half = values[half_length:2*half_length]
            
            # Calculate correlation between halves
            if len(first_half) == len(second_half) and len(first_half) > 0:
                correlation = self._calculate_correlation(first_half, second_half)
                return correlation > 0.3  # Threshold for seasonality
            
            return False
            
        except Exception as e:
            print(f"Error detecting seasonality: {e}")
            return False
    
    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability score (0-1, higher is more stable)"""
        try:
            if len(values) < 2:
                return 1.0
            
            # Calculate coefficient of variation (CV = std_dev / mean)
            mean = sum(values) / len(values)
            if mean == 0:
                return 1.0
            
            variance = self._calculate_variance(values)
            std_dev = variance ** 0.5
            cv = std_dev / mean
            
            # Convert CV to stability score (0-1)
            # Lower CV means higher stability
            stability = max(0, 1 - min(cv, 1))
            return stability
            
        except Exception as e:
            print(f"Error calculating stability: {e}")
            return 0.0
    
    def _calculate_parameter_correlation(self, param_name: str) -> float:
        """Calculate correlation between parameter changes and performance"""
        try:
            if param_name not in self.parameters:
                return 0.0
            
            param = self.parameters[param_name]
            if len(param.history) < 3:
                return 0.0
            
            # Get performance scores corresponding to parameter changes
            param_values = param.history
            performance_scores = []
            
            # Find performance scores for each parameter value
            for record in self.performance_history:
                if param_name in record.get("parameters", {}):
                    performance_scores.append(record["performance_score"])
            
            if len(performance_scores) < 3:
                return 0.0
            
            # Use the same number of values for correlation
            min_length = min(len(param_values), len(performance_scores))
            param_values = param_values[-min_length:]
            performance_scores = performance_scores[-min_length:]
            
            return self._calculate_correlation(param_values, performance_scores)
            
        except Exception as e:
            print(f"Error calculating parameter correlation: {e}")
            return 0.0
    
    def _find_optimal_parameter_value(self, param_name: str) -> Optional[float]:
        """Find the parameter value that produced the best performance"""
        try:
            if param_name not in self.parameters:
                return None
            
            param = self.parameters[param_name]
            if len(param.history) < 3:
                return None
            
            # Find the parameter value associated with the best performance
            # This is a simplified approach - in production you might use more sophisticated methods
            best_performance = 0.0
            best_param_value = None
            
            for record in self.performance_history:
                if param_name in record.get("parameters", {}):
                    if record["performance_score"] > best_performance:
                        best_performance = record["performance_score"]
                        best_param_value = record["parameters"][param_name]
            
            return best_param_value
            
        except Exception as e:
            print(f"Error finding optimal parameter value: {e}")
            return None
    
    async def reset_parameters(self) -> None:
        """Reset all parameters to their default values"""
        try:
            self._initialize_default_parameters()
            self.performance_history.clear()
            self.interaction_count = 0
            self.last_adaptation = 0
            print("All parameters have been reset to default values")
        except Exception as e:
            print(f"Error resetting parameters: {e}")
    
    async def export_learning_data(self) -> Dict[str, Any]:
        """Export learning system data for analysis"""
        try:
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "config": {
                    "learning_rate": self.config.learning_rate,
                    "momentum": self.config.momentum,
                    "performance_threshold": self.config.performance_threshold,
                    "adaptation_frequency": self.config.adaptation_frequency,
                    "history_window": self.config.history_window,
                    "min_confidence": self.config.min_confidence,
                    "exploration_rate": self.config.exploration_rate
                },
                "parameters": {},
                "performance_history": self.performance_history,
                "statistics": {
                    "total_interactions": self.interaction_count,
                    "total_adaptations": self.last_adaptation // self.config.adaptation_frequency,
                    "performance_history_size": len(self.performance_history)
                }
            }
            
            # Export parameter data
            for param_name, param in self.parameters.items():
                export_data["parameters"][param_name] = {
                    "current_value": param.current_value,
                    "min_value": param.min_value,
                    "max_value": param.max_value,
                    "learning_rate": param.learning_rate,
                    "momentum": param.momentum,
                    "history": param.history,
                    "trend": param.get_trend()
                }
            
            return export_data
            
        except Exception as e:
            print(f"Error exporting learning data: {e}")
            return {"error": str(e)}
    
    async def import_learning_data(self, import_data: Dict[str, Any]) -> bool:
        """Import learning data from external source"""
        try:
            if 'parameters' not in import_data or 'performance_history' not in import_data:
                return False
            
            # Import parameters
            for param_name, param_data in import_data['parameters'].items():
                if param_name in self.parameters:
                    param = self.parameters[param_name]
                    param.current_value = param_data.get('current_value', param.current_value)
                    param.history = param_data.get('history', param.history)
                    param.learning_rate = param_data.get('learning_rate', param.learning_rate)
                    param.momentum = param_data.get('momentum', param.momentum)
            
            # Import performance history
            self.performance_history = import_data.get('performance_history', [])
            
            # Import configuration
            if 'config' in import_data:
                config_data = import_data['config']
                self.config.learning_rate = config_data.get('learning_rate', self.config.learning_rate)
                self.config.momentum = config_data.get('momentum', self.config.momentum)
                self.config.performance_threshold = config_data.get('performance_threshold', self.config.performance_threshold)
            
            return True
            
        except Exception as e:
            print(f"Error importing learning data: {e}")
            return False

    async def optimize_hyperparameters(self) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search or Bayesian optimization"""
        try:
            optimization_results = {}
            
            # Grid search for learning rate
            learning_rates = [0.001, 0.01, 0.1, 0.5]
            best_lr = self.config.learning_rate
            best_performance = 0.0
            
            for lr in learning_rates:
                # Temporarily set learning rate
                original_lr = self.config.learning_rate
                self.config.learning_rate = lr
                
                # Simulate performance with this learning rate
                simulated_performance = await self._simulate_parameter_performance()
                
                if simulated_performance > best_performance:
                    best_performance = simulated_performance
                    best_lr = lr
                
                # Restore original learning rate
                self.config.learning_rate = original_lr
            
            optimization_results['learning_rate'] = {
                'current': self.config.learning_rate,
                'optimized': best_lr,
                'improvement': (best_performance - self._calculate_average_performance()) / self._calculate_average_performance() if self._calculate_average_performance() > 0 else 0
            }
            
            # Optimize momentum
            momentums = [0.5, 0.7, 0.9, 0.95]
            best_momentum = self.config.momentum
            best_performance = 0.0
            
            for momentum in momentums:
                original_momentum = self.config.momentum
                self.config.momentum = momentum
                
                simulated_performance = await self._simulate_parameter_performance()
                
                if simulated_performance > best_performance:
                    best_performance = simulated_performance
                    best_momentum = momentum
                
                self.config.momentum = original_momentum
            
            optimization_results['momentum'] = {
                'current': self.config.momentum,
                'optimized': best_momentum,
                'improvement': (best_performance - self._calculate_average_performance()) / self._calculate_average_performance() if self._calculate_average_performance() > 0 else 0
            }
            
            # Apply optimizations if significant improvement
            if optimization_results['learning_rate']['improvement'] > 0.1:
                self.config.learning_rate = best_lr
            
            if optimization_results['momentum']['improvement'] > 0.1:
                self.config.momentum = best_momentum
            
            return optimization_results
            
        except Exception as e:
            print(f"Error optimizing hyperparameters: {e}")
            return {}

    async def _simulate_parameter_performance(self) -> float:
        """Simulate performance with current parameters"""
        try:
            if len(self.performance_history) < 10:
                return 0.5
            
            # Use recent performance as baseline
            recent_performance = self.performance_history[-10:]
            baseline = statistics.mean(recent_performance)
            
            # Simulate improvement based on parameter settings
            lr_factor = 1.0 + (self.config.learning_rate * 10)  # Higher learning rate = faster adaptation
            momentum_factor = 1.0 + (self.config.momentum * 0.5)  # Higher momentum = more stable
            
            simulated_performance = baseline * lr_factor * momentum_factor
            
            return min(1.0, max(0.0, simulated_performance))
            
        except Exception as e:
            print(f"Error simulating parameter performance: {e}")
            return 0.5

    def _calculate_average_performance(self) -> float:
        """Calculate average performance from history"""
        try:
            if not self.performance_history:
                return 0.0
            
            return statistics.mean(self.performance_history)
        except Exception as e:
            print(f"Error calculating average performance: {e}")
            return 0.0

    async def get_parameter_importance(self) -> Dict[str, float]:
        """Calculate importance scores for each parameter"""
        try:
            importance_scores = {}
            
            for param_name, param in self.parameters.items():
                if len(param.history) < 3:
                    importance_scores[param_name] = 0.5
                    continue
                
                # Calculate correlation between parameter changes and performance
                param_changes = []
                performance_changes = []
                
                for i in range(1, len(param.history)):
                    param_change = param.history[i] - param.history[i-1]
                    param_changes.append(param_change)
                    
                    if i < len(self.performance_history):
                        perf_change = self.performance_history[i] - self.performance_history[i-1]
                        performance_changes.append(perf_change)
                
                if len(param_changes) > 1 and len(performance_changes) > 1:
                    try:
                        correlation = self._calculate_correlation(param_changes, performance_changes)
                        importance_scores[param_name] = abs(correlation)
                    except Exception:
                        importance_scores[param_name] = 0.5
                else:
                    importance_scores[param_name] = 0.5
            
            return importance_scores
            
        except Exception as e:
            print(f"Error calculating parameter importance: {e}")
            return {}

    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive learning insights and recommendations"""
        try:
            insights = {
                "performance_summary": await self.get_learning_summary(),
                "parameter_importance": await self.get_parameter_importance(),
                "optimization_recommendations": await self.get_optimization_recommendations(),
                "performance_insights": await self.get_performance_insights(),
                "hyperparameter_optimization": await self.optimize_hyperparameters()
            }
            
            # Add overall system health score
            health_score = self._calculate_system_health_score()
            insights["system_health"] = {
                "overall_score": health_score,
                "status": "healthy" if health_score > 0.7 else "needs_attention" if health_score > 0.4 else "critical",
                "recommendations": self._generate_health_recommendations(health_score)
            }
            
            return insights
            
        except Exception as e:
            print(f"Error getting learning insights: {e}")
            return {}

    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score"""
        try:
            if not self.performance_history:
                return 0.0
            
            # Performance stability
            recent_performance = self.performance_history[-20:] if len(self.performance_history) >= 20 else self.performance_history
            performance_stability = 1.0 - min(1.0, statistics.stdev(recent_performance) * 5)
            
            # Parameter stability
            parameter_stability = 0.0
            total_params = len(self.parameters)
            
            for param in self.parameters.values():
                if len(param.history) >= 3:
                    param_variance = statistics.variance(param.history[-10:]) if len(param.history) >= 10 else statistics.variance(param.history)
                    param_stability = max(0.0, 1.0 - (param_variance * 10))
                    parameter_stability += param_stability
            
            parameter_stability = parameter_stability / total_params if total_params > 0 else 0.0
            
            # Learning effectiveness
            if len(self.performance_history) >= 10:
                recent_avg = statistics.mean(self.performance_history[-10:])
                older_avg = statistics.mean(self.performance_history[-20:-10]) if len(self.performance_history) >= 20 else statistics.mean(self.performance_history[:10])
                learning_effectiveness = max(0.0, min(1.0, (recent_avg - older_avg) * 5 + 0.5))
            else:
                learning_effectiveness = 0.5
            
            # Weighted average
            health_score = (
                performance_stability * 0.4 +
                parameter_stability * 0.3 +
                learning_effectiveness * 0.3
            )
            
            return min(1.0, max(0.0, health_score))
            
        except Exception as e:
            print(f"Error calculating system health score: {e}")
            return 0.0

    def _generate_health_recommendations(self, health_score: float) -> List[str]:
        """Generate health-based recommendations"""
        recommendations = []
        
        if health_score < 0.3:
            recommendations.extend([
                "System performance is critically low. Consider resetting parameters.",
                "Check for data quality issues in performance metrics.",
                "Review learning rate and momentum settings."
            ])
        elif health_score < 0.6:
            recommendations.extend([
                "System needs attention. Monitor parameter stability.",
                "Consider reducing learning rate for more stable performance.",
                "Review recent performance trends for anomalies."
            ])
        elif health_score < 0.8:
            recommendations.extend([
                "System is performing well but could be optimized.",
                "Consider fine-tuning hyperparameters for better performance.",
                "Monitor for performance degradation patterns."
            ])
        else:
            recommendations.extend([
                "System is healthy and performing well.",
                "Continue monitoring for optimal performance.",
                "Consider exploring more aggressive optimization strategies."
            ])
        
        return recommendations

    async def get_experiment_analytics(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for an experiment"""
        try:
            experiment = await self._get_experiment(experiment_id)
            if not experiment:
                return {}
            
            results = await self._get_experiment_results(experiment_id)
            if not results:
                return {}
            
            analytics = {
                "experiment_summary": await self.get_experiment_summary(experiment_id),
                "statistical_analysis": {
                    "power_analysis": await self._calculate_statistical_power(experiment_id),
                    "significance_tests": self._run_significance_tests(results),
                    "confidence_intervals": self._calculate_confidence_intervals(results)
                },
                "performance_metrics": self._calculate_performance_metrics(results),
                "trend_analysis": self._analyze_performance_trends(results),
                "recommendations": await self.get_experiment_recommendations(experiment_id),
                "health_score": await self.get_experiment_health_score(experiment_id)
            }
            
            return analytics
            
        except Exception as e:
            print(f"Error getting experiment analytics: {e}")
            return {}

    def _calculate_performance_metrics(self, results: List[ABTestResult]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            if not results:
                return {}
            
            metrics = {}
            
            # Overall performance
            total_impressions = sum(r.impressions for r in results)
            total_clicks = sum(r.clicks for r in results)
            total_conversions = sum(r.conversions for r in results)
            total_revenue = sum(r.revenue for r in results)
            
            metrics["overall"] = {
                "total_impressions": total_impressions,
                "total_clicks": total_clicks,
                "total_conversions": total_conversions,
                "total_revenue": total_revenue,
                "overall_ctr": total_clicks / total_impressions if total_impressions > 0 else 0,
                "overall_conversion_rate": total_conversions / total_impressions if total_impressions > 0 else 0,
                "overall_cpc": total_revenue / total_clicks if total_clicks > 0 else 0,
                "overall_cpa": total_revenue / total_conversions if total_conversions > 0 else 0
            }
            
            # Per-variant metrics
            metrics["variants"] = {}
            for result in results:
                metrics["variants"][result.variant_name] = {
                    "impressions": result.impressions,
                    "clicks": result.clicks,
                    "conversions": result.conversions,
                    "revenue": result.revenue,
                    "ctr": result.ctr,
                    "conversion_rate": result.conversion_rate,
                    "cpc": result.cpc,
                    "cpa": result.cpa,
                    "roi": result.roi,
                    "engagement_rate": result.engagement_rate
                }
            
            # Performance rankings
            if len(results) > 1:
                ctr_ranking = sorted(results, key=lambda r: r.ctr, reverse=True)
                conversion_ranking = sorted(results, key=lambda r: r.conversion_rate, reverse=True)
                revenue_ranking = sorted(results, key=lambda r: r.revenue, reverse=True)
                
                metrics["rankings"] = {
                    "ctr": [r.variant_name for r in ctr_ranking],
                    "conversion_rate": [r.variant_name for r in conversion_ranking],
                    "revenue": [r.variant_name for r in revenue_ranking]
                }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return {}

    def _analyze_performance_trends(self, results: List[ABTestResult]) -> Dict[str, Any]:
        """Analyze performance trends and patterns"""
        try:
            if not results:
                return {}
            
            trends = {}
            
            # Calculate performance ratios between variants
            control = next((r for r in results if r.is_control), results[0])
            variants = [r for r in results if not r.is_control]
            
            for variant in variants:
                variant_name = variant.variant_name
                trends[variant_name] = {}
                
                # CTR improvement
                if control.ctr > 0:
                    ctr_improvement = (variant.ctr - control.ctr) / control.ctr
                    trends[variant_name]["ctr_improvement"] = ctr_improvement
                    trends[variant_name]["ctr_performance"] = "better" if ctr_improvement > 0.05 else "worse" if ctr_improvement < -0.05 else "similar"
                
                # Conversion rate improvement
                if control.conversion_rate > 0:
                    conv_improvement = (variant.conversion_rate - control.conversion_rate) / control.conversion_rate
                    trends[variant_name]["conversion_improvement"] = conv_improvement
                    trends[variant_name]["conversion_performance"] = "better" if conv_improvement > 0.05 else "worse" if conv_improvement < -0.05 else "similar"
                
                # Revenue improvement
                if control.revenue > 0:
                    revenue_improvement = (variant.revenue - control.revenue) / control.revenue
                    trends[variant_name]["revenue_improvement"] = revenue_improvement
                    trends[variant_name]["revenue_performance"] = "better" if revenue_improvement > 0.05 else "worse" if revenue_improvement < -0.05 else "similar"
            
            return trends
            
        except Exception as e:
            print(f"Error analyzing performance trends: {e}")
            return {}

    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview including all experiments and learning status"""
        try:
            overview = {
                "ab_testing": {
                    "total_experiments": len(self.experiments),
                    "active_experiments": len([e for e in self.experiments.values() if e.status == "active"]),
                    "completed_experiments": len([e for e in self.experiments.values() if e.status == "completed"]),
                    "paused_experiments": len([e for e in self.experiments.values() if e.status == "paused"]),
                    "recent_experiments": await self._get_recent_experiments(5)
                },
                "adaptive_learning": {
                    "total_parameters": len(self.parameters),
                    "total_interactions": self.interaction_count,
                    "total_adaptations": self.adaptation_count,
                    "system_health": self._calculate_system_health_score(),
                    "learning_summary": await self.get_learning_summary()
                },
                "performance_metrics": {
                    "average_experiment_health": await self._calculate_average_experiment_health(),
                    "learning_effectiveness": await self._calculate_learning_effectiveness(),
                    "system_efficiency": await self._calculate_system_efficiency()
                }
            }
            
            return overview
            
        except Exception as e:
            print(f"Error getting system overview: {e}")
            return {}

    async def _get_recent_experiments(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent experiments with basic info"""
        try:
            sorted_experiments = sorted(
                self.experiments.values(),
                key=lambda e: e.start_date,
                reverse=True
            )[:limit]
            
            recent = []
            for exp in sorted_experiments:
                recent.append({
                    "experiment_id": exp.experiment_id,
                    "experiment_name": exp.experiment_name,
                    "status": exp.status,
                    "start_date": exp.start_date.isoformat(),
                    "primary_metric": exp.primary_metric
                })
            
            return recent
            
        except Exception as e:
            print(f"Error getting recent experiments: {e}")
            return []

    async def _calculate_average_experiment_health(self) -> float:
        """Calculate average health score across all experiments"""
        try:
            if not self.experiments:
                return 0.0
            
            health_scores = []
            for exp_id in self.experiments.keys():
                health_score = await self.get_experiment_health_score(exp_id)
                health_scores.append(health_score)
            
            return statistics.mean(health_scores) if health_scores else 0.0
            
        except Exception as e:
            print(f"Error calculating average experiment health: {e}")
            return 0.0

    async def _calculate_learning_effectiveness(self) -> float:
        """Calculate learning system effectiveness"""
        try:
            if len(self.performance_history) < 10:
                return 0.5
            
            # Calculate improvement over time
            recent_performance = self.performance_history[-10:]
            older_performance = self.performance_history[-20:-10] if len(self.performance_history) >= 20 else self.performance_history[:10]
            
            recent_avg = statistics.mean(recent_performance)
            older_avg = statistics.mean(older_performance)
            
            if older_avg == 0:
                return 0.5
            
            improvement_ratio = (recent_avg - older_avg) / older_avg
            effectiveness = max(0.0, min(1.0, improvement_ratio * 2 + 0.5))
            
            return effectiveness
            
        except Exception as e:
            print(f"Error calculating learning effectiveness: {e}")
            return 0.5

    async def _calculate_system_efficiency(self) -> float:
        """Calculate overall system efficiency"""
        try:
            # Combine experiment health and learning effectiveness
            experiment_health = await self._calculate_average_experiment_health()
            learning_effectiveness = await self._calculate_learning_effectiveness()
            system_health = self._calculate_system_health_score()
            
            efficiency = (experiment_health * 0.4 + learning_effectiveness * 0.3 + system_health * 0.3)
            
            return min(1.0, max(0.0, efficiency))
            
        except Exception as e:
            print(f"Error calculating system efficiency: {e}")
            return 0.5
