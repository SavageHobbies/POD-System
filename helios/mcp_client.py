from __future__ import annotations

import anyio
import httpx
from dataclasses import dataclass
from typing import Any, Optional, Dict, List
from loguru import logger
import asyncio
from pathlib import Path


@dataclass
class MCPClient:
    base_url: str
    auth_token: Optional[str] = None
    timeout_s: float = 30.0

    async def call(self, tool: str, payload: dict[str, Any], timeout_s: Optional[float] = None) -> dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("MCP base_url not configured")
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        # Google MCP server uses /execute endpoint with tool in payload
        url = f"{self.base_url.rstrip('/')}/execute"
        mcp_payload = {
            "tool": tool,
            "parameters": payload
        }
        
        to = timeout_s or self.timeout_s
        logger.debug(f"MCP call {tool} -> {url} (timeout={to}s)")
        async with httpx.AsyncClient(timeout=to) as client:
            resp = await client.post(url, json=mcp_payload, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def try_call(self, tool: str, payload: dict[str, Any], timeout_s: Optional[float] = None) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        try:
            data = await self.call(tool, payload, timeout_s)
            return data, None
        except Exception as e:
            logger.warning(f"MCP call failed: {tool}: {e}")
            return None, str(e)

    # Google MCP AI Agent Methods
    async def orchestrator_ai(self, prompt: str) -> dict[str, Any]:
        """CEO/Orchestrator AI for business decisions using Gemini 2.0 Flash"""
        return await self.call("orchestrator_ai", {"prompt": prompt})

    async def trend_seeker(self, seed: str, geo: str = "US") -> dict[str, Any]:
        """Trend analysis using Gemini 2.0 Flash for fast trend detection"""
        return await self.call("trend_seeker", {"seed": seed, "geo": geo})

    async def ethics_ai(self, trend_name: str, keywords: List[str]) -> dict[str, Any]:
        """Ethical screening using Gemini 2.0 Flash for fast ethical analysis"""
        return await self.call("ethics_ai", {"trend_name": trend_name, "keywords": keywords})

    async def marketing_ai(self, product_info: Dict[str, Any]) -> dict[str, Any]:
        """Marketing copy generation using Gemini 2.0 Flash for creative marketing"""
        return await self.call("marketing_ai", {"product_info": product_info})

    async def creative_ai(self, design_brief: str) -> dict[str, Any]:
        """Creative design ideas using Gemini 2.0 Flash for design tasks"""
        return await self.call("creative_ai", {"design_brief": design_brief})

    async def image_generation(self, prompt: str) -> dict[str, Any]:
        """Image generation using Gemini 2.0 Flash for image tasks"""
        return await self.call("image_generation", {"prompt": prompt})

    async def multimodal_ai(self, prompt: str, image_path: Optional[Path] = None) -> dict[str, Any]:
        """Multimodal AI using Gemini 2.0 Flash for text+image processing"""
        payload = {"prompt": prompt}
        if image_path and image_path.exists():
            import base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            payload["image_data"] = image_data
        return await self.call("multimodal_ai", payload)

    # Google Cloud Services
    async def google_trends_keywords(self, geo: str = "US", top_n: int = 10) -> dict[str, Any]:
        """Get Google Trends keywords"""
        return await self.call("google_trends_keywords", {"geo": geo, "top_n": top_n})

    async def social_media_scanner(self, seed: str) -> dict[str, Any]:
        """Scan social media for trending keywords"""
        return await self.call("social_media_scanner", {"seed": seed})

    async def google_sheets_operation(self, operation: str, sheet_id: str, worksheet_name: str = "Sheet1", data: Optional[Dict] = None) -> dict[str, Any]:
        """Google Sheets operations (read/write/update)"""
        payload = {
            "operation": operation,
            "sheet_id": sheet_id,
            "worksheet_name": worksheet_name
        }
        if data:
            payload["data"] = data
        return await self.call("google_sheets_operation", payload)

    async def google_drive_operation(self, operation: str, folder_id: str, file_data: Optional[Dict] = None) -> dict[str, Any]:
        """Google Drive operations (list/upload)"""
        payload = {
            "operation": operation,
            "folder_id": folder_id
        }
        if file_data:
            payload["file_data"] = file_data
        return await self.call("google_drive_operation", payload)

    async def audience_analysis(self, analysis_request: Dict[str, Any]) -> dict[str, Any]:
        """Audience analysis using Gemini 1.5 Pro for comprehensive insights"""
        return await self.call("audience_analysis", analysis_request)

    async def vertex_ai_call(self, model: str, prompt: str) -> dict[str, Any]:
        """Vertex AI integration for advanced AI tasks"""
        return await self.call("vertex_ai_call", {"model": model, "prompt": prompt})

    # Batch Processing Methods
    async def process_trends_batch(self, seeds: List[str], geo: str = "US") -> List[dict[str, Any]]:
        """Process multiple trends in parallel using Gemini 2.0 Flash"""
        tasks = [self.trend_seeker(seed, geo) for seed in seeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Trend processing failed for {seeds[i]}: {result}")
                processed_results.append({"error": str(result), "seed": seeds[i]})
            else:
                processed_results.append(result)
        
        return processed_results

    async def analyze_products_batch(self, product_briefs: List[str]) -> List[dict[str, Any]]:
        """Analyze multiple products in parallel using Gemini 2.0 Flash"""
        tasks = [self.creative_ai(brief) for brief in product_briefs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Product analysis failed for brief {i}: {result}")
                processed_results.append({"error": str(result), "brief_index": i})
            else:
                processed_results.append(result)
        
        return processed_results

    @staticmethod
    def from_env(base_url: Optional[str], auth_token: Optional[str]) -> Optional["MCPClient"]:
        if not base_url:
            return None
        return MCPClient(base_url=base_url, auth_token=auth_token)
