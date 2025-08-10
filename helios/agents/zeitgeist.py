from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from ..mcp_client import MCPClient


@dataclass
class TrendData:
    trend_name: str
    keywords: list[str]
    opportunity_score: float
    velocity: str
    urgency_level: str
    commercial_indicators: dict[str, Any]
    timing_analysis: dict[str, Any]
    confidence_level: float


async def discover_trend(mcp: Optional[MCPClient], timeout_s: float = 30.0, dry_run: bool = False) -> TrendData:
    if dry_run or not mcp:
        # Return deterministic mock for fast testing
        logger.info("Zeitgeist(dry-run): returning mock trend")
        return TrendData(
            trend_name="retro_minimalist_memes",
            keywords=["retro", "minimalist", "meme", "tee"],
            opportunity_score=8.7,
            velocity="rising",
            urgency_level="immediate",
            commercial_indicators={
                "social_mentions_24h": 15400,
                "search_growth_7d": "+340%",
                "influencer_adoption": "early_mainstream",
                "product_categories": ["apparel", "accessories"],
            },
            timing_analysis={
                "predicted_peak": "14-21_days",
                "saturation_risk": "medium",
                "entry_window": "7_days_optimal",
            },
            confidence_level=0.87,
        )

    payload = {
        "timeout_s": timeout_s,
        "sources": ["twitter", "tiktok", "reddit", "google_trends"],
        "strategy": "velocity_correlation",
    }
    data, err = await mcp.try_call("zeitgeist_trend_finder", payload, timeout_s=timeout_s)
    if err or not data:
        logger.warning("Zeitgeist: MCP unavailable, using fallback mock")
        return await discover_trend(None, timeout_s=timeout_s, dry_run=True)

    return TrendData(
        trend_name=data.get("trend_name", "unknown"),
        keywords=data.get("keywords", []),
        opportunity_score=float(data.get("opportunity_score", 0.0)),
        velocity=data.get("velocity", "unknown"),
        urgency_level=data.get("urgency_level", "monitor"),
        commercial_indicators=data.get("commercial_indicators", {}),
        timing_analysis=data.get("timing_analysis", {}),
        confidence_level=float(data.get("confidence_level", 0.0)),
    )
