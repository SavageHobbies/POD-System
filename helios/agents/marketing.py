from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MarketingBatch:
    listings: list[dict[str, Any]]


async def write_listings(design_batch: list[dict[str, Any]], trend_name: str, dry_run: bool = False) -> MarketingBatch:
    if dry_run:
        listings: list[dict[str, Any]] = []
        for item in design_batch:
            product_id = item.get("product_id")
            for design in item.get("designs", []):
                title = f"{trend_name} - {design['concept_name']}"
                listings.append(
                    {
                        "product_id": product_id,
                        "title": title,
                        "description": "On-trend minimalist retro tee. High quality, fast fulfillment.",
                        "tags": ["retro", "minimalist", "viral", "gift"],
                        "price": 24.99,
                    }
                )
        return MarketingBatch(listings=listings)

    return await write_listings(design_batch, trend_name, dry_run=True)
