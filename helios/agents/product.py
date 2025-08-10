from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProductSelection:
    selected_products: list[dict[str, Any]]
    selection_confidence: float


async def select_products(audience: dict[str, Any], dry_run: bool = False) -> ProductSelection:
    if dry_run:
        return ProductSelection(
            selected_products=[
                {
                    "printify_product_id": "123",
                    "historical_success_rate": 0.84,
                    "profit_margin": 47,
                    "audience_fit_score": 9.1,
                    "fulfillment_speed": "2-3_days",
                },
                {
                    "printify_product_id": "456",
                    "historical_success_rate": 0.79,
                    "profit_margin": 42,
                    "audience_fit_score": 8.7,
                    "fulfillment_speed": "2-3_days",
                },
            ],
            selection_confidence=0.91,
        )

    return await select_products(audience, dry_run=True)
