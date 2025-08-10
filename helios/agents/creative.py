from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CreativeBatch:
    design_batch: list[dict[str, Any]]
    batch_confidence: float
    estimated_performance: str


async def generate_designs(selected_products: list[dict[str, Any]], persona: dict[str, Any], dry_run: bool = False) -> CreativeBatch:
    if dry_run:
        designs = []
        for prod in selected_products:
            designs.append(
                {
                    "product_id": prod["printify_product_id"],
                    "designs": [
                        {
                            "concept_name": "viral_statement_1",
                            "image_prompt": "retro minimalist bold text tee, high contrast, clean layout",
                            "viral_potential_score": 8.9,
                            "audience_alignment": 9.2,
                        },
                        {
                            "concept_name": "viral_statement_2",
                            "image_prompt": "retro badge style, limited palette, thick outlines",
                            "viral_potential_score": 8.6,
                            "audience_alignment": 8.8,
                        },
                        {
                            "concept_name": "viral_statement_3",
                            "image_prompt": "memetic phrase with retro type, minimalist icon",
                            "viral_potential_score": 8.5,
                            "audience_alignment": 8.7,
                        },
                    ],
                }
            )
        return CreativeBatch(design_batch=designs, batch_confidence=0.89, estimated_performance="high")

    return await generate_designs(selected_products, persona, dry_run=True)
