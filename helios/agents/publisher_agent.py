from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..publisher.printify_publisher import PrintifyPublisher


@dataclass
class PublicationResult:
    product_title: str
    printify_product_id: str | None
    image_upload_id: str | None
    status: str
    final_price_cents: int
    error_details: str | None = None


class PrintifyPublisherAgent:
    def __init__(self, api_token: str, shop_id: str) -> None:
        self.publisher = PrintifyPublisher(api_token=api_token, shop_id=shop_id)

    def run_batch(
        self,
        listings: List[Dict[str, Any]],
        margin: float,
        draft: bool,
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for item in listings:
            try:
                # 1) Upload image
                image_id = self.publisher.upload_design(Path(item["image_path"]))
                time.sleep(2)
                # 2) Create product
                price_cents = int(round(2499 * (1 + margin)))
                if price_cents % 100 < 99:
                    price_cents = price_cents - (price_cents % 100) + 99
                product = self.publisher.create_product(
                    title=item["title"],
                    description=item["description"],
                    blueprint_id=int(item["blueprint_id"]),
                    print_provider_id=int(item["print_provider_id"]),
                    print_area_file_id=image_id,
                    variant_price_cents=price_cents,
                    colors=item.get("colors"),
                    sizes=item.get("sizes"),
                )
                product_id = product.get("id") or product.get("product_id")
                if not product_id:
                    raise RuntimeError(f"No product id returned: {product}")
                time.sleep(2)
                # 3) Publish to Etsy via Printify
                publish = self.publisher.publish_product(product_id=str(product_id), publish_to_store=True, publish_as_draft=draft)
                results.append({
                    "design_id": Path(item["image_path"]).name,
                    "product_title": item["title"],
                    "printify_product_id": str(product_id),
                    "image_upload_id": image_id,
                    "status": "published" if publish else "pending",
                    "sales_channels": ["etsy"],
                    "final_price": price_cents / 100.0,
                    "profit_margin": margin,
                    "publication_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "error_details": None,
                })
            except Exception as e:
                results.append({
                    "design_id": Path(item["image_path"]).name,
                    "product_title": item["title"],
                    "printify_product_id": None,
                    "image_upload_id": None,
                    "status": "failed",
                    "sales_channels": ["etsy"],
                    "final_price": 0,
                    "profit_margin": margin,
                    "publication_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "error_details": str(e),
                })
                continue
        return {
            "publication_results": results,
            "batch_summary": {
                "total_processed": len(results),
                "successful_publications": sum(1 for r in results if r["status"] == "published"),
                "failed_publications": sum(1 for r in results if r["status"] == "failed"),
            },
        }

    async def publish_batch(self, products_to_publish: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Publish a batch of products to Printify"""
        try:
            # Convert to the format expected by run_batch
            listings = []
            for product_data in products_to_publish:
                design = product_data.get("design", {})
                product = product_data.get("product", {})
                marketing = product_data.get("marketing", {})
                
                # Create listing format
                listing = {
                    "image_path": design.get("image_path", ""),
                    "title": design.get("title", product.get("name", "Unknown Product")),
                    "description": marketing.get("product_description", design.get("description", "")),
                    "blueprint_id": product.get("blueprint_id", 1),
                    "print_provider_id": product.get("print_provider_id", 1),
                    "colors": product.get("colors", ["white"]),
                    "sizes": product.get("sizes", ["M"]),
                    "marketing_copy": marketing
                }
                listings.append(listing)
            
            # Use existing run_batch method
            result = self.run_batch(listings, margin=0.5, draft=True)
            
            return result.get("publication_results", [])
            
        except Exception as e:
            print(f"Error in batch publishing: {e}")
            return []

    async def publish_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Publish a single product to Printify"""
        try:
            design = product_data.get("design", {})
            product = product_data.get("product", {})
            marketing = product_data.get("marketing", {})
            
            # Create listing format
            listing = {
                "image_path": design.get("image_path", ""),
                "title": design.get("title", product.get("name", "Unknown Product")),
                "description": marketing.get("product_description", design.get("description", "")),
                "blueprint_id": product.get("blueprint_id", 1),
                "print_provider_id": product.get("print_provider_id", 1),
                "colors": product.get("colors", ["white"]),
                "sizes": product.get("sizes", ["M"]),
                "marketing_copy": marketing
            }
            
            # Use existing run_batch method with single item
            result = self.run_batch([listing], margin=0.5, draft=True)
            
            if result.get("publication_results"):
                return result["publication_results"][0]
            else:
                return {
                    "design_id": design.get("id", "unknown"),
                    "product_title": listing["title"],
                    "printify_product_id": None,
                    "image_upload_id": None,
                    "status": "failed",
                    "sales_channels": ["etsy"],
                    "final_price": 0,
                    "profit_margin": 0.5,
                    "publication_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "error_details": "No result returned from batch processing"
                }
                
        except Exception as e:
            print(f"Error publishing single product: {e}")
            return {
                "design_id": product_data.get("design", {}).get("id", "unknown"),
                "product_title": "Unknown Product",
                "printify_product_id": None,
                "image_upload_id": None,
                "status": "failed",
                "sales_channels": ["etsy"],
                "final_price": 0,
                "profit_margin": 0.5,
                "publication_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "error_details": str(e)
            }
