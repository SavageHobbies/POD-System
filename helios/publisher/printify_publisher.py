from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class VariantConfig:
    variant_id: int
    price_cents: int
    is_enabled: bool = True


class PrintifyPublisher:
    BASE_URL = "https://api.printify.com/v1"

    def __init__(self, api_token: str, shop_id: str) -> None:
        self.api_token = api_token
        self.shop_id = shop_id
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{path}"
        response = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        return response.json()

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{path}"
        response = requests.get(url, headers=self.headers, params=params, timeout=60)
        response.raise_for_status()
        return response.json()

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def _upload(self, file_path: Path) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/uploads/images.json"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "image/png")}
            response = requests.post(url, headers=headers, files=files, timeout=120)
            response.raise_for_status()
            return response.json()

    def upload_design(self, file_path: Path) -> str:
        payload = self._upload(file_path)
        return payload.get("id")

    def estimate_price(self, catalog_variant_cost_cents: Optional[int], margin: float, fallback_cents: int = 2499) -> int:
        if catalog_variant_cost_cents is None:
            return int(fallback_cents)
        # price = cost * (1 + margin)
        price = int(round(catalog_variant_cost_cents * (1 + margin)))
        # Ensure ends with .99
        if price % 100 < 99:
            price = price - (price % 100) + 99
        return price

    def create_product(
        self,
        title: str,
        description: str,
        blueprint_id: int,
        print_provider_id: int,
        print_area_file_id: str,
        variant_price_cents: int,
        colors: Optional[List[str]] = None,
        sizes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Minimal product body for front print
        payload: Dict[str, Any] = {
            "title": title,
            "description": description,
            "blueprint_id": blueprint_id,
            "print_provider_id": print_provider_id,
            "variants": [],
            "print_areas": [
                {
                    "variant_ids": [],
                    "placeholders": [
                        {
                            "position": "front",
                            "images": [
                                {
                                    "id": print_area_file_id,
                                    "x": 0.5,
                                    "y": 0.35,
                                    "scale": 1.0,
                                    "angle": 0,
                                }
                            ],
                        }
                    ],
                }
            ],
            "options": [],
        }

        # Fetch blueprint/provider variants to map colors/sizes to IDs
        # If this fails, we will let Printify auto-enable defaults
        try:
            catalog = self._get(f"/catalog/blueprints/{blueprint_id}/print_providers/{print_provider_id}.json")
            variant_map: Dict[int, Dict[str, Any]] = {v["id"]: v for v in catalog.get("variants", [])}
            color_opt = next((o for o in catalog.get("options", []) if o.get("type") == "color"), None)
            size_opt = next((o for o in catalog.get("options", []) if o.get("type") == "size"), None)

            wanted_colors = set((colors or []) or [])
            wanted_sizes = set((sizes or []) or [])

            enabled_variant_ids: List[int] = []
            for v in catalog.get("variants", []):
                color_name = v.get("options", {}).get(str(color_opt["id"])) if color_opt else None
                size_name = v.get("options", {}).get(str(size_opt["id"])) if size_opt else None
                if (not wanted_colors or color_name in wanted_colors) and (not wanted_sizes or size_name in wanted_sizes):
                    enabled_variant_ids.append(v["id"])

            # Variants
            for vid in enabled_variant_ids[:]:
                payload["variants"].append({
                    "id": vid,
                    "price": variant_price_cents,
                    "is_enabled": True,
                })

            # Print areas should target enabled variants
            if enabled_variant_ids:
                payload["print_areas"][0]["variant_ids"] = enabled_variant_ids

            # Options for storefront filters
            if color_opt and colors:
                payload["options"].append({"name": color_opt["name"], "type": "color", "values": list(wanted_colors)})
            if size_opt and sizes:
                payload["options"].append({"name": size_opt["name"], "type": "size", "values": list(wanted_sizes)})
        except Exception:
            # Proceed without catalog enrichment
            pass

        product = self._post(f"/shops/{self.shop_id}/products.json", payload)
        return product

    def publish_product(
        self,
        product_id: str,
        publish_to_store: bool = True,
        publish_as_draft: bool = True,
    ) -> Dict[str, Any]:
        # Publish to connected sales channel (e.g., Etsy) via Printify
        payload = {
            "title": True,
            "description": True,
            "images": True,
            "variants": True,
            "tags": True,
            "key_features": True,
            "shipping_template": True,
            "variants_prices": True,
            "send_to_store": publish_to_store,
            "publish": not publish_as_draft,
        }
        return self._post(f"/shops/{self.shop_id}/products/{product_id}/publish.json", payload)
