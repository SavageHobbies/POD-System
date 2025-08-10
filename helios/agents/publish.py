from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..config import HeliosConfig
from ..providers.google_drive import upload_assets
from ..providers.google_sheets import log_row
from ..providers.etsy import EtsyClient


@dataclass
class PublicationPlan:
    publication_queue: list[dict[str, Any]]


async def plan_publication(
    listings: list[dict[str, Any]],
    config: HeliosConfig,
    dry_run: bool = False,
) -> PublicationPlan:
    queue: list[dict[str, Any]] = []

    # Log to Sheets only when live publishing is enabled and not in dry-run
    if (
        not dry_run
        and config.allow_live_publishing
        and config.google_sheets_tracking_id
        and config.service_account_dict()
    ):
        for l in listings[:10]:
            await log_row(
                sheet_id=config.google_sheets_tracking_id,
                creds_dict=config.service_account_dict() or {},
                data={
                    "product_id": l.get("product_id"),
                    "title": l.get("title"),
                    "price": l.get("price"),
                    "status": "simulated" if dry_run else "pending",
                },
            )

    # Upload placeholder asset records to Drive only when live and not dry-run
    if (
        not dry_run
        and config.allow_live_publishing
        and config.google_drive_folder_id
        and config.service_account_dict()
    ):
        await upload_assets(
            files=[f"listing_{i}.txt" for i in range(min(5, len(listings)))],
            folder_id=config.google_drive_folder_id,
            creds_dict=config.service_account_dict() or {},
        )

    for l in listings:
        queue.append(
            {
                "product_id": l.get("product_id"),
                "action": "create_or_update",
                "platforms": ["google_drive", "printify", "etsy_sync"],
                "status": "simulated" if (dry_run or not config.allow_live_publishing) else "pending",
            }
        )

    # Optional: create Etsy draft listings when live
    if (
        not dry_run
        and config.allow_live_publishing
        and config.etsy_api_key
        and config.etsy_shop_id
        and config.etsy_taxonomy_id
    ):
        etsy = EtsyClient(api_key=config.etsy_api_key, oauth_token=config.etsy_oauth_token)
        for l in listings[:3]:  # limit to a few per run to avoid rate issues
            try:
                resp = await etsy.create_draft_listing(
                    shop_id=str(config.etsy_shop_id),
                    title=str(l.get("title", "Listing"))[:139],
                    description=str(l.get("description", ""))[:9999],
                    price=float(l.get("price", 24.99)),
                    taxonomy_id=int(config.etsy_taxonomy_id),
                    shipping_profile_id=int(config.etsy_shipping_profile_id) if config.etsy_shipping_profile_id else None,
                    return_policy_id=int(config.etsy_return_policy_id) if config.etsy_return_policy_id else None,
                    tags=[str(t) for t in (l.get("tags") or [])][:13],
                )
                queue.append({
                    "platform": "etsy",
                    "listing_id": resp.get("listing_id"),
                    "status": "draft_created",
                })
            except Exception as _:
                queue.append({
                    "platform": "etsy",
                    "status": "draft_failed",
                    "title": l.get("title"),
                })

    return PublicationPlan(publication_queue=queue)
