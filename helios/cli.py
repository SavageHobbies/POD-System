from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .config import load_config
from .designer.text_art import create_text_design
from .generator.idea_generator import generate_ideas
from .trends.google_trends import fetch_trends
from .publisher.printify_publisher import PrintifyPublisher


DEFAULT_BLUEPRINTS = {
    # Heuristic defaults; adjust in .env if you have exact IDs
    "tee": {
        "blueprint_id": 482,  # Example placeholder; verify via Printify Catalog
        "print_provider_id": 1,  # Printify's default in-network provider placeholder
    }
}


def run_end_to_end(
    seed: Optional[str],
    geo: str,
    num_ideas: int,
    draft: Optional[bool],
    margin: Optional[float],
    blueprint_id: Optional[int],
    print_provider_id: Optional[int],
) -> None:
    cfg = load_config()

    # Step 1: Trends
    trends = fetch_trends(seed=seed, geo=geo, timeframe="now 7-d", top_n=10)

    # Step 2: Ideas
    ideas = generate_ideas(trends, num_ideas=num_ideas)
    if not ideas:
        raise RuntimeError("No ideas generated")
    slogan = ideas[0]

    # Step 3: Design
    out_path = create_text_design(
        text=slogan,
        out_dir=cfg.output_dir,
        fonts_dir=cfg.fonts_dir,
    )

    print(f"Created design: {out_path}")

    if cfg.dry_run:
        print("DRY_RUN enabled. Skipping Printify publish.")
        return

    # Step 4: Publish
    final_blueprint_id = blueprint_id or cfg.blueprint_id or DEFAULT_BLUEPRINTS["tee"]["blueprint_id"]
    final_provider_id = print_provider_id or cfg.print_provider_id or DEFAULT_BLUEPRINTS["tee"]["print_provider_id"]

    publisher = PrintifyPublisher(api_token=cfg.printify_api_token, shop_id=cfg.printify_shop_id)
    file_id = publisher.upload_design(out_path)

    price_cents = int(round(2499 * (1 + (margin if margin is not None else cfg.default_margin))))
    # Round to .99
    if price_cents % 100 < 99:
        price_cents = price_cents - (price_cents % 100) + 99

    product = publisher.create_product(
        title=slogan,
        description=f"Auto-generated design: {slogan}",
        blueprint_id=final_blueprint_id,
        print_provider_id=final_provider_id,
        print_area_file_id=file_id,
        variant_price_cents=price_cents,
        colors=cfg.default_colors,
        sizes=cfg.default_sizes,
    )

    product_id = product.get("id") or product.get("product_id")
    if not product_id:
        raise RuntimeError(f"Failed to create product: {product}")

    publish_as_draft = cfg.default_draft if draft is None else draft
    result = publisher.publish_product(product_id=product_id, publish_to_store=True, publish_as_draft=publish_as_draft)
    print(f"Published product (draft={publish_as_draft}): {result}")



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Helios autonomous merch pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run end-to-end: trends -> ideas -> design -> publish")
    run.add_argument("--seed", type=str, default=None, help="Optional seed keyword for trends")
    run.add_argument("--geo", type=str, default="US", help="Geo for Google Trends (default: US)")
    run.add_argument("--num-ideas", type=int, default=8, help="How many ideas to generate")
    run.add_argument("--draft", type=lambda x: x.lower() == "true", default=None, help="Publish as draft (true/false)")
    run.add_argument("--margin", type=float, default=None, help="Profit margin as fraction (0.5 = 50%)")
    run.add_argument("--blueprint-id", type=int, default=None, help="Printify blueprint id")
    run.add_argument("--print-provider-id", type=int, default=None, help="Printify print provider id")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_end_to_end(
            seed=args.seed,
            geo=args.geo,
            num_ideas=args.num_ideas,
            draft=args.draft,
            margin=args.margin,
            blueprint_id=args.blueprint_id,
            print_provider_id=args.print_provider_id,
        )
