from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .utils import split_csv


@dataclass
class HeliosConfig:
    # Auth / IDs
    printify_api_token: str
    printify_shop_id: str

    # Product defaults
    blueprint_id: Optional[int] = None
    print_provider_id: Optional[int] = None
    default_colors: list[str] = None
    default_sizes: list[str] = None

    # Behavior
    default_margin: float = 0.5
    default_draft: bool = True
    dry_run: bool = True

    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    assets_dir: Path = project_root / "assets"
    fonts_dir: Path = assets_dir / "fonts"
    output_dir: Path = project_root / "output"


def load_config(env_path: Optional[Path] = None) -> HeliosConfig:
    # Load .env if present
    if env_path is None:
        env_path = Path.cwd() / ".env"
    load_dotenv(dotenv_path=env_path if env_path.exists() else None)

    api_token = os.getenv("PRINTIFY_API_TOKEN", "").strip()
    shop_id = os.getenv("PRINTIFY_SHOP_ID", "").strip()
    if not api_token or not shop_id:
        raise RuntimeError("PRINTIFY_API_TOKEN and PRINTIFY_SHOP_ID must be set (see .env.example)")

    blueprint_id = os.getenv("BLUEPRINT_ID")
    provider_id = os.getenv("PRINT_PROVIDER_ID")

    def parse_int(value: Optional[str]) -> Optional[int]:
        try:
            return int(value) if value not in (None, "") else None
        except ValueError:
            return None

    colors = split_csv(os.getenv("DEFAULT_COLOR", "white"))
    sizes = split_csv(os.getenv("DEFAULT_SIZES", "S,M,L,XL,2XL"))

    default_margin = float(os.getenv("DEFAULT_MARGIN", "0.5"))
    default_draft = os.getenv("DEFAULT_DRAFT", "true").lower() == "true"
    dry_run = os.getenv("DRY_RUN", "true").lower() == "true"

    return HeliosConfig(
        printify_api_token=api_token,
        printify_shop_id=shop_id,
        blueprint_id=parse_int(blueprint_id),
        print_provider_id=parse_int(provider_id),
        default_colors=colors,
        default_sizes=sizes,
        default_margin=default_margin,
        default_draft=default_draft,
        dry_run=dry_run,
    )
