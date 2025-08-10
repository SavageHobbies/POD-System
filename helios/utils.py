import os
import re
from pathlib import Path
from typing import Iterable


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\-\s]", "", text)
    text = re.sub(r"\s+", "-", text).strip("-")
    text = re.sub(r"-+", "-", text)
    return text or "design"


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def first_non_empty(values: Iterable[str | None], default: str | None = None) -> str | None:
    for v in values:
        if v:
            return v
    return default
