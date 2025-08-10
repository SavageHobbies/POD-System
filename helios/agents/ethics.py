from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class EthicsResult:
    status: Literal["approved", "moderate", "high_risk", "critical"]
    notes: str


async def screen_ethics(trend_name: str, keywords: list[str], dry_run: bool = False) -> EthicsResult:
    if dry_run:
        return EthicsResult(status="approved", notes="No sensitive content detected")

    sensitive = {"violence", "hate", "adult", "illegal"}
    if any(word.lower() in sensitive for word in keywords):
        return EthicsResult(status="high_risk", notes="Sensitive keywords present")
    return EthicsResult(status="approved", notes="Pass")
