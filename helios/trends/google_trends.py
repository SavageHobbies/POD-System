from __future__ import annotations

from typing import List
import requests
import xml.etree.ElementTree as ET


def fetch_trends(seed: str | None = None, geo: str = "US", timeframe: str = "now 7-d", top_n: int = 10) -> List[str]:
    """
    Returns a list of trend terms using public Google Trends RSS as a last-resort fallback.
    """
    # Fallback: Google Trends RSS (Top daily trends)
    try:
        # Example RSS: https://trends.google.com/trends/trendingsearches/daily/rss?geo=US
        url = f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo.upper()}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {"ns": "http://www.w3.org/2005/Atom"}
        # The feed is RSS 2.0 with channel/item
        items = root.findall("channel/item")
        terms: List[str] = []
        for it in items[: top_n * 2]:
            title_el = it.find("title")
            if title_el is not None and title_el.text:
                terms.append(title_el.text)
        if terms:
            return terms[:top_n]
    except Exception:
        pass

    return ["minimalist quote", "positive vibes", "summer aesthetic", "pet lover", "hiking"][:top_n]
