from __future__ import annotations

from typing import List


def fetch_trends(seed: str | None = None, geo: str = "US", timeframe: str = "now 7-d", top_n: int = 10) -> List[str]:
    """
    Returns a list of trend terms. If pytrends is available, use it; otherwise return a fallback list.
    """
    try:
        from pytrends.request import TrendReq  # type: ignore
    except Exception:
        # Fallback: lightweight seed-based ideas
        fallback = [
            "retro vaporwave",
            "funny cat meme",
            "fitness motivation",
            "dad jokes shirt",
            "ai generated art",
            "cottagecore aesthetic",
            "booktok quotes",
            "gaming nostalgia",
            "plant mom",
            "coffee lover",
        ]
        return fallback[:top_n]

    pytrends = TrendReq(hl="en-US", tz=360)

    if seed:
        kw_list = [seed]
        pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo=geo, gprop="")
        related = pytrends.related_queries()
        try:
            top_df = related.get(seed, {}).get("top")
            if top_df is not None and not top_df.empty:
                return [str(x) for x in top_df["query"].head(top_n).tolist()]
        except Exception:
            pass

    # If no seed or related failed, fallback to trending searches
    try:
        trending_df = pytrends.trending_searches(pn="united_states")
        if trending_df is not None and not trending_df.empty:
            return [str(x) for x in trending_df[0].head(top_n).tolist()]
    except Exception:
        pass

    # Final fallback
    return ["minimalist quote", "positive vibes", "summer aesthetic", "pet lover", "hiking"][:top_n]
