from __future__ import annotations

import os
import random
from typing import List


SYSTEM_PROMPT = (
    "You are a creative merch ideation assistant. Generate short, catchy, PG-13 safe shirt slogans (2-6 words) tailored to the given trend/audience. Avoid trademarks and brand names."
)


def generate_ideas(trend_terms: List[str], num_ideas: int = 8) -> List[str]:
    """Generate slogan ideas. Uses OpenAI if available; otherwise rule-based fallback."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            # Lazy import to avoid hard dependency
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            prompt = (
                f"Trend terms: {', '.join(trend_terms)}\n"
                "Generate unique, punchy t-shirt slogans, 2-6 words, no brands/trademarks."
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                n=1,
                temperature=0.9,
            )
            text = response.choices[0].message.content or ""
            # Split lines and clean
            ideas = [line.strip("- \n\t") for line in text.splitlines() if line.strip()]
            cleaned = [i for i in ideas if 2 <= len(i.split()) <= 6][: num_ideas * 2]
            # Deduplicate and limit
            seen = set()
            result: List[str] = []
            for idea in cleaned:
                k = idea.lower()
                if k not in seen:
                    seen.add(k)
                    result.append(idea)
                if len(result) >= num_ideas:
                    break
            if result:
                return result
        except Exception:
            pass

    # Fallback: simple templates
    seeds = [t for t in trend_terms if t]
    if not seeds:
        seeds = ["vibes", "adventure", "coffee", "cats", "books", "retro", "minimal"]

    templates = [
        "Powered by {seed}",
        "Less talk, more {seed}",
        "I need {seed}",
        "Trust me, I'm {seed}",
        "Born for {seed}",
        "Keep it {seed}",
        "Just add {seed}",
        "Made of {seed}",
        "Future of {seed}",
        "Certified {seed}",
    ]

    out: List[str] = []
    random.shuffle(seeds)
    for seed in seeds:
        token = seed.split()[0].lower()
        token = token.replace("#", "").replace("@", "")
        for tmpl in random.sample(templates, k=min(3, len(templates))):
            out.append(tmpl.format(seed=token))
            if len(out) >= num_ideas:
                return out
    return out[:num_ideas]
