from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class HeliosConfig:
    gemini_api_key: Optional[str]
    openai_api_key: Optional[str]

    printify_api_key: Optional[str]
    printify_shop_id: Optional[str]

    google_drive_folder_id: Optional[str]
    google_sheets_tracking_id: Optional[str]
    google_service_account_json: Optional[str]

    news_api_key: Optional[str]
    trends_api_key: Optional[str]
    social_api_keys: Optional[str]

    mcp_server_url: Optional[str]
    mcp_auth_token: Optional[str]

    min_opportunity_score: float
    min_audience_confidence: float
    min_profit_margin: float
    max_execution_time: int

    enable_parallel_processing: bool
    enable_batch_creation: bool
    enable_adaptive_learning: bool
    enable_auto_optimization: bool

    allow_live_publishing: bool

    # Etsy
    etsy_api_key: Optional[str]
    etsy_oauth_token: Optional[str]
    etsy_shop_id: Optional[str]
    etsy_taxonomy_id: Optional[str]
    etsy_shipping_profile_id: Optional[str]
    etsy_return_policy_id: Optional[str]

    @staticmethod
    def load(env_file: str | None = None) -> "HeliosConfig":
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        def getenv_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except Exception:
                return default

        def getenv_bool(name: str, default: bool) -> bool:
            val = os.getenv(name)
            if val is None:
                return default
            return val.lower() in {"1", "true", "yes", "on"}

        return HeliosConfig(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            printify_api_key=os.getenv("PRINTIFY_API_KEY"),
            printify_shop_id=os.getenv("PRINTIFY_SHOP_ID"),
            google_drive_folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
            google_sheets_tracking_id=os.getenv("GOOGLE_SHEETS_TRACKING_ID"),
            google_service_account_json=os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"),
            news_api_key=os.getenv("NEWS_API_KEY"),
            trends_api_key=os.getenv("TRENDS_API_KEY"),
            social_api_keys=os.getenv("SOCIAL_API_KEYS"),
            mcp_server_url=os.getenv("MCP_SERVER_URL"),
            mcp_auth_token=os.getenv("MCP_AUTH_TOKEN"),
            min_opportunity_score=getenv_float("MIN_OPPORTUNITY_SCORE", 7.0),
            min_audience_confidence=getenv_float("MIN_AUDIENCE_CONFIDENCE", 7.0),
            min_profit_margin=getenv_float("MIN_PROFIT_MARGIN", 0.35),
            max_execution_time=int(getenv_float("MAX_EXECUTION_TIME", 300)),
            enable_parallel_processing=getenv_bool("ENABLE_PARALLEL_PROCESSING", True),
            enable_batch_creation=getenv_bool("ENABLE_BATCH_CREATION", True),
            enable_adaptive_learning=getenv_bool("ENABLE_ADAPTIVE_LEARNING", True),
            enable_auto_optimization=getenv_bool("ENABLE_AUTO_OPTIMIZATION", True),
            allow_live_publishing=getenv_bool("ALLOW_LIVE_PUBLISHING", False),
            etsy_api_key=os.getenv("ETSY_API_KEY"),
            etsy_oauth_token=os.getenv("ETSY_OAUTH_TOKEN"),
            etsy_shop_id=os.getenv("ETSY_SHOP_ID"),
            etsy_taxonomy_id=os.getenv("ETSY_TAXONOMY_ID"),
            etsy_shipping_profile_id=os.getenv("ETSY_SHIPPING_PROFILE_ID"),
            etsy_return_policy_id=os.getenv("ETSY_RETURN_POLICY_ID"),
        )

    def service_account_dict(self) -> Optional[dict[str, Any]]:
        if not self.google_service_account_json:
            return None
        raw = self.google_service_account_json.strip()
        if raw.startswith("{"):
            try:
                return json.loads(raw)
            except Exception:
                return None
        if os.path.exists(raw):
            try:
                with open(raw, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None
