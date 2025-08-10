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

    # Google Cloud & MCP Integration
    google_api_key: Optional[str] = None  # Gemini
    gemini_model: Optional[str] = None
    google_cloud_project: Optional[str] = None
    google_service_account_json: Optional[str] = None
    google_drive_folder_id: Optional[str] = None
    
    # Google MCP Configuration
    google_mcp_url: Optional[str] = None
    google_mcp_auth_token: Optional[str] = None
    
    # Google Sheets
    gsheet_id: Optional[str] = None
    
    # Performance Thresholds
    min_opportunity_score: float = 7.0
    min_audience_confidence: float = 0.7  # 0-1 scale (70% confidence)
    min_profit_margin: float = 0.35
    max_execution_time: int = 300
    
    # Optimization Settings
    enable_parallel_processing: bool = True
    enable_batch_creation: bool = True
    enable_adaptive_learning: bool = True
    enable_auto_optimization: bool = True
    
    # Safety & Publishing
    allow_live_publishing: bool = False
    
    # Other integrations
    serpapi_key: Optional[str] = None
    
    # Google Cloud Infrastructure Settings
    google_cloud_region: str = "us-central1"
    google_cloud_location: str = "us-central1"
    
    # Vertex AI Configuration
    vertex_ai_project_id: Optional[str] = None
    vertex_ai_location: Optional[str] = None
    
    # Gemini AI Models
    gemini_pro_model: str = "gemini-1.5-pro"
    gemini_flash_model: str = "gemini-1.5-flash"
    gemini_ultra_model: str = "gemini-1.0-ultra"
    
    # Image Generation
    imagen_model: str = "imagen-3"
    image_resolution: str = "1024x1024"
    image_format: str = "PNG"
    image_quality: str = "high"
    
    # MCP Tools Configuration
    enable_google_trends_tool: bool = True
    enable_social_media_scanner: bool = True
    enable_news_analyzer: bool = True
    enable_competitor_intelligence: bool = True
    
    # Performance Monitoring
    enable_performance_monitoring: bool = True
    enable_cloud_monitoring: bool = True
    enable_cloud_logging: bool = True
    enable_cloud_trace: bool = True
    
    # Caching Configuration
    enable_redis_caching: bool = True
    cache_ttl_trend_data: int = 3600  # 1 hour
    cache_ttl_printify_catalog: int = 86400  # 24 hours
    cache_ttl_api_responses: int = 300  # 5 minutes


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

    def parse_float(value: Optional[str], default: float) -> float:
        try:
            return float(value) if value not in (None, "") else default
        except ValueError:
            return default

    def parse_bool(value: Optional[str], default: bool) -> bool:
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    colors = split_csv(os.getenv("DEFAULT_COLOR", "white"))
    sizes = split_csv(os.getenv("DEFAULT_SIZES", "S,M,L,XL,2XL"))

    default_margin = parse_float(os.getenv("DEFAULT_MARGIN"), 0.5)
    default_draft = parse_bool(os.getenv("DEFAULT_DRAFT"), True)
    dry_run = parse_bool(os.getenv("DRY_RUN"), True)

    # Performance thresholds
    min_opportunity_score = parse_float(os.getenv("MIN_OPPORTUNITY_SCORE"), 7.0)
    min_audience_confidence = parse_float(os.getenv("MIN_AUDIENCE_CONFIDENCE"), 7.0) / 10.0  # Convert from 0-10 to 0-1 scale
    min_profit_margin = parse_float(os.getenv("MIN_PROFIT_MARGIN"), 0.35)
    max_execution_time = parse_int(os.getenv("MAX_EXECUTION_TIME")) or 300

    # Optimization settings
    enable_parallel_processing = parse_bool(os.getenv("ENABLE_PARALLEL_PROCESSING"), True)
    enable_batch_creation = parse_bool(os.getenv("ENABLE_BATCH_CREATION"), True)
    enable_adaptive_learning = parse_bool(os.getenv("ENABLE_ADAPTIVE_LEARNING"), True)
    enable_auto_optimization = parse_bool(os.getenv("ENABLE_AUTO_OPTIMIZATION"), True)

    # Safety settings
    allow_live_publishing = parse_bool(os.getenv("ALLOW_LIVE_PUBLISHING"), False)

    # Google Cloud settings
    google_cloud_region = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
    google_cloud_location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

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
        
        # Google Cloud & MCP
        google_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
        google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        google_service_account_json=os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"),
        google_drive_folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
        
        # Google MCP
        google_mcp_url=os.getenv("GOOGLE_MCP_URL"),
        google_mcp_auth_token=os.getenv("GOOGLE_MCP_AUTH_TOKEN"),
        
        # Google Sheets
        gsheet_id=os.getenv("GOOGLE_SHEETS_TRACKING_ID"),
        
        # Performance thresholds
        min_opportunity_score=min_opportunity_score,
        min_audience_confidence=min_audience_confidence,
        min_profit_margin=min_profit_margin,
        max_execution_time=max_execution_time,
        
        # Optimization settings
        enable_parallel_processing=enable_parallel_processing,
        enable_batch_creation=enable_batch_creation,
        enable_adaptive_learning=enable_adaptive_learning,
        enable_auto_optimization=enable_auto_optimization,
        
        # Safety settings
        allow_live_publishing=allow_live_publishing,
        
        # Other integrations
        serpapi_key=os.getenv("SERPAPI_API_KEY"),
        
        # Google Cloud Infrastructure
        google_cloud_region=google_cloud_region,
        google_cloud_location=google_cloud_location,
        vertex_ai_project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
        vertex_ai_location=google_cloud_location,
        
        # Gemini AI Models
        gemini_pro_model=os.getenv("GEMINI_PRO_MODEL", "gemini-1.5-pro"),
        gemini_flash_model=os.getenv("GEMINI_FLASH_MODEL", "gemini-1.5-flash"),
        gemini_ultra_model=os.getenv("GEMINI_ULTRA_MODEL", "gemini-1.0-ultra"),
        
        # Image Generation
        imagen_model=os.getenv("IMAGEN_MODEL", "imagen-3"),
        image_resolution=os.getenv("IMAGE_RESOLUTION", "1024x1024"),
        image_format=os.getenv("IMAGE_FORMAT", "PNG"),
        image_quality=os.getenv("IMAGE_QUALITY", "high"),
        
        # MCP Tools
        enable_google_trends_tool=parse_bool(os.getenv("ENABLE_GOOGLE_TRENDS_TOOL"), True),
        enable_social_media_scanner=parse_bool(os.getenv("ENABLE_SOCIAL_MEDIA_SCANNER"), True),
        enable_news_analyzer=parse_bool(os.getenv("ENABLE_NEWS_ANALYZER"), True),
        enable_competitor_intelligence=parse_bool(os.getenv("ENABLE_COMPETITOR_INTELLIGENCE"), True),
        
        # Performance Monitoring
        enable_performance_monitoring=parse_bool(os.getenv("ENABLE_PERFORMANCE_MONITORING"), True),
        enable_cloud_monitoring=parse_bool(os.getenv("ENABLE_CLOUD_MONITORING"), True),
        enable_cloud_logging=parse_bool(os.getenv("ENABLE_CLOUD_LOGGING"), True),
        enable_cloud_trace=parse_bool(os.getenv("ENABLE_CLOUD_TRACE"), True),
        
        # Caching
        enable_redis_caching=parse_bool(os.getenv("ENABLE_REDIS_CACHING"), True),
        cache_ttl_trend_data=parse_int(os.getenv("CACHE_TTL_TREND_DATA")) or 3600,
        cache_ttl_printify_catalog=parse_int(os.getenv("CACHE_TTL_PRINTIFY_CATALOG")) or 86400,
        cache_ttl_api_responses=parse_int(os.getenv("CACHE_TTL_API_RESPONSES")) or 300,
    )
