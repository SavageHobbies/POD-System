from __future__ import annotations

import argparse
from pathlib import Path
import json
import time
from typing import Optional

from .config import load_config
from .agents.zeitgeist import ZeitgeistAgent
from .agents.ceo import HeliosCEO
from .agents.audience import AudienceAnalyst
from .agents.product import ProductStrategist
from .agents.creative import CreativeDirector
from .agents.marketing import MarketingCopywriter
from .agents.publisher_agent import PrintifyPublisherAgent
from .utils_google import append_rows


DEFAULT_BLUEPRINTS = {
    "tee": {"blueprint_id": 482, "print_provider_id": 1}
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

    # High-priority: Zeitgeist discovery
    start_ts = time.time()
    zeitgeist = ZeitgeistAgent().run(seed=seed)
    decision = HeliosCEO().validate_trend(zeitgeist)
    if not decision.approved:
        result_obj = {
            "execution_summary": {
                "total_time_seconds": int(time.time() - start_ts),
                "agents_used": 2,
                "parallel_executions": 0,
                "quality_scores": {
                    "opportunity_score": decision.opportunity_score,
                    "confidence_level": decision.confidence_level,
                },
            },
            "trend_data": zeitgeist,
            "audience_insights": None,
            "product_portfolio": [],
            "creative_concepts": [],
            "marketing_materials": [],
            "publication_queue": [],
        }
        print(json.dumps(result_obj, indent=2))
        return

    # Parallel: Audience + Product strategy (sequential here, light-weight)
    audience = AudienceAnalyst().run(zeitgeist)
    product = ProductStrategist().run(audience)

    # Batch creative
    creative = CreativeDirector(output_dir=cfg.output_dir, fonts_dir=cfg.fonts_dir)
    creative_batch = creative.run(zeitgeist, product.get("selected_products", []), num_designs_per_product=3)

    # Batch marketing copy
    marketing = MarketingCopywriter().run(creative_batch)

    # Build base result object
    result_obj = {
        "execution_summary": {
            "total_time_seconds": int(time.time() - start_ts),
            "agents_used": 6,
            "parallel_executions": 1,
            "quality_scores": {
                "opportunity_score": decision.opportunity_score,
                "confidence_level": decision.confidence_level,
            },
        },
        "trend_data": zeitgeist,
        "audience_insights": audience,
        "product_portfolio": product.get("selected_products", []),
        "creative_concepts": creative_batch,
        "marketing_materials": marketing,
        "publication_queue": marketing.get("listings", []),
    }

    # Exit early if dry run: emit JSON and write report and sheets log if configured
    if cfg.dry_run:
        report_path = cfg.output_dir / f"run-report-{int(start_ts)}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result_obj, indent=2))
        print(json.dumps(result_obj, indent=2))
        # Log to Sheets if configured
        if cfg.gsheet_id and cfg.gservice_account_json:
            rows = []
            for l in marketing.get("listings", []):
                rows.append([
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    zeitgeist.get("trend_name"),
                    l.get("title"),
                    l.get("product_key"),
                    "auto_text_design",
                    json.dumps(audience.get("primary_persona", {})),
                    ", ".join(l.get("tags", [])),
                    "",
                    "",
                    0,
                    (margin if margin is not None else cfg.default_margin),
                    "DRY_RUN",
                    l.get("image_path"),
                    zeitgeist.get("opportunity_score"),
                    zeitgeist.get("velocity"),
                ])
            append_rows(cfg.gsheet_id, "Product_Launches", rows, cfg.gservice_account_json)
        return

    # Publish batch via Printify → Etsy
    final_blueprint_id = blueprint_id or cfg.blueprint_id or DEFAULT_BLUEPRINTS["tee"]["blueprint_id"]
    final_provider_id = print_provider_id or cfg.print_provider_id or DEFAULT_BLUEPRINTS["tee"]["print_provider_id"]

    # Inject defaults into listings
    for l in marketing["listings"]:
        l.setdefault("blueprint_id", final_blueprint_id)
        l.setdefault("print_provider_id", final_provider_id)
        l.setdefault("colors", cfg.default_colors)
        l.setdefault("sizes", cfg.default_sizes)

    publisher_agent = PrintifyPublisherAgent(api_token=cfg.printify_api_token, shop_id=cfg.printify_shop_id)
    publish_as_draft = cfg.default_draft if draft is None else draft
    # Adjust margin by urgency premium if needed
    eff_margin = (margin if margin is not None else cfg.default_margin)
    if zeitgeist.get("velocity") in ("peak",):
        eff_margin = max(eff_margin + 0.10, 0.35)

    result = publisher_agent.run_batch(marketing["listings"], margin=eff_margin, draft=publish_as_draft)
    result_obj["publication_queue"] = result.get("publication_results", [])
    result_obj["execution_summary"]["total_time_seconds"] = int(time.time() - start_ts)
    report_path = cfg.output_dir / f"run-report-{int(start_ts)}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result_obj, indent=2))
    print(json.dumps(result_obj, indent=2))
    # Log to Sheets if configured
    if cfg.gsheet_id and cfg.gservice_account_json:
        rows = []
        for r in result.get("publication_results", []):
            rows.append([
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                zeitgeist.get("trend_name"),
                r.get("product_title"),
                "tee",
                "auto_text_design",
                json.dumps(audience.get("primary_persona", {})),
                "",
                r.get("printify_product_id"),
                "",
                r.get("final_price"),
                eff_margin,
                r.get("status"),
                r.get("design_id"),
                zeitgeist.get("opportunity_score"),
                zeitgeist.get("velocity"),
            ])
        append_rows(cfg.gsheet_id, "Product_Launches", rows, cfg.gservice_account_json)



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

    test_monitoring = sub.add_parser("test-monitoring", help="Test the Helios monitoring infrastructure")
    
    orchestrator = sub.add_parser("orchestrator", help="Run the Helios Autonomous Store Orchestrator")
    orchestrator.add_argument("--continuous", action="store_true", help="Run in continuous mode (every 6 hours)")
    orchestrator.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    orchestrator.add_argument("--config-file", default="config/development.yaml", help="Configuration file path")
    
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
    elif args.command == "test-monitoring":
        import asyncio
        from .services.google_cloud.performance_monitor import PerformanceMonitor
        
        async def run_monitoring_test():
            try:
                # Load configuration
                cfg = load_config()
                
                # Initialize performance monitor
                monitor = PerformanceMonitor(cfg)
                
                # Set up monitoring infrastructure
                print("🔧 Setting up monitoring infrastructure...")
                await monitor.setup_monitoring_infrastructure()
                
                # Record some test metrics
                print("📊 Recording test metrics...")
                await monitor.record_pipeline_metric(
                    execution_time=120.5,
                    trend_opportunity_score=0.85,
                    audience_confidence=0.92,
                    design_generation_success=0.88,
                    publication_success=0.95
                )
                
                await monitor.record_stage_metric(
                    stage="trend_discovery",
                    duration=45.2,
                    success=True,
                    error_count=0,
                    additional_data={"trends_found": 12, "confidence_avg": 0.87}
                )
                
                # Wait for metrics to be flushed
                print("⏳ Waiting for metrics to be flushed...")
                await asyncio.sleep(35)
                
                # Get performance summary
                summary = await monitor.get_performance_summary()
                print("📈 Performance Summary:")
                print(f"  - Total metrics recorded: {summary.get('total_metrics', 0)}")
                print(f"  - Average execution time: {summary.get('avg_execution_time', 0):.2f}s")
                print(f"  - Success rate: {summary.get('success_rate', 0):.2%}")
                
                print("✅ Monitoring test completed successfully!")
                
            except Exception as e:
                print(f"❌ Monitoring test failed: {e}")
                return 1
            
            return 0
        
        exit_code = asyncio.run(run_monitoring_test())
        exit(exit_code)
    elif args.command == "orchestrator":
        import asyncio
        from .services.helios_orchestrator import create_helios_orchestrator
        
        async def run_orchestrator():
            try:
                # Load configuration
                cfg = load_config(args.config_file)
                
                # Override dry_run if specified
                if args.dry_run:
                    cfg.dry_run = True
                    cfg.allow_live_publishing = False
                
                print("🚀 Starting Helios Autonomous Store Orchestrator...")
                print(f"📁 Config: {args.config_file}")
                print(f"🔒 Dry Run: {cfg.dry_run}")
                print(f"🌐 Project: {cfg.google_cloud_project}")
                
                # Create orchestrator
                orchestrator = await create_helios_orchestrator(cfg)
                
                if args.continuous:
                    print("🔄 Starting continuous operation (every 6 hours)...")
                    await orchestrator.start_continuous_operation()
                else:
                    print("🔄 Running single orchestration cycle...")
                    result = await orchestrator.run_complete_cycle()
                    
                    # Display results
                    print("\n📊 Orchestration Results:")
                    print(f"   Status: {result.get('status')}")
                    print(f"   Session ID: {result.get('session_id')}")
                    print(f"   Execution Time: {result.get('execution_time', 0):.1f}s")
                    print(f"   Trends Discovered: {result.get('trends_discovered', 0)}")
                    print(f"   Products Generated: {result.get('products_generated', 0)}")
                    print(f"   Experiments Created: {result.get('experiments_created', 0)}")
                    
                    # Get summary
                    summary = await orchestrator.get_orchestration_summary()
                    print(f"\n📈 Total Sessions: {summary.get('total_sessions', 0)}")
                    
                await orchestrator.cleanup()
                
            except Exception as e:
                print(f"❌ Orchestrator failed: {e}")
                return 1
            
            return 0
        
        exit_code = asyncio.run(run_orchestrator())
        exit(exit_code)
