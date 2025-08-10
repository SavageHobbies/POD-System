from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List
from loguru import logger

from .agents.zeitgeist import ZeitgeistAgent
from .agents.ceo import HeliosCEO
from .agents.ethics import screen_ethics, EthicalGuardianAgent
from .agents.audience import AudienceAnalyst
from .agents.creative import CreativeDirector
from .agents.marketing import MarketingCopywriter
from .agents.product import ProductStrategist
from .agents.publisher_agent import PrintifyPublisherAgent
from .agents.performance import PerformanceIntelligenceAgent
from .config import load_config
from .mcp_client import MCPClient


async def run_helios_pipeline(
    seed: str | None = None,
    dry_run: bool = True,
    enable_parallel: bool = True
) -> Dict[str, Any]:
    """Run the complete Helios pipeline with enhanced priority-driven execution and parallel processing"""
    
    start_time = time.time()
    config = load_config()
    
    # Override dry_run if configured
    if config.dry_run is not None:
        dry_run = config.dry_run
    
    # Override parallel processing if configured
    if config.enable_parallel_processing is not None:
        enable_parallel = config.enable_parallel_processing
    
    logger.info(f"🚀 Starting Helios pipeline (dry_run={dry_run}, parallel={enable_parallel})")
    logger.info(f"🎯 Target execution time: < {config.max_execution_time}s")
    
    try:
        # STAGE 1: Discovery & Validation (Target: 60 seconds)
        logger.info("🔍 STAGE 1: Trend Discovery & Validation (Target: 60s)")
        stage1_start = time.time()
        
        # Parallel execution: Trend detection + CEO validation
        zeitgeist = ZeitgeistAgent()
        ceo = HeliosCEO(
            min_opportunity=config.min_opportunity_score,
            min_confidence=config.min_audience_confidence
        )
        
        if enable_parallel:
            # Run trend detection and CEO validation in parallel
            trend_task = zeitgeist.run(seed)
            ceo_task = ceo.prepare_validation()  # Pre-load CEO agent
            
            trend_data, ceo_prep = await asyncio.gather(trend_task, ceo_task)
        else:
            trend_data = await zeitgeist.run(seed)
            ceo_prep = await ceo.prepare_validation()
        
        if trend_data.get("status") != "approved":
            logger.warning(f"❌ Trend rejected: {trend_data.get('trend_name', 'unknown')}")
            return {"status": "trend_rejected", "reason": "Trend did not meet approval criteria"}
        
        logger.info(f"✅ Trend approved: {trend_data.get('trend_name')} (score: {trend_data.get('opportunity_score')})")
        
        # CEO validation with enhanced quality gates
        ceo_decision = await ceo.validate_trend(trend_data)
        
        if not ceo_decision.approved:
            logger.warning(f"❌ CEO rejected trend: {ceo_decision.trend_name}")
            return {
                "status": "ceo_rejected",
                "reason": f"Opportunity score {ceo_decision.opportunity_score} below threshold {config.min_opportunity_score}",
                "ceo_decision": ceo_decision
            }
        
        logger.info(f"✅ CEO approved trend: {ceo_decision.trend_name}")
        logger.info(f"   Priority: {ceo_decision.priority.value}, MCP Model: {ceo_decision.mcp_model_used}")
        
        stage1_time = time.time() - stage1_start
        logger.info(f"⏱️  Stage 1 completed in {stage1_time:.1f}s")
        
        # STAGE 2: Analysis & Strategy (Target: 45 seconds)
        logger.info("🧠 STAGE 2: Audience & Product Analysis (Target: 45s)")
        stage2_start = time.time()
        
        # Parallel execution: Audience analysis + Product strategy
        audience_analyst = AudienceAnalyst()
        product_manager = ProductStrategist(config)
        
        if enable_parallel:
            # Run audience analysis and product strategy in parallel
            audience_task = audience_analyst.run(trend_data)
            product_task = product_manager.get_products_async()
            
            audience_result, products = await asyncio.gather(audience_task, product_task)
        else:
            audience_result = await audience_analyst.run(trend_data)
            products = product_manager.get_products()
        
        if audience_result.confidence_score < config.min_audience_confidence:
            logger.warning(f"❌ Audience confidence too low: {audience_result.confidence_score:.2f} < {config.min_audience_confidence}")
            return {
                "status": "audience_rejected",
                "reason": f"Audience confidence {audience_result.confidence_score:.2f} below threshold {config.min_audience_confidence}",
                "audience_result": audience_result
            }
        
        if not products:
            logger.error("❌ No products configured")
            return {"status": "error", "reason": "No products configured"}
        
        logger.info(f"✅ Analysis completed: {audience_result.primary_persona.demographic_cluster}")
        logger.info(f"   Confidence: {audience_result.confidence_score:.2f}, Products: {len(products)} types")
        
        stage2_time = time.time() - stage2_start
        logger.info(f"⏱️  Stage 2 completed in {stage2_time:.1f}s")
        
        # STAGE 3: Ethical Screening (Target: 30 seconds)
        logger.info("⚖️ STAGE 3: Ethical Screening (Target: 30s)")
        stage3_start = time.time()
        
        ethics_result = await screen_ethics(
            trend_name=ceo_decision.trend_name,
            keywords=ceo_decision.keywords,
            dry_run=dry_run
        )
        
        if ethics_result.status not in ["approved", "moderate"]:
            logger.warning(f"❌ Ethics screening failed: {ethics_result.status}")
            return {
                "status": "ethics_rejected",
                "reason": f"Ethical concerns: {ethics_result.notes}",
                "ethics_result": ethics_result
            }
        
        logger.info(f"✅ Ethics screening passed: {ethics_result.status}")
        
        stage3_time = time.time() - stage3_start
        logger.info(f"⏱️  Stage 3 completed in {stage3_time:.1f}s")
        
        # STAGE 4: Creative & Marketing (Target: 90 seconds)
        logger.info("🎨 STAGE 4: Creative & Marketing Generation (Target: 90s)")
        stage4_start = time.time()
        
        # Enhanced batch processing for creative and marketing
        creative = CreativeDirector(
            output_dir=config.output_dir,
            fonts_dir=config.fonts_dir
        )
        
        marketing = MarketingCopywriter()
        
        if config.enable_batch_creation and enable_parallel:
            # Batch process creative and marketing in parallel
            creative_task = creative.run_batch(trend_data, products, num_designs_per_product=3)
            marketing_task = marketing.prepare_batch(trend_data, products)
            
            creative_result, marketing_prep = await asyncio.gather(creative_task, marketing_task)
            
            # Generate marketing copy for all designs at once
            marketing_result = await marketing.run_batch(creative_result["designs"], marketing_prep)
        else:
            # Sequential processing
            creative_result = await creative.run(trend_data, products, num_designs_per_product=3)
            
            creative_batch = {
                "trend_name": ceo_decision.trend_name,
                "keywords": ceo_decision.keywords,
                "products": products,
                "designs": creative_result["designs"]
            }
            marketing_result = await marketing.run(creative_batch)
        
        if not creative_result.get("designs"):
            logger.error("❌ No designs generated")
            return {"status": "error", "reason": "No designs generated"}
        
        if not marketing_result.get("marketing_copy"):
            logger.error("❌ No marketing copy generated")
            return {"status": "error", "reason": "No marketing copy generated"}
        
        logger.info(f"✅ Creative & Marketing completed: {len(creative_result['designs'])} designs")
        
        stage4_time = time.time() - stage4_start
        logger.info(f"⏱️  Stage 4 completed in {stage4_time:.1f}s")
        
        # STAGE 5: Publication (Target: 30 seconds)
        logger.info("🚀 STAGE 5: Publication (Target: 30s)")
        stage5_start = time.time()
        
        publish_results = []
        if not dry_run and config.allow_live_publishing:
            publisher = PrintifyPublisherAgent(
                api_token=config.printify_api_token,
                shop_id=str(config.printify_shop_id)
            )
            
            # Prepare products for publishing
            products_to_publish = []
            for design in creative_result["designs"]:
                for product in products:
                    products_to_publish.append({
                        "design": design,
                        "product": product,
                        "marketing": marketing_result["marketing_copy"]
                    })
            
            if config.enable_batch_creation:
                # Publish in batches for efficiency
                publish_results = await publisher.publish_batch(products_to_publish)
            else:
                # Publish individually
                for product_data in products_to_publish:
                    result = await publisher.publish_product(product_data)
                    publish_results.append(result)
            
            logger.info(f"✅ Publishing completed: {len(publish_results)} products")
        else:
            logger.info("⏸️ Publishing skipped (dry_run or not allowed)")
        
        stage5_time = time.time() - stage5_start
        logger.info(f"⏱️  Stage 5 completed in {stage5_time:.1f}s")
        
        # Performance Analysis & Optimization
        total_execution_time = time.time() - start_time
        
        # Check performance against targets
        performance_status = "✅ EXCELLENT" if total_execution_time < 300 else "⚠️  SLOW" if total_execution_time < 400 else "❌ TOO SLOW"
        logger.info(f"{performance_status} Total execution time: {total_execution_time:.1f}s (target: <300s)")
        
        # Run performance intelligence analysis
        logger.info("📊 Analyzing performance intelligence...")
        performance_agent = PerformanceIntelligenceAgent()
        performance_analysis = await performance_agent.analyze_pipeline_performance({
            "status": "success",
            "pipeline_execution_time": total_execution_time,
            "stage_times": {
                "discovery": stage1_time,
                "analysis": stage2_time,
                "ethics": stage3_time,
                "creative": stage4_time,
                "publication": stage5_time
            },
            "trend_data": trend_data,
            "ceo_decision": ceo_decision,
            "audience_result": audience_result,
            "ethics_result": ethics_result,
            "creative_result": creative_result,
            "marketing_result": marketing_result,
            "publish_results": publish_results,
            "performance_metrics": {
                "total_time": total_execution_time,
                "trend_score": ceo_decision.opportunity_score,
                "confidence_level": audience_result.confidence_score,
                "designs_generated": len(creative_result.get("designs", [])),
                "products_configured": len(products),
                "parallel_processing": enable_parallel,
                "batch_creation": config.enable_batch_creation,
                "mcp_models_used": {
                    "ceo": ceo_decision.mcp_model_used,
                    "audience": audience_result.mcp_model_used,
                    "ethics": ethics_result.mcp_model_used,
                    "creative": creative_result.get("mcp_model_used"),
                    "marketing": marketing_result.get("mcp_model_used")
                }
            }
        })
        
        logger.info(f"✅ Performance analysis completed")
        logger.info(f"   Success Prediction: {performance_analysis.success_prediction:.1%}")
        logger.info(f"   Optimization Triggers: {len(performance_analysis.optimization_triggers)}")
        
        # Compile final results
        results = {
            "status": "success",
            "pipeline_execution_time": total_execution_time,
            "stage_times": {
                "discovery": stage1_time,
                "analysis": stage2_time,
                "ethics": stage3_time,
                "creative": stage4_time,
                "publication": stage5_time
            },
            "performance_status": performance_status,
            "trend_data": trend_data,
            "ceo_decision": ceo_decision,
            "audience_result": audience_result,
            "ethics_result": ethics_result,
            "creative_result": creative_result,
            "marketing_result": marketing_result,
            "publish_results": publish_results,
            "performance_analysis": performance_analysis,
            "performance_metrics": {
                "total_time": total_execution_time,
                "trend_score": ceo_decision.opportunity_score,
                "confidence_level": audience_result.confidence_score,
                "designs_generated": len(creative_result.get("designs", [])),
                "products_configured": len(products),
                "parallel_processing": enable_parallel,
                "batch_creation": config.enable_batch_creation,
                "mcp_models_used": {
                    "ceo": ceo_decision.mcp_model_used,
                    "audience": audience_result.mcp_model_used,
                    "ethics": ethics_result.mcp_model_used,
                    "creative": creative_result.get("mcp_model_used"),
                    "marketing": marketing_result.get("mcp_model_used")
                }
            }
        }
        
        # Final performance summary
        if total_execution_time <= config.max_execution_time:
            logger.info(f"🎉 Helios pipeline completed successfully in {total_execution_time:.1f}s")
            logger.info(f"🚀 Performance target achieved: {total_execution_time:.1f}s < {config.max_execution_time}s")
        else:
            logger.warning(f"⚠️  Pipeline completed but exceeded target time: {total_execution_time:.1f}s > {config.max_execution_time}s")
        
        return results
        
    except Exception as e:
        total_execution_time = time.time() - start_time
        logger.error(f"❌ Helios pipeline failed: {e}")
        return {
            "status": "error",
            "reason": str(e),
            "pipeline_execution_time": total_execution_time
        }


def main():
    """Main entry point for Helios"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Helios - AI-Powered Print-on-Demand Pipeline")
    parser.add_argument("--seed", type=str, help="Seed trend to analyze")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Run in dry-run mode")
    parser.add_argument("--live", action="store_true", help="Enable live publishing")
    parser.add_argument("--parallel", action="store_true", default=True, help="Enable parallel processing")
    parser.add_argument("--batch", action="store_true", default=True, help="Enable batch creation")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Override config with command line args
    if args.live:
        config.allow_live_publishing = True
        args.dry_run = False
    
    # Run pipeline
    results = asyncio.run(run_helios_pipeline(
        seed=args.seed,
        dry_run=args.dry_run,
        enable_parallel=args.parallel
    ))
    
    # Print results
    if results["status"] == "success":
        print(f"\n✅ Pipeline completed successfully!")
        print(f"   Trend: {results['trend_data']['trend_name']}")
        print(f"   Score: {results['performance_metrics']['trend_score']}")
        print(f"   Audience: {results['audience_result'].primary_persona.demographic_cluster}")
        print(f"   Confidence: {results['audience_result'].confidence_score:.2f}")
        print(f"   Designs: {results['performance_metrics']['designs_generated']}")
        print(f"   Total Time: {results['pipeline_execution_time']:.1f}s")
        print(f"   Performance: {results['performance_status']}")
        print(f"   Success Prediction: {results['performance_analysis'].success_prediction:.1%}")
        
        # Print stage times
        print(f"\n📊 Stage Performance:")
        for stage, time_taken in results['stage_times'].items():
            print(f"   {stage.title()}: {time_taken:.1f}s")
    else:
        print(f"\n❌ Pipeline failed: {results.get('reason', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    main()
