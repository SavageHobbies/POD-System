from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
from ..config import load_config
from ..mcp_client import MCPClient


@dataclass
class PerformanceMetrics:
    immediate_metrics: Dict[str, float]
    short_term_metrics: Dict[str, float]
    strategic_metrics: Dict[str, float]
    execution_time: float
    success_rate: float
    roi_estimate: float


@dataclass
class OptimizationTrigger:
    trigger_type: str
    threshold: float
    current_value: float
    recommendation: str
    priority: str
    automated_action: Optional[str] = None


@dataclass
class PerformanceAnalysis:
    metrics: PerformanceMetrics
    optimization_triggers: List[OptimizationTrigger]
    recommendations: List[str]
    risk_factors: List[str]
    success_prediction: float
    execution_time_ms: int
    mcp_model_used: str


class PerformanceIntelligenceAgent:
    """Performance monitoring and optimization agent for the Helios system."""

    def __init__(self) -> None:
        self.config = load_config()
        self.mcp_client = MCPClient.from_env(self.config.google_mcp_url, self.config.google_mcp_auth_token)
        
        # Performance thresholds from config
        self.min_opportunity_score = self.config.min_opportunity_score
        self.min_audience_confidence = self.config.min_audience_confidence
        self.min_profit_margin = self.config.min_profit_margin
        self.max_execution_time = self.config.max_execution_time
        
        # Historical performance data storage
        self.performance_history_file = Path("output/performance_history.json")
        self.performance_history_file.parent.mkdir(exist_ok=True)
        
        # Load historical data
        self.historical_data = self._load_performance_history()

    async def analyze_pipeline_performance(self, pipeline_results: Dict[str, Any]) -> PerformanceAnalysis:
        """Analyze pipeline performance and provide optimization insights."""
        start_time = time.time()
        
        try:
            # Extract performance metrics
            metrics = self._extract_performance_metrics(pipeline_results)
            
            # Check for optimization triggers
            optimization_triggers = self._check_optimization_triggers(metrics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics, optimization_triggers)
            
            # Assess risk factors
            risk_factors = self._assess_risk_factors(metrics)
            
            # Predict success probability
            success_prediction = self._predict_success_probability(metrics)
            
            # Store performance data for historical analysis
            self._store_performance_data(pipeline_results, metrics)
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            return PerformanceAnalysis(
                metrics=metrics,
                optimization_triggers=optimization_triggers,
                recommendations=recommendations,
                risk_factors=risk_factors,
                success_prediction=success_prediction,
                execution_time_ms=execution_time_ms,
                mcp_model_used="local_analysis"
            )
            
        except Exception as e:
            print(f"Error in performance analysis: {e}")
            return self._fallback_analysis(pipeline_results)

    def _extract_performance_metrics(self, pipeline_results: Dict[str, Any]) -> PerformanceMetrics:
        """Extract performance metrics from pipeline results."""
        # Immediate metrics (available right after execution)
        immediate_metrics = {
            "execution_time": pipeline_results.get("pipeline_execution_time", 0),
            "trend_score": pipeline_results.get("performance_metrics", {}).get("trend_score", 0),
            "confidence_level": pipeline_results.get("performance_metrics", {}).get("confidence_level", 0),
            "designs_generated": pipeline_results.get("performance_metrics", {}).get("designs_generated", 0),
            "products_configured": pipeline_results.get("performance_metrics", {}).get("products_configured", 0)
        }
        
        # Short-term metrics (available within 48 hours)
        short_term_metrics = {
            "conversion_rate": 0.0,  # Would be updated from external data
            "click_through_rate": 0.0,
            "favorite_rate": 0.0,
            "time_to_first_view": 0.0
        }
        
        # Strategic metrics (long-term performance)
        strategic_metrics = {
            "lifetime_value": 0.0,
            "repeat_purchase_rate": 0.0,
            "seasonal_performance": 0.0,
            "market_share": 0.0
        }
        
        # Calculate success rate
        success_rate = 1.0 if pipeline_results.get("status") == "success" else 0.0
        
        # Estimate ROI based on trend score and confidence
        trend_score = immediate_metrics["trend_score"]
        confidence = immediate_metrics["confidence_level"]
        roi_estimate = (trend_score / 10.0) * confidence * 3.0  # Base 300% target
        
        return PerformanceMetrics(
            immediate_metrics=immediate_metrics,
            short_term_metrics=short_term_metrics,
            strategic_metrics=strategic_metrics,
            execution_time=immediate_metrics["execution_time"],
            success_rate=success_rate,
            roi_estimate=roi_estimate
        )

    def _check_optimization_triggers(self, metrics: PerformanceMetrics) -> List[OptimizationTrigger]:
        """Check for optimization triggers based on performance thresholds."""
        triggers = []
        
        # Check execution time
        if metrics.execution_time > self.max_execution_time:
            triggers.append(OptimizationTrigger(
                trigger_type="execution_time_exceeded",
                threshold=self.max_execution_time,
                current_value=metrics.execution_time,
                recommendation="Optimize pipeline execution for faster processing",
                priority="high",
                automated_action="enable_parallel_processing"
            ))
        
        # Check trend score
        if metrics.immediate_metrics["trend_score"] < self.min_opportunity_score:
            triggers.append(OptimizationTrigger(
                trigger_type="low_trend_score",
                threshold=self.min_opportunity_score,
                current_value=metrics.immediate_metrics["trend_score"],
                recommendation="Improve trend discovery and scoring algorithms",
                priority="medium",
                automated_action="enhance_trend_analysis"
            ))
        
        # Check confidence level
        if metrics.immediate_metrics["confidence_level"] < self.min_audience_confidence:
            triggers.append(OptimizationTrigger(
                trigger_type="low_confidence",
                threshold=self.min_audience_confidence,
                current_value=metrics.immediate_metrics["confidence_level"],
                recommendation="Enhance audience analysis and data quality",
                priority="medium",
                automated_action="improve_audience_data"
            ))
        
        # Check success rate
        if metrics.success_rate < 0.85:  # 85% target
            triggers.append(OptimizationTrigger(
                trigger_type="low_success_rate",
                threshold=0.85,
                current_value=metrics.success_rate,
                recommendation="Investigate pipeline failures and improve error handling",
                priority="high",
                automated_action="enhance_error_recovery"
            ))
        
        # Check ROI estimate
        if metrics.roi_estimate < 3.0:  # 300% target
            triggers.append(OptimizationTrigger(
                trigger_type="low_roi_estimate",
                threshold=3.0,
                current_value=metrics.roi_estimate,
                recommendation="Optimize product selection and pricing strategy",
                priority="high",
                automated_action="adjust_pricing_strategy"
            ))
        
        return triggers

    def _generate_recommendations(self, metrics: PerformanceMetrics, triggers: List[OptimizationTrigger]) -> List[str]:
        """Generate optimization recommendations based on performance analysis."""
        recommendations = []
        
        # Add recommendations based on triggers
        for trigger in triggers:
            recommendations.append(f"{trigger.recommendation} (Priority: {trigger.priority})")
        
        # Add general optimization suggestions
        if metrics.execution_time > self.max_execution_time * 0.8:
            recommendations.append("Consider enabling batch processing for better efficiency")
        
        if metrics.immediate_metrics["designs_generated"] < 3:
            recommendations.append("Increase design generation for better product variety")
        
        if metrics.immediate_metrics["products_configured"] < 2:
            recommendations.append("Expand product catalog for broader market coverage")
        
        # Add psychological marketing recommendations
        if metrics.immediate_metrics["trend_score"] >= 8.0:
            recommendations.append("High-performing trend detected - consider rapid scaling")
        
        if metrics.immediate_metrics["confidence_level"] >= 0.8:
            recommendations.append("Strong audience confidence - optimize for conversion")
        
        return recommendations

    def _assess_risk_factors(self, metrics: PerformanceMetrics) -> List[str]:
        """Assess potential risk factors based on performance metrics."""
        risk_factors = []
        
        # Execution risks
        if metrics.execution_time > self.max_execution_time:
            risk_factors.append("Pipeline execution time exceeds limits")
        
        if metrics.success_rate < 0.7:
            risk_factors.append("Low success rate indicates system instability")
        
        # Quality risks
        if metrics.immediate_metrics["trend_score"] < 6.0:
            risk_factors.append("Low trend scores may indicate poor market timing")
        
        if metrics.immediate_metrics["confidence_level"] < 0.6:
            risk_factors.append("Low confidence suggests unreliable audience data")
        
        # Business risks
        if metrics.roi_estimate < 2.0:
            risk_factors.append("Low ROI estimates may indicate poor product-market fit")
        
        # Operational risks
        if metrics.immediate_metrics["designs_generated"] == 0:
            risk_factors.append("No designs generated - creative pipeline failure")
        
        if metrics.immediate_metrics["products_configured"] == 0:
            risk_factors.append("No products configured - product pipeline failure")
        
        return risk_factors

    def _predict_success_probability(self, metrics: PerformanceMetrics) -> float:
        """Predict success probability based on historical data and current metrics."""
        base_probability = 0.5
        
        # Boost for good performance indicators
        if metrics.immediate_metrics["trend_score"] >= 8.0:
            base_probability += 0.2
        elif metrics.immediate_metrics["trend_score"] >= 7.0:
            base_probability += 0.1
        
        if metrics.immediate_metrics["confidence_level"] >= 0.8:
            base_probability += 0.15
        elif metrics.immediate_metrics["confidence_level"] >= 0.7:
            base_probability += 0.1
        
        if metrics.execution_time <= self.max_execution_time * 0.7:
            base_probability += 0.1
        
        if metrics.immediate_metrics["designs_generated"] >= 3:
            base_probability += 0.05
        
        # Historical performance boost
        if self.historical_data:
            recent_success_rate = self._calculate_recent_success_rate()
            if recent_success_rate > 0.8:
                base_probability += 0.1
            elif recent_success_rate > 0.7:
                base_probability += 0.05
        
        return min(base_probability, 0.95)

    def _store_performance_data(self, pipeline_results: Dict[str, Any], metrics: PerformanceMetrics) -> None:
        """Store performance data for historical analysis."""
        try:
            performance_record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pipeline_status": pipeline_results.get("status", "unknown"),
                "metrics": {
                    "execution_time": metrics.execution_time,
                    "trend_score": metrics.immediate_metrics["trend_score"],
                    "confidence_level": metrics.immediate_metrics["confidence_level"],
                    "designs_generated": metrics.immediate_metrics["designs_generated"],
                    "products_configured": metrics.immediate_metrics["products_configured"],
                    "success_rate": metrics.success_rate,
                    "roi_estimate": metrics.roi_estimate
                },
                "trend_name": pipeline_results.get("trend_data", {}).get("trend_name", "unknown"),
                "mcp_models_used": pipeline_results.get("performance_metrics", {}).get("mcp_models_used", {})
            }
            
            # Load existing data
            if self.performance_history_file.exists():
                with open(self.performance_history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Add new record
            history.append(performance_record)
            
            # Keep only last 100 records
            if len(history) > 100:
                history = history[-100:]
            
            # Save updated history
            with open(self.performance_history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            # Update in-memory data
            self.historical_data = history
            
        except Exception as e:
            print(f"Error storing performance data: {e}")

    def _load_performance_history(self) -> List[Dict[str, Any]]:
        """Load historical performance data."""
        try:
            if self.performance_history_file.exists():
                with open(self.performance_history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading performance history: {e}")
        
        return []

    def _calculate_recent_success_rate(self) -> float:
        """Calculate recent success rate from historical data."""
        if not self.historical_data:
            return 0.0
        
        # Consider last 20 records
        recent_records = self.historical_data[-20:]
        successful_runs = sum(1 for record in recent_records if record.get("pipeline_status") == "success")
        
        return successful_runs / len(recent_records) if recent_records else 0.0

    def _fallback_analysis(self, pipeline_results: Dict[str, Any]) -> PerformanceAnalysis:
        """Fallback analysis when main analysis fails."""
        metrics = PerformanceMetrics(
            immediate_metrics={"execution_time": 0, "trend_score": 0, "confidence_level": 0, "designs_generated": 0, "products_configured": 0},
            short_term_metrics={},
            strategic_metrics={},
            execution_time=0,
            success_rate=0,
            roi_estimate=0
        )
        
        return PerformanceAnalysis(
            metrics=metrics,
            optimization_triggers=[],
            recommendations=["System analysis failed - manual review required"],
            risk_factors=["Performance analysis system failure"],
            success_prediction=0.0,
            execution_time_ms=0,
            mcp_model_used="fallback"
        )

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of overall system performance."""
        if not self.historical_data:
            return {"status": "no_data", "message": "No performance data available"}
        
        # Calculate summary statistics
        total_runs = len(self.historical_data)
        successful_runs = sum(1 for record in self.historical_data if record.get("pipeline_status") == "success")
        success_rate = successful_runs / total_runs if total_runs > 0 else 0
        
        # Calculate average metrics
        avg_execution_time = sum(record["metrics"]["execution_time"] for record in self.historical_data) / total_runs
        avg_trend_score = sum(record["metrics"]["trend_score"] for record in self.historical_data) / total_runs
        avg_confidence = sum(record["metrics"]["confidence_level"] for record in self.historical_data) / total_runs
        
        return {
            "status": "success",
            "summary": {
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "success_rate": success_rate,
                "average_execution_time": avg_execution_time,
                "average_trend_score": avg_trend_score,
                "average_confidence": avg_confidence
            },
            "trends": self._analyze_performance_trends()
        }

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.historical_data) < 5:
            return {"status": "insufficient_data"}
        
        # Simple trend analysis
        recent_data = self.historical_data[-10:]
        older_data = self.historical_data[-20:-10] if len(self.historical_data) >= 20 else self.historical_data[:-10]
        
        if not older_data:
            return {"status": "insufficient_data"}
        
        # Calculate trend changes
        recent_avg_score = sum(record["metrics"]["trend_score"] for record in recent_data) / len(recent_data)
        older_avg_score = sum(record["metrics"]["trend_score"] for record in older_data) / len(older_data)
        
        recent_avg_time = sum(record["metrics"]["execution_time"] for record in recent_data) / len(recent_data)
        older_avg_time = sum(record["metrics"]["execution_time"] for record in older_data) / len(older_data)
        
        return {
            "trend_score_change": recent_avg_score - older_avg_score,
            "execution_time_change": recent_avg_time - older_avg_time,
            "trend_direction": "improving" if recent_avg_score > older_avg_score else "declining",
            "efficiency_direction": "improving" if recent_avg_time < older_avg_time else "declining"
        }
