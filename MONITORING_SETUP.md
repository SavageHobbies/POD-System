# Helios Monitoring Infrastructure Setup

This document describes the complete monitoring infrastructure for the Helios Autonomous Store, including Cloud Monitoring dashboards, custom metrics, alerting policies, and performance monitoring.

## 🏗️ Architecture Overview

The monitoring system consists of three main components:

1. **CloudMonitoringDashboard** - Creates and manages Google Cloud Monitoring resources
2. **PerformanceMonitor** - Collects, analyzes, and reports performance metrics
3. **CLI Integration** - Command-line tools for testing and managing monitoring

## 🚀 Quick Start

### 1. Test the Monitoring Infrastructure

```bash
# Test the complete monitoring setup
python test_monitoring.py

# Or use the CLI command
python -m helios.cli test-monitoring
```

### 2. Set Up Monitoring in Your Code

```python
from helios.config import load_config
from helios.services.google_cloud.performance_monitor import PerformanceMonitor

# Load configuration
config = load_config()

# Initialize performance monitor
monitor = PerformanceMonitor(config)

# Set up monitoring infrastructure (creates dashboards, metrics, alerts)
await monitor.setup_monitoring_infrastructure()

# Record pipeline metrics
await monitor.record_pipeline_metric(
    execution_time=120.5,
    trend_opportunity_score=0.85,
    audience_confidence=0.92,
    design_generation_success=0.88,
    publication_success=0.95
)

# Record stage-specific metrics
await monitor.record_stage_metric(
    stage="trend_discovery",
    duration=45.2,
    success=True,
    error_count=0,
    additional_data={"trends_found": 12, "confidence_avg": 0.87}
)
```

## 📊 Monitoring Components

### 1. Main Helios Dashboard

The main dashboard displays:
- **Pipeline Execution Time** - Total time from trend discovery to publication
- **Success Rate** - Overall pipeline success percentage
- **Trend Opportunity Scores** - Market opportunity assessment
- **Audience Confidence** - Target audience analysis confidence
- **Design Generation Success** - Creative process success rate
- **Publication Success** - Product publishing success rate

### 2. Performance Intelligence Dashboard

Dedicated dashboard for:
- **Performance Trends** - Historical performance analysis
- **Resource Utilization** - System resource consumption
- **Optimization Opportunities** - AI-powered recommendations
- **Success Predictions** - Predictive analytics for pipeline success

### 3. Custom Metrics

The system creates these custom metrics:
- `custom.googleapis.com/helios/pipeline_execution_time`
- `custom.googleapis.com/helios/trend_opportunity_score`
- `custom.googleapis.com/helios/audience_confidence`
- `custom.googleapis.com/helios/design_generation_success`
- `custom.googleapis.com/helios/publication_success`

### 4. Alerting Policies

Automated alerts for:
- **Pipeline Execution Time** > 5 minutes
- **Success Rate** < 85%
- **Error Rate** > 15%
- **Resource Utilization** > 80%

## 🔧 Configuration

### Required Google Cloud APIs

Ensure these APIs are enabled in your project:
- Cloud Monitoring API
- Cloud Monitoring Dashboards API
- Cloud Monitoring Alerting API

### Service Account Permissions

Your service account needs these roles:
- `roles/monitoring.admin` - For creating dashboards and metrics
- `roles/monitoring.alertPolicyEditor` - For creating alerting policies
- `roles/monitoring.metricWriter` - For writing custom metrics

### Environment Variables

```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

## 📈 Metrics Collection

### Pipeline-Level Metrics

```python
await monitor.record_pipeline_metric(
    execution_time=float,           # Total execution time in seconds
    trend_opportunity_score=float,  # 0.0 to 1.0
    audience_confidence=float,      # 0.0 to 1.0
    design_generation_success=float, # 0.0 to 1.0
    publication_success=float       # 0.0 to 1.0
)
```

### Stage-Level Metrics

```python
await monitor.record_stage_metric(
    stage=str,                      # Stage name (e.g., "trend_discovery")
    duration=float,                 # Stage duration in seconds
    success=bool,                   # Stage success status
    error_count=int,                # Number of errors encountered
    additional_data=dict            # Custom stage-specific data
)
```

## 🔄 Automatic Metric Flushing

Metrics are automatically flushed to Google Cloud Monitoring every 30 seconds. The system:
- Buffers metrics in memory for efficiency
- Batches writes to Cloud Monitoring (200 metrics per batch)
- Handles failures gracefully with retry logic
- Maintains metrics in buffer if flushing fails

## 📊 Performance Analysis

### Real-Time Analysis

```python
# Get current performance summary
summary = await monitor.get_performance_summary()

# Analyze performance trends
analysis = await monitor.analyze_performance()
```

### Analysis Features

- **Success Rate Prediction** - AI-powered success forecasting
- **Optimization Triggers** - Automatic optimization recommendations
- **Performance Trends** - Historical performance analysis
- **Resource Optimization** - Efficiency improvement suggestions

## 🚨 Alerting and Notifications

### Alert Types

1. **Critical Alerts** - Immediate attention required
   - Pipeline execution time > 5 minutes
   - Success rate < 85%

2. **Warning Alerts** - Monitor closely
   - Error rate > 15%
   - Resource utilization > 80%

3. **Info Alerts** - Keep informed
   - New trends discovered
   - Products published successfully

### Alert Channels

Alerts can be configured to send notifications via:
- Email
- Slack
- PagerDuty
- Webhooks
- SMS

## 🧪 Testing and Validation

### 1. Unit Tests

```bash
# Run monitoring unit tests
python -m pytest tests/unit/test_performance_monitor.py
```

### 2. Integration Tests

```bash
# Run monitoring integration tests
python -m pytest tests/integration/test_monitoring_integration.py
```

### 3. Manual Testing

```bash
# Test complete monitoring setup
python test_monitoring.py

# Test via CLI
python -m helios.cli test-monitoring
```

## 🔍 Troubleshooting

### Common Issues

1. **Permission Denied**
   - Verify service account has required roles
   - Check API enablement status

2. **Metrics Not Appearing**
   - Check metric buffer status
   - Verify Cloud Monitoring API quotas
   - Check service account permissions

3. **Dashboard Creation Fails**
   - Verify project ID configuration
   - Check Cloud Monitoring API status
   - Verify dashboard permissions

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

```python
# Check monitoring service health
health = await monitor.check_health()
print(f"Monitoring health: {health}")
```

## 📚 API Reference

### PerformanceMonitor Class

```python
class PerformanceMonitor:
    async def setup_monitoring_infrastructure() -> None
    async def record_pipeline_metric(...) -> None
    async def record_stage_metric(...) -> None
    async def analyze_performance() -> Dict[str, Any]
    async def get_performance_summary() -> Dict[str, Any]
    async def flush_metrics() -> None
```

### CloudMonitoringDashboard Class

```python
class CloudMonitoringDashboard:
    async def setup_complete_monitoring() -> None
    async def create_helios_dashboard() -> None
    async def create_performance_dashboard() -> None
    async def setup_custom_metrics() -> None
    async def create_alerting_policies() -> None
```

## 🚀 Next Steps

1. **Customize Dashboards** - Modify dashboard layouts and widgets
2. **Add Custom Metrics** - Create project-specific metrics
3. **Configure Alerting** - Set up notification channels
4. **Performance Tuning** - Optimize metric collection frequency
5. **Integration** - Connect with external monitoring tools

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Cloud Monitoring logs
3. Verify configuration and permissions
4. Check API quotas and limits

---

**Note**: This monitoring infrastructure is designed to work with Google Cloud Platform. Ensure your project has the necessary APIs enabled and service account permissions configured.
