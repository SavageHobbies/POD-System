# Helios Autonomous Store Orchestrator

The Helios Orchestrator is the central coordination system that automates all three main components of the Helios AI e-commerce platform:

1. **Automated Trend Discovery** (every 6 hours)
2. **Product Generation Pipeline** 
3. **Performance Optimization & A/B Testing**

## 🚀 Quick Start

### Option 1: Interactive Startup Script
```bash
python start_orchestrator.py
```

### Option 2: CLI Command
```bash
# Single cycle
python -m helios.cli orchestrator

# Continuous operation (every 6 hours)
python -m helios.cli orchestrator --continuous

# Dry run mode
python -m helios.cli orchestrator --dry-run
```

### Option 3: Programmatic Usage
```python
from helios.config import load_config
from helios.services.helios_orchestrator import create_helios_orchestrator

async def main():
    config = load_config()
    orchestrator = await create_helios_orchestrator(config)
    
    # Run single cycle
    result = await orchestrator.run_complete_cycle()
    
    # Or start continuous operation
    await orchestrator.start_continuous_operation()
    
    await orchestrator.cleanup()

# Run
asyncio.run(main())
```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Helios Orchestrator                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Trend Discovery │  │Product Pipeline │  │Performance │ │
│  │   (6h cycles)  │  │   (4h cycles)   │  │Optimization│ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│              Google Cloud Infrastructure                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Scheduler  │  │ Firestore  │  │      Redis Cache    │ │
│  │ (Automation)│  │ (Data Store)│  │   (Performance)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Orchestration Cycle

### 1. Trend Discovery Phase (Every 6 Hours)
- **Multi-source trend analysis**: Google Trends, social media, news, competitor intelligence
- **AI-powered validation**: CEO agent evaluates opportunities using configurable thresholds
- **Real-time market data**: Live analysis of trending niches and opportunities
- **Automated scoring**: Composite scoring based on market size, competition, velocity

### 2. Product Generation Phase (Every 4 Hours)
- **Design automation**: AI generates designs for trending niches
- **Ethical screening**: Automatic content validation and copyright review
- **Marketing copy generation**: AI-generated copy for each product
- **Printify integration**: Automated publishing to your store

### 3. Performance Optimization Phase (Daily)
- **A/B testing**: Marketing copy and design variants
- **Analytics**: Sales performance and trend analysis
- **Learning system**: Continuous improvement based on results
- **Traffic optimization**: Dynamic allocation based on performance

## ⚙️ Configuration

The orchestrator uses your existing `.env` file and configuration system:

```yaml
# config/development.yaml
google_cloud_project: "helios-pod-system"
google_cloud_region: "us-central1"
enable_redis_caching: true
enable_performance_monitoring: true
dry_run: true
allow_live_publishing: false
```

### Key Environment Variables
```bash
# Google Cloud
GOOGLE_CLOUD_PROJECT=helios-pod-system
GOOGLE_CLOUD_REGION=us-central1

# Performance thresholds
MIN_OPPORTUNITY_SCORE=7.0
MIN_AUDIENCE_CONFIDENCE=7.0
MIN_PROFIT_MARGIN=0.35

# Automation settings
ENABLE_PARALLEL_PROCESSING=true
ENABLE_BATCH_CREATION=true
ENABLE_ADAPTIVE_LEARNING=true
```

## 📊 Monitoring & Analytics

### Real-time Metrics
- **Execution time**: Per-cycle and cumulative performance
- **Success rates**: Trend discovery, product generation, optimization
- **Resource utilization**: CPU, memory, API calls
- **Business metrics**: Opportunities found, products created, experiments run

### Cloud Monitoring
- **Google Cloud Monitoring**: Custom dashboards and alerts
- **Firestore Analytics**: Data flow and storage metrics
- **Redis Performance**: Caching efficiency and hit rates
- **Scheduler Health**: Automated workflow status

## 🔧 Advanced Features

### Custom Scheduling
```python
# Modify intervals
orchestrator.trend_discovery_interval = timedelta(hours=4)  # More frequent
orchestrator.product_generation_interval = timedelta(hours=2)  # Faster generation
orchestrator.performance_analysis_interval = timedelta(hours=12)  # Twice daily
```

### Seed Keyword Management
```python
# Override default seed keywords
async def custom_seed_keywords():
    return [
        "your-custom-trend",
        "niche-market",
        "seasonal-opportunity"
    ]

orchestrator._get_seed_keywords = custom_seed_keywords
```

### Custom Validation Rules
```python
# Modify opportunity validation
orchestrator.min_opportunity_threshold = 8.0  # Higher bar
orchestrator.min_confidence_threshold = 8.5   # Stricter validation
```

## 🚨 Error Handling & Recovery

### Automatic Retry Logic
- **Failed cycles**: Automatic retry after 1 hour
- **Service failures**: Graceful degradation to fallback modes
- **Data persistence**: All results stored before cleanup
- **Session recovery**: Resume interrupted operations

### Health Checks
```python
# Check system health
summary = await orchestrator.get_orchestration_summary()
print(f"Active sessions: {summary['active_session']}")
print(f"Total sessions: {summary['total_sessions']}")
print(f"Recent errors: {summary.get('recent_errors', [])}")
```

## 📈 Performance Optimization

### Parallel Processing
- **Concurrent execution**: Multiple services run simultaneously
- **Batch operations**: Efficient processing of multiple trends/products
- **Resource pooling**: Shared connections and caches
- **Async I/O**: Non-blocking operations throughout

### Caching Strategy
- **Trend data**: 1-hour TTL for real-time accuracy
- **Product catalog**: 24-hour TTL for stability
- **API responses**: 5-minute TTL for efficiency
- **ML models**: Persistent storage with versioning

## 🔒 Security & Compliance

### Data Protection
- **Encrypted storage**: All data encrypted at rest
- **Access control**: Service account-based authentication
- **Audit logging**: Complete operation history
- **GDPR compliance**: Data retention and deletion policies

### Ethical AI
- **Content screening**: Automatic ethical validation
- **Copyright review**: AI-powered infringement detection
- **Bias detection**: Continuous monitoring for algorithmic bias
- **Transparency**: Explainable AI decisions

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your credentials

# Run orchestrator
python start_orchestrator.py
```

### Google Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy helios-orchestrator \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Docker
```bash
# Build and run
docker build -t helios-orchestrator .
docker run -p 8080:8080 --env-file .env helios-orchestrator
```

## 📚 API Reference

### Core Methods
```python
class HeliosOrchestrator:
    async def initialize_services() -> bool
    async def setup_automated_workflows() -> Dict[str, Any]
    async def run_complete_cycle() -> Dict[str, Any]
    async def start_continuous_operation() -> None
    async def get_orchestration_summary() -> Dict[str, Any]
    async def cleanup()
```

### Data Structures
```python
@dataclass
class OrchestrationSession:
    session_id: str
    start_time: datetime
    trends_discovered: int
    products_generated: int
    experiments_created: int
    execution_time: float
    status: str
```

## 🐛 Troubleshooting

### Common Issues

#### Service Initialization Failures
```bash
# Check Google Cloud credentials
gcloud auth application-default login

# Verify project access
gcloud projects list
gcloud config set project helios-pod-system
```

#### Performance Issues
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check resource usage
summary = await orchestrator.get_orchestration_summary()
print(f"Execution time: {summary['execution_time']:.2f}s")
```

#### Data Storage Issues
```python
# Verify Firestore connection
from helios.services.google_cloud.firestore_client import FirestoreClient
client = FirestoreClient(project_id="helios-pod-system")
await client.test_connection()
```

### Debug Mode
```bash
# Run with verbose logging
python -m helios.cli orchestrator --debug

# Check individual services
python -m helios.cli test-monitoring
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/smoke.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [PHASE1_README.md](PHASE1_README.md)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Wiki**: Project Wiki

---

**🚀 Ready to automate your e-commerce empire? Start the Helios Orchestrator today!**
