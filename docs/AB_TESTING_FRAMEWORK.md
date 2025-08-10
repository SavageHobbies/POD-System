# Helios A/B Testing Framework & Adaptive Learning System

## Overview

The Helios Marketing Agent includes a comprehensive A/B Testing Framework and Adaptive Learning System designed to continuously optimize marketing campaigns through data-driven experimentation and machine learning.

## 🚀 A/B Testing Framework

### Core Features

- **Content Variation Testing**: Test different marketing content, headlines, CTAs, and creative elements
- **Performance Comparison**: Comprehensive metrics tracking across all variants
- **Statistical Significance Testing**: Proper statistical analysis using chi-square tests, t-tests, and confidence intervals
- **Traffic Allocation**: Flexible traffic distribution between control and variant groups
- **Real-time Monitoring**: Live tracking of experiment performance and health

### Key Components

#### ABTestExperiment
```python
@dataclass
class ABTestExperiment:
    experiment_id: str
    experiment_name: str
    description: str
    variants: List[ABTestVariant]
    start_date: datetime
    end_date: Optional[datetime]
    status: str  # active, paused, completed, cancelled
    primary_metric: str
    confidence_level: float
    minimum_sample_size: int
    traffic_split: str  # equal, weighted, dynamic
```

#### ABTestVariant
```python
@dataclass
class ABTestVariant:
    variant_id: str
    variant_name: str
    content_variations: Dict[str, Any]
    traffic_allocation: float  # Percentage of traffic (0.0 to 1.0)
    is_control: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
```

#### ABTestResult
```python
@dataclass
class ABTestResult:
    variant_id: str
    variant_name: str
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    revenue: float = 0.0
    engagement_rate: float = 0.0
    conversion_rate: float = 0.0
    ctr: float = 0.0
    cpc: float = 0.0
    cpa: float = 0.0
    roi: float = 0.0
```

### Usage Examples

#### Creating an Experiment
```python
from helios.agents.marketing import ABTestingFramework

# Initialize framework
framework = ABTestingFramework()

# Create experiment
experiment_config = {
    "experiment_name": "Email Subject Line Test",
    "description": "Testing different email subject lines for newsletter",
    "variants": [
        {"variant_name": "Control", "is_control": True, "traffic_allocation": 0.5},
        {"variant_name": "Urgency", "is_control": False, "traffic_allocation": 0.25},
        {"variant_name": "Personalization", "is_control": False, "traffic_allocation": 0.25}
    ],
    "primary_metric": "open_rate",
    "confidence_level": 0.95,
    "minimum_sample_size": 1000
}

experiment = await framework.create_experiment(experiment_config)
```

#### Recording Interactions
```python
# Record user interactions
await framework.record_interaction(
    experiment_id="exp_1",
    variant_id="var_control",
    interaction_type="email_opened",
    value=1.0
)
```

#### Getting Results
```python
# Get comprehensive analytics
analytics = await framework.get_experiment_analytics("exp_1")

# Get winning variant
winner = await framework.get_winning_variant("exp_1")

# Get experiment summary
summary = await framework.get_experiment_summary("exp_1")
```

### Statistical Analysis

The framework automatically performs:

- **Chi-square tests** for conversion rate comparisons
- **T-tests** for continuous metrics like revenue
- **Confidence intervals** for key metrics
- **Statistical power analysis** to determine sample size requirements
- **Effect size calculations** using Cohen's d

## 🧠 Adaptive Learning System

### Core Features

- **Performance-based Parameter Adjustment**: Automatic optimization using gradient descent
- **Historical Data Analysis**: Trend detection and pattern recognition
- **Continuous Optimization**: Real-time parameter updates based on performance
- **Hyperparameter Optimization**: Grid search and Bayesian optimization
- **System Health Monitoring**: Comprehensive health scoring and recommendations

### Key Components

#### LearningParameter
```python
@dataclass
class LearningParameter:
    name: str
    current_value: float
    min_value: float
    max_value: float
    learning_rate: float
    momentum: float = 0.9
    history: List[float] = field(default_factory=list)
```

#### AdaptiveLearningConfig
```python
@dataclass
class AdaptiveLearningConfig:
    learning_rate: float = 0.01
    momentum: float = 0.9
    performance_threshold: float = 0.7
    adaptation_frequency: int = 100
    history_window: int = 1000
    min_confidence: float = 0.8
    exploration_rate: float = 0.1
```

### Usage Examples

#### Initializing the System
```python
from helios.agents.marketing import AdaptiveLearningSystem

# Initialize with default configuration
learning_system = AdaptiveLearningSystem()

# Or with custom configuration
config = AdaptiveLearningConfig(
    learning_rate=0.02,
    momentum=0.95,
    performance_threshold=0.8
)
learning_system = AdaptiveLearningSystem(config)
```

#### Recording Performance
```python
# Record performance data
performance_data = {
    "score": 0.85,
    "timestamp": datetime.utcnow().isoformat(),
    "context": "email_campaign",
    "metrics": {
        "open_rate": 0.25,
        "click_rate": 0.08,
        "conversion_rate": 0.03
    }
}

await learning_system.record_performance(performance_data)
```

#### Getting Optimized Parameters
```python
# Get current optimized parameters
params = await learning_system.get_optimized_parameters()

# Get comprehensive insights
insights = await learning_system.get_learning_insights()

# Get learning summary
summary = await learning_system.get_learning_summary()
```

### Learning Algorithm

The system uses **gradient descent with momentum**:

1. **Parameter Update**: `new_value = current_value + learning_rate * gradient + momentum * previous_change`
2. **Performance Correlation**: Calculates correlation between parameter changes and performance improvements
3. **Adaptive Learning Rate**: Adjusts learning rate based on performance trends
4. **Exploration vs Exploitation**: Balances exploration of new parameter values with exploitation of known good values

## 🔄 Integrated Workflow

### Complete Marketing Optimization Pipeline

1. **Trend Analysis** → Identify opportunities
2. **A/B Test Creation** → Design experiments
3. **Content Generation** → Create variants using AI
4. **Traffic Distribution** → Route users to variants
5. **Performance Tracking** → Monitor real-time metrics
6. **Statistical Analysis** → Determine significance
7. **Learning Integration** → Update parameters
8. **Continuous Optimization** → Iterate and improve

### Example Workflow
```python
# 1. Create A/B test
experiment = await framework.create_experiment(experiment_config)

# 2. Generate content variants
variants = await creative_agent.generate_variants(experiment)

# 3. Run experiment
await framework.start_experiment(experiment.experiment_id)

# 4. Monitor performance
while experiment.status == "active":
    # Record interactions
    await framework.record_interaction(experiment_id, variant_id, "conversion", 1.0)
    
    # Check for statistical significance
    if await framework.is_statistically_significant(experiment_id):
        winner = await framework.get_winning_variant(experiment_id)
        break
    
    await asyncio.sleep(3600)  # Check every hour

# 5. Apply learnings
performance_data = {"score": winner.performance_score}
await learning_system.record_performance(performance_data)

# 6. Optimize for next campaign
optimized_params = await learning_system.get_optimized_parameters()
```

## 📊 Analytics & Reporting

### Experiment Analytics
- **Performance Metrics**: CTR, conversion rates, revenue, ROI
- **Statistical Analysis**: Significance tests, confidence intervals, power analysis
- **Trend Analysis**: Performance improvements over time
- **Health Scoring**: Overall experiment health and recommendations

### Learning Insights
- **Parameter Importance**: Which parameters most affect performance
- **Optimization Recommendations**: Actionable insights for improvement
- **System Health**: Overall learning system status
- **Performance Trends**: Historical performance analysis

### System Overview
```python
# Get comprehensive system status
overview = await framework.get_system_overview()

print(f"Active Experiments: {overview['ab_testing']['active_experiments']}")
print(f"System Health: {overview['adaptive_learning']['system_health']:.2f}")
print(f"Learning Effectiveness: {overview['performance_metrics']['learning_effectiveness']:.2f}")
```

## 🛠️ Advanced Features

### Dynamic Traffic Allocation
- **Equal Split**: Traditional A/B testing
- **Weighted Split**: Custom traffic distribution
- **Dynamic Split**: Real-time optimization based on performance

### Multi-Variant Testing
- **A/B Testing**: Two variants
- **A/B/N Testing**: Multiple variants
- **Multivariate Testing**: Multiple factors simultaneously

### Automated Optimization
- **Early Stopping**: Stop underperforming variants
- **Traffic Reallocation**: Move traffic to better performers
- **Statistical Monitoring**: Continuous significance checking

### Export & Import
```python
# Export experiment data
export_data = await framework.export_experiment_data("exp_1")

# Import learning data
success = await learning_system.import_learning_data(import_data)
```

## 📈 Best Practices

### Experiment Design
1. **Clear Hypothesis**: Define what you're testing and why
2. **Proper Sample Size**: Ensure statistical power
3. **Control Group**: Always include a control variant
4. **Single Variable**: Test one change at a time
5. **Meaningful Metrics**: Focus on business-impacting KPIs

### Learning System
1. **Regular Updates**: Record performance frequently
2. **Parameter Bounds**: Set reasonable min/max values
3. **Learning Rate**: Start conservative, adjust based on stability
4. **Monitoring**: Watch for parameter drift or instability
5. **Validation**: Test parameter changes before deployment

### Statistical Rigor
1. **Confidence Level**: Use 95% or higher for business decisions
2. **Multiple Testing**: Account for multiple comparisons
3. **Effect Size**: Consider practical significance, not just statistical
4. **Duration**: Run tests long enough to capture seasonal effects
5. **Validation**: Replicate results before full deployment

## 🚨 Troubleshooting

### Common Issues

#### Low Statistical Power
- **Symptom**: Experiments never reach significance
- **Solution**: Increase sample size or effect size
- **Check**: Minimum sample size calculations

#### Parameter Instability
- **Symptom**: Parameters oscillate wildly
- **Solution**: Reduce learning rate, increase momentum
- **Check**: Parameter history and variance

#### Poor Performance
- **Symptom**: Learning system not improving
- **Solution**: Check data quality, adjust thresholds
- **Check**: Performance history and trends

### Debugging Tools
```python
# Check experiment health
health_score = await framework.get_experiment_health_score("exp_1")

# Get detailed recommendations
recommendations = await framework.get_experiment_recommendations("exp_1")

# Analyze system health
system_health = await learning_system.get_learning_insights()
```

## 🔮 Future Enhancements

### Planned Features
- **Bayesian Optimization**: More sophisticated parameter search
- **Multi-Objective Optimization**: Balance multiple performance metrics
- **Real-time Learning**: Instant parameter updates
- **Predictive Analytics**: Forecast performance improvements
- **Integration APIs**: Connect with external analytics platforms

### Research Areas
- **Reinforcement Learning**: Advanced optimization algorithms
- **Causal Inference**: Better understanding of cause-effect relationships
- **Personalization**: Individual-level optimization
- **Cross-Platform Learning**: Unified learning across channels

## 📚 Additional Resources

### Documentation
- [Marketing Agent Overview](../README.md)
- [API Reference](../api/README.md)
- [Configuration Guide](../config/README.md)

### Examples
- [Basic A/B Testing](../examples/basic_ab_testing.py)
- [Advanced Learning](../examples/advanced_learning.py)
- [Integration Workflow](../examples/integration_workflow.py)

### Support
- [Issues](https://github.com/your-repo/issues)
- [Discussions](https://github.com/your-repo/discussions)
- [Wiki](https://github.com/your-repo/wiki)

---

**Note**: This framework is designed for production use and includes comprehensive error handling, logging, and monitoring capabilities. Always test thoroughly in development environments before deploying to production.
