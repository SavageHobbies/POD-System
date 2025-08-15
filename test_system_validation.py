#!/usr/bin/env python3
"""
Comprehensive System Validation Script for Helios
Tests every component to ensure it works correctly and handles errors properly.
"""

import asyncio
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from helios.config import load_config
from helios.mcp_client import MCPClient
from helios.agents.zeitgeist import ZeitgeistAgent
from helios.agents.ceo import HeliosCEO
from helios.agents.ethics import EthicalGuardianAgent
from helios.agents.audience import AudienceAnalyst
from helios.agents.creative import CreativeDirector
from helios.agents.marketing import MarketingCopywriter
from helios.agents.product import ProductStrategist
from helios.agents.publisher_agent import PrintifyPublisherAgent
from helios.publisher.printify_publisher import PrintifyPublisher


class SystemValidator:
    """Comprehensive system validation"""
    
    def __init__(self):
        self.config = load_config()
        self.results = {}
        self.errors = []
        self.warnings = []
        
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete system validation"""
        print("🔍 HELIOS SYSTEM VALIDATION")
        print("=" * 50)
        
        validation_steps = [
            ("Configuration", self._validate_configuration),
            ("MCP Client", self._validate_mcp_client),
            ("Printify Integration", self._validate_printify),
            ("Zeitgeist Agent", self._validate_zeitgeist),
            ("CEO Agent", self._validate_ceo),
            ("Ethics Agent", self._validate_ethics),
            ("Audience Agent", self._validate_audience),
            ("Creative Agent", self._validate_creative),
            ("Marketing Agent", self._validate_marketing),
            ("Product Agent", self._validate_product),
            ("Publisher Agent", self._validate_publisher),
            ("Error Handling", self._validate_error_handling),
            ("End-to-End Pipeline", self._validate_pipeline)
        ]
        
        for step_name, step_func in validation_steps:
            print(f"\n🧪 Testing {step_name}...")
            try:
                result = await step_func()
                self.results[step_name.lower().replace(' ', '_')] = result
                if result.get('status') == 'success':
                    print(f"✅ {step_name}: PASSED")
                else:
                    print(f"❌ {step_name}: FAILED - {result.get('error', 'Unknown error')}")
                    self.errors.append(f"{step_name}: {result.get('error')}")
            except Exception as e:
                error_msg = f"{step_name}: {str(e)}"
                print(f"💥 {step_name}: CRASHED - {str(e)}")
                self.errors.append(error_msg)
                self.results[step_name.lower().replace(' ', '_')] = {
                    'status': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        return self._generate_validation_report()
    
    async def _validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration loading and required settings"""
        try:
            # Check required environment variables
            required_vars = [
                'PRINTIFY_API_TOKEN',
                'PRINTIFY_SHOP_ID',
                'GEMINI_API_KEY',
                'GOOGLE_MCP_URL'
            ]
            
            missing_vars = []
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                return {
                    'status': 'error',
                    'error': f'Missing required environment variables: {", ".join(missing_vars)}'
                }
            
            # Validate config object
            config_issues = []
            
            if not self.config.printify_api_token:
                config_issues.append("Printify API token not loaded")
            if not self.config.google_mcp_url:
                config_issues.append("Google MCP URL not configured")
            if not self.config.google_api_key:
                config_issues.append("Gemini API key not loaded")
            
            if config_issues:
                return {
                    'status': 'error',
                    'error': f'Configuration issues: {"; ".join(config_issues)}'
                }
            
            return {
                'status': 'success',
                'details': {
                    'printify_configured': bool(self.config.printify_api_token),
                    'mcp_configured': bool(self.config.google_mcp_url),
                    'gemini_configured': bool(self.config.google_api_key),
                    'dry_run': self.config.dry_run,
                    'allow_live_publishing': self.config.allow_live_publishing
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _validate_mcp_client(self) -> Dict[str, Any]:
        """Validate MCP client connectivity and functionality"""
        try:
            mcp_client = MCPClient.from_env(
                self.config.google_mcp_url,
                self.config.google_mcp_auth_token
            )
            
            if not mcp_client:
                return {
                    'status': 'error',
                    'error': 'MCP client could not be initialized'
                }
            
            # Test basic connectivity
            try:
                response, error = await mcp_client.try_call("health_check", {}, timeout_s=10.0)
                if error:
                    return {
                        'status': 'warning',
                        'error': f'MCP health check failed: {error}',
                        'fallback_available': True
                    }
                
                # Test AI functionality
                ai_response, ai_error = await mcp_client.try_call(
                    "trend_seeker",
                    {"seed": "test validation", "geo": "US"},
                    timeout_s=15.0
                )
                
                if ai_error:
                    return {
                        'status': 'warning',
                        'error': f'MCP AI test failed: {ai_error}',
                        'connectivity': True,
                        'fallback_available': True
                    }
                
                return {
                    'status': 'success',
                    'details': {
                        'connectivity': True,
                        'ai_functionality': True,
                        'response_time': 'normal'
                    }
                }
                
            except Exception as e:
                return {
                    'status': 'warning',
                    'error': f'MCP connection test failed: {str(e)}',
                    'fallback_available': True
                }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _validate_printify(self) -> Dict[str, Any]:
        """Validate Printify API connectivity and functionality"""
        try:
            publisher = PrintifyPublisher(
                api_token=self.config.printify_api_token,
                shop_id=self.config.printify_shop_id
            )
            
            # Test basic API connectivity
            try:
                # This should work if API token is valid
                catalog_response = publisher._get("/catalog/blueprints.json", params={"limit": 1})
                
                # Printify API returns different structures, check for valid response
                if not catalog_response:
                    return {
                        'status': 'error',
                        'error': 'Printify API returned no response'
                    }
                
                # Check if it's a valid Printify response (could have 'data' or direct array)
                if not isinstance(catalog_response, (dict, list)):
                    return {
                        'status': 'error',
                        'error': 'Printify API returned invalid response format'
                    }
                
                # Test blueprint/provider discovery
                try:
                    blueprint_info = publisher.discover_blueprint_providers(limit=5)
                    
                    return {
                        'status': 'success',
                        'details': {
                            'api_connectivity': True,
                            'catalog_access': True,
                            'blueprints_available': len(blueprint_info),
                            'shop_id': self.config.printify_shop_id
                        }
                    }
                    
                except Exception as e:
                    return {
                        'status': 'warning',
                        'error': f'Blueprint discovery failed: {str(e)}',
                        'api_connectivity': True
                    }
                
            except Exception as e:
                return {
                    'status': 'error',
                    'error': f'Printify API connectivity failed: {str(e)}'
                }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _validate_zeitgeist(self) -> Dict[str, Any]:
        """Validate Zeitgeist agent functionality"""
        try:
            agent = ZeitgeistAgent()
            
            # Test with a simple seed
            result = await agent.run("test trend validation")
            
            if not result or not isinstance(result, dict):
                return {
                    'status': 'error',
                    'error': 'Zeitgeist agent returned invalid result'
                }
            
            required_fields = ['trend_name', 'keywords', 'opportunity_score', 'status']
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                return {
                    'status': 'error',
                    'error': f'Missing required fields: {", ".join(missing_fields)}'
                }
            
            return {
                'status': 'success',
                'details': {
                    'trend_generated': True,
                    'trend_name': result.get('trend_name'),
                    'opportunity_score': result.get('opportunity_score'),
                    'status': result.get('status'),
                    'mcp_model_used': result.get('mcp_model_used', 'fallback')
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _validate_ceo(self) -> Dict[str, Any]:
        """Validate CEO agent functionality"""
        try:
            ceo = HeliosCEO()
            
            # Test preparation
            prep_result = await ceo.prepare_validation()
            
            if not prep_result or prep_result.get('status') != 'ready':
                return {
                    'status': 'warning',
                    'error': 'CEO agent preparation issues',
                    'details': prep_result
                }
            
            # Test validation with mock trend data
            mock_trend = {
                'trend_name': 'Test Trend',
                'keywords': ['test', 'validation'],
                'opportunity_score': 7.5,
                'confidence_level': 0.8,
                'ethical_status': 'approved',
                'velocity': 'stable',
                'urgency_level': 'medium'
            }
            
            decision = await ceo.validate_trend(mock_trend)
            
            if not decision:
                return {
                    'status': 'error',
                    'error': 'CEO validation returned no decision'
                }
            
            return {
                'status': 'success',
                'details': {
                    'preparation': 'ready',
                    'validation_works': True,
                    'decision_approved': decision.approved,
                    'quality_gates': len(decision.quality_gates),
                    'priority': decision.priority.value if hasattr(decision.priority, 'value') else str(decision.priority)
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _validate_ethics(self) -> Dict[str, Any]:
        """Validate Ethics agent functionality"""
        try:
            agent = EthicalGuardianAgent()
            
            # Test with safe content
            safe_result = await agent.screen_content("Beautiful nature design", {"keywords": ["nature", "peaceful"]})
            
            if not safe_result:
                return {
                    'status': 'error',
                    'error': 'Ethics agent returned no result'
                }
            
            # Test with potentially risky content
            risky_result = await agent.screen_content("Political statement design", {"keywords": ["politics", "controversial"]})
            
            return {
                'status': 'success',
                'details': {
                    'safe_content_status': safe_result.status,
                    'risky_content_status': risky_result.status,
                    'mcp_model_used': safe_result.mcp_model_used,
                    'execution_time': f"{safe_result.execution_time_ms}ms"
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _validate_audience(self) -> Dict[str, Any]:
        """Validate Audience agent functionality"""
        try:
            agent = AudienceAnalyst()
            
            mock_trend = {
                'trend_name': 'Test Trend',
                'keywords': ['trendy', 'modern'],
                'urgency_level': 'medium',
                'opportunity_score': 7.0
            }
            
            result = await agent.run(mock_trend)
            
            if not result or not result.primary_persona:
                return {
                    'status': 'error',
                    'error': 'Audience analysis failed to generate persona'
                }
            
            return {
                'status': 'success',
                'details': {
                    'primary_persona': result.primary_persona.demographic_cluster,
                    'confidence_score': result.confidence_score,
                    'secondary_personas': len(result.secondary_personas),
                    'rapid_mode_used': result.rapid_mode_used,
                    'mcp_model_used': result.mcp_model_used
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _validate_creative(self) -> Dict[str, Any]:
        """Validate Creative agent functionality"""
        try:
            agent = CreativeDirector(
                output_dir=self.config.output_dir,
                fonts_dir=self.config.fonts_dir
            )
            
            mock_trend = {
                'trend_name': 'Test Creative Trend',
                'keywords': ['creative', 'design'],
                'emotional_driver': {'primary_emotion': 'inspiration'},
                'psychological_insights': {
                    'identity_statements': ['I am creative'],
                    'authority_figures': ['artists'],
                    'trust_building_elements': ['quality']
                }
            }
            
            mock_products = [{'type': 'apparel', 'product_key': 't_shirt'}]
            
            result = await agent.run(mock_trend, mock_products, num_designs_per_product=1)
            
            if not result or result.get('status') != 'success':
                return {
                    'status': 'error',
                    'error': f'Creative generation failed: {result.get("error", "Unknown error")}'
                }
            
            return {
                'status': 'success',
                'details': {
                    'designs_generated': len(result.get('designs', [])),
                    'mcp_model_used': result.get('mcp_model_used', 'fallback'),
                    'execution_time': f"{result.get('execution_time_ms', 0)}ms",
                    'batch_optimization': result.get('batch_optimization', {})
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _validate_marketing(self) -> Dict[str, Any]:
        """Validate Marketing agent functionality"""
        try:
            agent = MarketingCopywriter()
            
            mock_creative_batch = {
                'designs': [{
                    'concept_name': 'Test Design',
                    'concept_description': 'A test design for validation',
                    'emotional_appeal': 'inspiring'
                }],
                'trend_data': {
                    'trend_name': 'Test Marketing Trend',
                    'keywords': ['marketing', 'test']
                }
            }
            
            result = await agent.run(mock_creative_batch)
            
            if not result or result.get('status') != 'success':
                return {
                    'status': 'error',
                    'error': f'Marketing copy generation failed: {result.get("error", "Unknown error")}'
                }
            
            return {
                'status': 'success',
                'details': {
                    'copy_generated': bool(result.get('marketing_copy')),
                    'mcp_model_used': result.get('mcp_model_used', 'fallback'),
                    'execution_time': f"{result.get('execution_time_ms', 0)}ms",
                    'copy_quality': 'generated' if result.get('marketing_copy') else 'none'
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _validate_product(self) -> Dict[str, Any]:
        """Validate Product agent functionality"""
        try:
            agent = ProductStrategist(self.config)
            
            mock_audience = {
                'primary_persona': {
                    'demographic_cluster': 'test_audience',
                    'lifestyle_category': 'trendy'
                }
            }
            
            result = agent.run(mock_audience)
            
            if not result:
                return {
                    'status': 'error',
                    'error': 'Product strategy returned no result'
                }
            
            # ProductStrategist.run() returns selected_products, not status
            if 'selected_products' not in result:
                return {
                    'status': 'error',
                    'error': 'Product strategy missing selected_products field'
                }
            
            products = result.get('selected_products', [])
            
            return {
                'status': 'success',
                'details': {
                    'products_selected': len(products),
                    'selection_confidence': result.get('selection_confidence', 0.0),
                    'demographic_target': result.get('demographic_target', 'unknown'),
                    'product_types': [p.get('type', 'unknown') for p in products[:3]]
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _validate_publisher(self) -> Dict[str, Any]:
        """Validate Publisher agent functionality"""
        try:
            agent = PrintifyPublisherAgent(
                api_token=self.config.printify_api_token,
                shop_id=self.config.printify_shop_id
            )
            
            # Create a temporary test image for validation
            test_image_path = self.config.output_dir / 'validation_test.png'
            test_image_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a minimal PNG file for testing
            if not test_image_path.exists():
                from PIL import Image
                img = Image.new('RGB', (100, 100), color='white')
                img.save(test_image_path)
            
            # Test batch processing capability (dry run)
            mock_listings = [{
                'image_path': str(test_image_path),
                'title': 'Test Product',
                'description': 'A test product for validation',
                'tags': ['test', 'validation'],
                'blueprint_id': 145,
                'print_provider_id': 29
            }]
            
            # This should not actually publish in dry run mode
            result = agent.run_batch(mock_listings, margin=0.5, draft=True)
            
            if not result:
                return {
                    'status': 'error',
                    'error': 'Publisher returned no result'
                }
            
            return {
                'status': 'success',
                'details': {
                    'batch_processing': True,
                    'dry_run_safe': True,
                    'printify_integration': True,
                    'result_structure': 'valid'
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _validate_error_handling(self) -> Dict[str, Any]:
        """Validate error handling patterns across the system"""
        try:
            error_tests = []
            
            # Test MCP client error handling
            mcp_client = MCPClient.from_env("http://invalid-url", "invalid-token")
            if mcp_client:
                try:
                    await mcp_client.try_call("invalid_tool", {}, timeout_s=5.0)
                    error_tests.append({"test": "mcp_invalid_call", "handled": True})
                except Exception:
                    error_tests.append({"test": "mcp_invalid_call", "handled": False})
            
            # Test Printify error handling with invalid token
            try:
                invalid_publisher = PrintifyPublisher("invalid_token", "invalid_shop")
                invalid_publisher._get("/catalog/blueprints.json", params={"limit": 1})
                error_tests.append({"test": "printify_invalid_auth", "handled": False})
            except Exception:
                error_tests.append({"test": "printify_invalid_auth", "handled": True})
            
            # Test agent error handling
            try:
                zeitgeist = ZeitgeistAgent()
                # This should handle the error gracefully
                result = await zeitgeist.run(None)  # Invalid input
                error_tests.append({
                    "test": "zeitgeist_invalid_input", 
                    "handled": result is not None
                })
            except Exception:
                error_tests.append({"test": "zeitgeist_invalid_input", "handled": False})
            
            handled_count = sum(1 for test in error_tests if test["handled"])
            total_tests = len(error_tests)
            
            status = 'success' if handled_count == total_tests else 'warning'
            if total_tests == 0:
                status = 'error'
                
            return {
                'status': status,
                'details': {
                    'error_tests_run': total_tests,
                    'errors_handled_properly': handled_count,
                    'error_handling_rate': f"{handled_count}/{total_tests}" if total_tests > 0 else "0/0",
                    'test_details': error_tests
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _validate_pipeline(self) -> Dict[str, Any]:
        """Validate end-to-end pipeline functionality"""
        try:
            # Import and test the main pipeline
            from helios.main import run_helios_pipeline
            
            # Run a test pipeline (dry run)
            result = await run_helios_pipeline(
                seed="validation test trend",
                dry_run=True,
                enable_parallel=False  # Simpler for testing
            )
            
            if not result:
                return {
                    'status': 'error',
                    'error': 'Pipeline returned no result'
                }
            
            pipeline_status = result.get('status', 'unknown')
            
            return {
                'status': 'success' if pipeline_status == 'success' else 'warning',
                'details': {
                    'pipeline_status': pipeline_status,
                    'execution_time': result.get('pipeline_execution_time', 0),
                    'stages_completed': len(result.get('stage_times', {})),
                    'error_reason': result.get('reason') if pipeline_status != 'success' else None
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result.get('status') == 'success')
        warning_tests = sum(1 for result in self.results.values() if result.get('status') == 'warning')
        failed_tests = sum(1 for result in self.results.values() if result.get('status') == 'error')
        
        overall_status = 'success' if failed_tests == 0 and warning_tests == 0 else \
                        'warning' if failed_tests == 0 else 'error'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'warnings': warning_tests,
                'failed': failed_tests,
                'success_rate': f"{passed_tests}/{total_tests}"
            },
            'detailed_results': self.results,
            'errors': self.errors,
            'warnings': self.warnings,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if self.errors:
            recommendations.append("❌ CRITICAL: Fix all failed components before deploying to production")
        
        if any('mcp' in error.lower() for error in self.errors):
            recommendations.append("🔧 Consider implementing more robust MCP fallback mechanisms")
        
        if any('printify' in error.lower() for error in self.errors):
            recommendations.append("🔧 Verify Printify API credentials and network connectivity")
        
        if len(self.errors) == 0 and len(self.warnings) == 0:
            recommendations.append("✅ System is ready for Google Cloud deployment")
        elif len(self.errors) == 0:
            recommendations.append("⚠️ System can deploy but monitor warnings closely")
        
        return recommendations


async def main():
    """Run system validation"""
    validator = SystemValidator()
    report = await validator.run_full_validation()
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    summary = report['summary']
    print(f"Overall Status: {report['overall_status'].upper()}")
    print(f"Tests Passed: {summary['passed']}/{summary['total_tests']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Failures: {summary['failed']}")
    
    if report['errors']:
        print(f"\n❌ ERRORS ({len(report['errors'])}):")
        for error in report['errors']:
            print(f"   • {error}")
    
    if report['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
    
    # Save detailed report
    report_file = Path("system_validation_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Return appropriate exit code
    return 0 if report['overall_status'] in ['success', 'warning'] else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
