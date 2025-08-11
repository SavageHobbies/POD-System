#!/usr/bin/env python3
"""
Test script for Helios Orchestrator
This script tests the basic functionality without requiring full Google Cloud setup
"""

import asyncio
import sys
from pathlib import Path

# Add the helios package to the path
sys.path.insert(0, str(Path(__file__).parent))

from helios.config import load_config


async def test_config_loading():
    """Test configuration loading"""
    print("🔧 Testing configuration loading...")
    
    try:
        config = load_config()
        print(f"✅ Config loaded successfully")
        print(f"   Project: {config.google_cloud_project}")
        print(f"   Region: {config.google_cloud_region}")
        print(f"   Dry Run: {config.dry_run}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


async def test_imports():
    """Test that all required modules can be imported"""
    print("📦 Testing module imports...")
    
    try:
        # Test core imports
        from helios.services.helios_orchestrator import HeliosOrchestrator, create_helios_orchestrator
        print("✅ Core orchestrator imports successful")
        
        # Test service imports
        from helios.services.automated_trend_discovery import AutomatedTrendDiscovery
        from helios.services.product_generation_pipeline import ProductGenerationPipeline
        from helios.services.performance_optimization import PerformanceOptimizationService
        print("✅ Service imports successful")
        
        # Test cloud service imports
        from helios.services.google_cloud.scheduler_client import CloudSchedulerClient
        from helios.services.google_cloud.firestore_client import FirestoreClient
        print("✅ Cloud service imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False


async def test_orchestrator_creation():
    """Test orchestrator creation (without full initialization)"""
    print("🏗️ Testing orchestrator creation...")
    
    try:
        from helios.services.helios_orchestrator import HeliosOrchestrator
        
        # Create config
        config = load_config()
        
        # Create orchestrator instance (without initializing services)
        orchestrator = HeliosOrchestrator(config)
        print("✅ Orchestrator instance created successfully")
        
        # Test basic properties
        print(f"   Discovery interval: {orchestrator.trend_discovery_interval}")
        print(f"   Generation interval: {orchestrator.product_generation_interval}")
        print(f"   Analysis interval: {orchestrator.performance_analysis_interval}")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator creation failed: {e}")
        return False


async def test_cli_commands():
    """Test CLI command availability"""
    print("🖥️ Testing CLI commands...")
    
    try:
        # Test that CLI can be imported
        from helios.cli import build_parser
        
        # Build parser
        parser = build_parser()
        
        # Check for orchestrator command - be more defensive
        if hasattr(parser, '_subparsers') and parser._subparsers:
            # Get the subparsers action
            subparsers_action = parser._subparsers._group_actions[0] if parser._subparsers._group_actions else None
            if subparsers_action and hasattr(subparsers_action, 'choices'):
                if 'orchestrator' in subparsers_action.choices:
                    print("✅ Orchestrator CLI command found")
                    return True
        
        print("⚠️ Orchestrator CLI command not found")
        return False
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


async def main():
    """Main test function"""
    print("🧪 Helios Orchestrator Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Module Imports", test_imports),
        ("Orchestrator Creation", test_orchestrator_creation),
        ("CLI Commands", test_cli_commands),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The orchestrator is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Ensure your .env file is configured")
        print("   2. Run: python start_orchestrator.py")
        print("   3. Or use CLI: python -m helios.cli orchestrator")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Check your Python environment")
        print("   2. Verify all dependencies are installed")
        print("   3. Check your configuration files")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)
