#!/usr/bin/env python3
"""
Helios Autonomous Store Orchestrator Startup Script
Run this script to start the automated trend discovery, product generation, and optimization systems.
"""

import asyncio
import sys
from pathlib import Path

# Add the helios package to the path
sys.path.insert(0, str(Path(__file__).parent))

from helios.config import load_config
from helios.services.helios_orchestrator import create_helios_orchestrator


async def main():
    """Main function to run the Helios orchestrator"""
    try:
        print("🚀 Starting Helios Autonomous Store Orchestrator...")
        print("=" * 60)
        
        # Load configuration from .env file
        config = load_config()
        
        print(f"📁 Configuration loaded")
        print(f"🌐 Google Cloud Project: {config.google_cloud_project}")
        print(f"🔒 Dry Run Mode: {config.dry_run}")
        print(f"🚫 Live Publishing: {config.allow_live_publishing}")
        print(f"⚡ Parallel Processing: {config.enable_parallel_processing}")
        print(f"📊 Performance Monitoring: {config.enable_performance_monitoring}")
        print("=" * 60)
        
        # Create and initialize orchestrator
        print("🔧 Initializing Helios services...")
        orchestrator = await create_helios_orchestrator(config)
        
        print("✅ All services initialized successfully!")
        print("=" * 60)
        
        # Ask user for operation mode
        print("Choose operation mode:")
        print("1. Single cycle (run once)")
        print("2. Continuous operation (every 6 hours)")
        print("3. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                
                if choice == "1":
                    print("\n🔄 Running single orchestration cycle...")
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
                    break
                    
                elif choice == "2":
                    print("\n🔄 Starting continuous operation (every 6 hours)...")
                    print("Press Ctrl+C to stop")
                    await orchestrator.start_continuous_operation()
                    break
                    
                elif choice == "3":
                    print("👋 Exiting...")
                    break
                    
                else:
                    print("❌ Invalid choice. Please enter 1, 2, or 3.")
                    
            except KeyboardInterrupt:
                print("\n🛑 Operation interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                break
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        await orchestrator.cleanup()
        print("✅ Cleanup completed")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Startup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        sys.exit(1)
