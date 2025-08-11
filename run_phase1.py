#!/usr/bin/env python3
"""
Phase 1 Execution Script for Helios Vintage Gaming POD Business
Run this script to execute all Phase 1 components
"""

import asyncio
import sys
from pathlib import Path

# Add helios to path
sys.path.append(str(Path(__file__).parent / "helios"))

from helios.services.phase1_executor import Phase1Executor

async def main():
    """Main execution function"""
    print("🎮 Helios Vintage Gaming POD Business")
    print("🚀 Phase 1: Foundation & Research")
    print("=" * 50)
    
    try:
        # Execute Phase 1
        executor = Phase1Executor()
        results = await executor.execute_phase1()
        
        # Display results summary
        print("\n" + "=" * 50)
        print("📊 PHASE 1 EXECUTION RESULTS")
        print("=" * 50)
        
        components = [
            ("Market Research", "market_research"),
            ("Copyright Review", "copyright_review"), 
            ("Supplier Vetting", "supplier_vetting"),
            ("Ethical Code", "ethical_code")
        ]
        
        for name, key in components:
            status = "✅ SUCCESS" if "error" not in results.get(key, {}) else "❌ FAILED"
            print(f"{name:<20} : {status}")
        
        print("\n📁 All reports saved to: output/phase1/")
        print("📋 Check the summary report for detailed results and next steps")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Phase 1 execution failed: {e}")
        print("Please check your configuration and try again")
        return None

if __name__ == "__main__":
    # Run the async main function
    results = asyncio.run(main())
    
    if results:
        print("\n🎉 Phase 1 completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Phase 1 failed!")
        sys.exit(1)
