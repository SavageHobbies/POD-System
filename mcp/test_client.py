#!/usr/bin/env python3
"""
Test client for Helios MCP Server
Tests all available tools and endpoints
"""

import asyncio
import json
import time
import httpx
from typing import Dict, Any

BASE_URL = "http://localhost:8080"

async def test_health():
    """Test health endpoint"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/health")
            print(f"✅ Health check: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Services: {data.get('services', {})}")
                print(f"   Models: {data.get('models', {})}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False

async def test_tool(tool_name: str, parameters: Dict[str, Any]) -> bool:
    """Test a specific MCP tool"""
    async with httpx.AsyncClient() as client:
        try:
            payload = {
                "tool": tool_name,
                "parameters": parameters
            }
            
            start_time = time.time()
            response = await client.post(
                f"{BASE_URL}/execute",
                json=payload,
                timeout=30.0
            )
            execution_time = (time.time() - start_time) * 1000
            
            print(f"🔧 {tool_name}: {response.status_code} ({execution_time:.0f}ms)")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Status: {data.get('status')}")
                print(f"   Model: {data.get('model')}")
                print(f"   Response: {data.get('response', '')[:100]}...")
                return True
            else:
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ {tool_name} failed: {e}")
            return False

async def run_tests():
    """Run all tests"""
    print("🧪 Testing Helios MCP Server...")
    print("=" * 50)
    
    # Test health endpoint
    if not await test_health():
        print("❌ Server not healthy, stopping tests")
        return
    
    print("\n🔧 Testing MCP Tools...")
    print("-" * 30)
    
    # Test AI tools
    ai_tools = [
        ("orchestrator_ai", {"prompt": "Analyze the current market for sustainable fashion trends"}),
        ("trend_seeker", {"seed": "sustainable fashion", "geo": "US"}),
        ("ethics_ai", {"trend_name": "eco-friendly products", "keywords": ["green", "sustainable", "eco"]}),
        ("marketing_ai", {"product_info": {"name": "Eco T-Shirt", "style": "Modern", "target_audience": "Eco-conscious millennials", "keywords": ["sustainable", "fashion", "eco"]}}),
        ("creative_ai", {"design_brief": "Create a modern, minimalist design for an eco-friendly t-shirt"}),
    ]
    
    # Test Google services
    google_tools = [
        ("google_trends_keywords", {"geo": "US", "top_n": 5}),
        ("social_media_scanner", {"seed": "sustainable fashion"}),
    ]
    
    # Test all tools
    all_tools = ai_tools + google_tools
    
    results = []
    for tool_name, params in all_tools:
        success = await test_tool(tool_name, params)
        results.append((tool_name, success))
        await asyncio.sleep(1)  # Rate limiting
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for tool_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {tool_name}")
    
    print(f"\nOverall: {successful}/{total} tools working")
    
    if successful == total:
        print("🎉 All tests passed! Your MCP server is ready.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    asyncio.run(run_tests())
