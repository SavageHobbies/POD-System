#!/usr/bin/env python3
"""
Test Helios Agent Integration with MCP Server
Shows how the actual agents work with the MCP backend
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_ceo_agent_with_mcp():
    """Test the CEO agent using MCP for decision making"""
    print("🎯 Testing CEO Agent with MCP Integration")
    print("=" * 50)
    
    try:
        from helios.agents.ceo import CEOAgent
        
        # Initialize CEO agent with MCP
        ceo = CEOAgent()
        
        # Test trend analysis
        trend_data = {
            "name": "AI-Powered Personalization",
            "keywords": ["AI", "personalization", "customization", "tech"],
            "trend_score": 9.2,
            "velocity": "exploding"
        }
        
        print(f"📊 Analyzing trend: {trend_data['name']}")
        print(f"🔑 Keywords: {', '.join(trend_data['keywords'])}")
        print(f"📈 Score: {trend_data['trend_score']}/10")
        print(f"🚀 Velocity: {trend_data['velocity']}")
        
        # Use CEO agent to make decision
        decision = await ceo.analyze_trend(trend_data)
        
        print(f"✅ CEO Decision: {decision}")
        return decision
        
    except Exception as e:
        print(f"❌ CEO agent test failed: {e}")
        return None

async def test_trend_agent_with_mcp():
    """Test the trend agent using MCP for analysis"""
    print("\n🔍 Testing Trend Agent with MCP Integration")
    print("=" * 50)
    
    try:
        from helios.agents.zeitgeist import ZeitgeistAgent
        
        # Initialize trend agent
        trend_agent = ZeitgeistAgent()
        
        # Test trend discovery
        seed_term = "sustainable tech"
        print(f"🌱 Seed term: {seed_term}")
        
        # Discover trends
        trends = await trend_agent.discover_trends(seed_term)
        
        print(f"✅ Discovered {len(trends)} trends")
        for i, trend in enumerate(trends[:3]):  # Show first 3
            print(f"   {i+1}. {trend.get('name', 'Unknown')} - Score: {trend.get('score', 'N/A')}")
        
        return trends
        
    except Exception as e:
        print(f"❌ Trend agent test failed: {e}")
        return None

async def test_ethics_agent_with_mcp():
    """Test the ethics agent using MCP for screening"""
    print("\n⚖️ Testing Ethics Agent with MCP Integration")
    print("=" * 50)
    
    try:
        from helios.agents.ethics import EthicsAgent
        
        # Initialize ethics agent
        ethics_agent = EthicsAgent()
        
        # Test ethical screening
        trend_name = "AI-Powered Personalization"
        keywords = ["AI", "personalization", "data", "privacy"]
        
        print(f"📋 Screening trend: {trend_name}")
        print(f"🔑 Keywords: {', '.join(keywords)}")
        
        # Screen for ethics
        screening_result = await ethics_agent.screen_trend(trend_name, keywords)
        
        print(f"✅ Ethical screening completed")
        print(f"   Status: {screening_result.get('status', 'Unknown')}")
        print(f"   Score: {screening_result.get('ethics_score', 'N/A')}")
        
        return screening_result
        
    except Exception as e:
        print(f"❌ Ethics agent test failed: {e}")
        return None

async def test_marketing_agent_with_mcp():
    """Test the marketing agent using MCP for copy generation"""
    print("\n📢 Testing Marketing Agent with MCP Integration")
    print("=" * 50)
    
    try:
        from helios.agents.marketing import MarketingAgent
        
        # Initialize marketing agent
        marketing_agent = MarketingAgent()
        
        # Test marketing copy generation
        product_info = {
            "name": "AI Personalization T-Shirt",
            "style": "Modern tech design with AI elements",
            "target_audience": "Tech-savvy millennials 25-40",
            "price_point": "$34.99",
            "unique_selling_points": ["AI-powered design", "Modern tech aesthetic", "Premium quality", "Limited edition"]
        }
        
        print(f"🎨 Product: {product_info['name']}")
        print(f"🎯 Target: {product_info['target_audience']}")
        print(f"💰 Price: {product_info['price_point']}")
        
        # Generate marketing copy
        marketing_copy = await marketing_agent.generate_copy(product_info)
        
        print(f"✅ Marketing copy generated")
        print(f"   Title: {marketing_copy.get('title', 'N/A')}")
        print(f"   Description: {marketing_copy.get('description', 'N/A')[:100]}...")
        
        return marketing_copy
        
    except Exception as e:
        print(f"❌ Marketing agent test failed: {e}")
        return None

async def test_creative_agent_with_mcp():
    """Test the creative agent using MCP for design ideas"""
    print("\n🎨 Testing Creative Agent with MCP Integration")
    print("=" * 50)
    
    try:
        from helios.agents.creative import CreativeAgent
        
        # Initialize creative agent
        creative_agent = CreativeAgent()
        
        # Test design generation
        design_brief = "Create a modern, tech-inspired design for an AI personalization t-shirt that appeals to tech-savvy millennials"
        
        print(f"🎨 Design brief: {design_brief}")
        
        # Generate design concepts
        design_concepts = await creative_agent.generate_designs(design_brief)
        
        print(f"✅ Design concepts generated")
        print(f"   Concepts: {len(design_concepts)}")
        
        for i, concept in enumerate(design_concepts[:2]):  # Show first 2
            print(f"   {i+1}. {concept.get('name', 'Unknown')} - {concept.get('description', 'N/A')[:80]}...")
        
        return design_concepts
        
    except Exception as e:
        print(f"❌ Creative agent test failed: {e}")
        return None

async def test_full_agent_workflow():
    """Test the complete agent workflow"""
    print("\n🚀 Testing Complete Agent Workflow")
    print("=" * 50)
    
    try:
        # Simulate a complete product creation workflow
        print("🔄 Starting complete workflow...")
        
        # 1. Trend Discovery
        from helios.agents.zeitgeist import ZeitgeistAgent
        trend_agent = ZeitgeistAgent()
        trends = await trend_agent.discover_trends("AI personalization")
        
        if not trends:
            print("❌ No trends discovered, stopping workflow")
            return None
        
        best_trend = trends[0]
        print(f"✅ Best trend: {best_trend.get('name', 'Unknown')}")
        
        # 2. Ethical Screening
        from helios.agents.ethics import EthicsAgent
        ethics_agent = EthicsAgent()
        screening = await ethics_agent.screen_trend(
            best_trend.get('name', 'Unknown'),
            best_trend.get('keywords', [])
        )
        
        if screening.get('status') != 'approved':
            print("❌ Trend failed ethical screening, stopping workflow")
            return None
        
        print(f"✅ Ethical screening passed")
        
        # 3. Business Decision
        from helios.agents.ceo import CEOAgent
        ceo = CEOAgent()
        decision = await ceo.analyze_trend(best_trend)
        
        if decision.get('decision') != 'proceed':
            print("❌ CEO decided not to proceed, stopping workflow")
            return None
        
        print(f"✅ CEO approved: {decision.get('rationale', 'N/A')[:100]}...")
        
        # 4. Creative Design
        from helios.agents.creative import CreativeAgent
        creative_agent = CreativeAgent()
        designs = await creative_agent.generate_designs(
            f"Create a design for {best_trend.get('name', 'Unknown')} trend"
        )
        
        print(f"✅ Designs generated: {len(designs)} concepts")
        
        # 5. Marketing Copy
        from helios.agents.marketing import MarketingAgent
        marketing_agent = MarketingAgent()
        marketing = await marketing_agent.generate_copy({
            "name": f"{best_trend.get('name', 'Trend')} T-Shirt",
            "style": "Modern design based on trend analysis",
            "target_audience": "Trend-aware consumers",
            "price_point": "$29.99"
        })
        
        print(f"✅ Marketing copy generated")
        
        # Workflow Summary
        workflow_result = {
            "trend": best_trend,
            "ethics": screening,
            "decision": decision,
            "designs": designs,
            "marketing": marketing,
            "status": "completed"
        }
        
        print(f"\n🎉 Complete workflow completed successfully!")
        print(f"   Trend: {workflow_result['trend'].get('name', 'Unknown')}")
        print(f"   Ethics: {workflow_result['ethics'].get('status', 'Unknown')}")
        print(f"   Decision: {workflow_result['decision'].get('decision', 'Unknown')}")
        print(f"   Designs: {len(workflow_result['designs'])}")
        print(f"   Marketing: Ready")
        
        return workflow_result
        
    except Exception as e:
        print(f"❌ Complete workflow failed: {e}")
        return None

async def main():
    """Main test function"""
    print("🧪 Helios Agent Integration Test Suite")
    print("=" * 70)
    
    # Test individual agents
    ceo_result = await test_ceo_agent_with_mcp()
    trend_result = await test_trend_agent_with_mcp()
    ethics_result = await test_ethics_agent_with_mcp()
    marketing_result = await test_marketing_agent_with_mcp()
    creative_result = await test_creative_agent_with_mcp()
    
    # Test complete workflow
    workflow_result = await test_full_agent_workflow()
    
    # Final Summary
    print("\n" + "=" * 70)
    print("📋 AGENT INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    agents_tested = [
        ("CEO Agent", ceo_result),
        ("Trend Agent", trend_result),
        ("Ethics Agent", ethics_result),
        ("Marketing Agent", marketing_result),
        ("Creative Agent", creative_result)
    ]
    
    successful_agents = 0
    for agent_name, result in agents_tested:
        if result:
            print(f"✅ {agent_name}: PASSED")
            successful_agents += 1
        else:
            print(f"❌ {agent_name}: FAILED")
    
    print(f"\n📊 Agent Test Results: {successful_agents}/{len(agents_tested)} agents working")
    
    if workflow_result:
        print(f"✅ Complete Workflow: PASSED")
        print(f"   Status: {workflow_result.get('status', 'Unknown')}")
    else:
        print(f"❌ Complete Workflow: FAILED")
    
    overall_success = successful_agents == len(agents_tested) and workflow_result
    print(f"\n🎯 Overall Status: {'READY FOR PRODUCTION' if overall_success else 'NEEDS ATTENTION'}")
    
    if overall_success:
        print(f"\n🚀 Your Helios agents are fully integrated with MCP!")
        print(f"   You can now run complete product creation workflows.")
    else:
        print(f"\n⚠️  Some agents need attention before production use.")

if __name__ == "__main__":
    asyncio.run(main())
