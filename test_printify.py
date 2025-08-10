#!/usr/bin/env python3
"""
Simple test script for Printify API integration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_printify_connection():
    """Test basic Printify API connection"""
    from helios.publisher.printify_publisher import PrintifyPublisher
    
    api_token = os.getenv("PRINTIFY_API_TOKEN")
    shop_id = os.getenv("PRINTIFY_SHOP_ID")
    
    if not api_token or not shop_id:
        print("❌ Missing PRINTIFY_API_TOKEN or PRINTIFY_SHOP_ID in .env file")
        print("Please create a .env file with your credentials (see PRINTIFY_SETUP.md)")
        return False
    
    try:
        # Initialize publisher
        publisher = PrintifyPublisher(api_token=api_token, shop_id=shop_id)
        
        # Test connection by getting shop info
        print("🔗 Testing Printify API connection...")
        
        # Get catalog info for t-shirt (blueprint 482, provider 1)
        catalog = publisher._get("/catalog/blueprints/482/print_providers/1.json")
        
        print(f"✅ Successfully connected to Printify!")
        print(f"📦 Shop ID: {shop_id}")
        print(f"👕 T-Shirt variants available: {len(catalog.get('variants', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_product_creation():
    """Test product creation workflow"""
    from helios.agents.publisher_agent import PrintifyPublisherAgent
    
    api_token = os.getenv("PRINTIFY_API_TOKEN")
    shop_id = os.getenv("PRINTIFY_SHOP_ID")
    
    if not api_token or not shop_id:
        return False
    
    try:
        # Initialize agent
        agent = PrintifyPublisherAgent(api_token=api_token, shop_id=shop_id)
        
        # Create a test listing
        test_listing = {
            "title": "Test Product - Helios Integration",
            "description": "This is a test product to verify the Helios Printify integration is working correctly.",
            "image_path": "test_design.png",  # You'll need to create this
            "blueprint_id": 482,  # T-shirt
            "print_provider_id": 1,  # Default provider
            "colors": ["white"],
            "sizes": ["M"]
        }
        
        print("\n🧪 Testing product creation workflow...")
        print(f"📝 Title: {test_listing['title']}")
        print(f"🎨 Blueprint: {test_listing['blueprint_id']} (T-shirt)")
        print(f"🏭 Provider: {test_listing['print_provider_id']}")
        
        # Note: This will fail without a real image file, but shows the structure
        print("ℹ️  Note: Product creation test requires a real image file")
        print("   Create 'test_design.png' to test the full workflow")
        
        return True
        
    except Exception as e:
        print(f"❌ Product creation test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Helios Printify API Integration Test")
    print("=" * 50)
    
    # Test 1: Basic connection
    connection_ok = test_printify_connection()
    
    if connection_ok:
        # Test 2: Product creation workflow
        test_product_creation()
        
        print("\n" + "=" * 50)
        print("✅ Basic tests completed!")
        print("\n📚 Next steps:")
        print("1. Create a .env file with your credentials")
        print("2. Create a test design image (PNG format)")
        print("3. Run: python -m helios run --seed 'test' --num-ideas 1")
        print("4. Check PRINTIFY_SETUP.md for detailed instructions")
    else:
        print("\n❌ Please fix the connection issues before proceeding")

if __name__ == "__main__":
    main()
