#!/usr/bin/env python3
"""
Example: Create and publish a product to Etsy via Printify API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_single_product():
    """Create a single product and publish it to Etsy"""
    from helios.agents.publisher_agent import PrintifyPublisherAgent
    
    # Get credentials from environment
    api_token = os.getenv("PRINTIFY_API_TOKEN")
    shop_id = os.getenv("PRINTIFY_SHOP_ID")
    
    if not api_token or not shop_id:
        print("❌ Missing credentials. Please create a .env file with:")
        print("   PRINTIFY_API_TOKEN=your_token")
        print("   PRINTIFY_SHOP_ID=your_shop_id")
        return
    
    # Initialize the publisher agent
    agent = PrintifyPublisherAgent(api_token=api_token, shop_id=shop_id)
    
    # Define your product
    product_listing = {
        "title": "Vintage Gaming T-Shirt - Helios Test",
        "description": "A cool vintage gaming design created with Helios automation system. Perfect for gamers who love retro aesthetics.",
        "image_path": "assets/test_design.png",  # Path to your design file
        "blueprint_id": 482,  # T-shirt blueprint
        "print_provider_id": 1,  # Default print provider
        "colors": ["white", "black"],  # Available colors
        "sizes": ["S", "M", "L", "XL", "2XL"]  # Available sizes
    }
    
    print("🚀 Creating product via Printify API...")
    print(f"📝 Title: {product_listing['title']}")
    print(f"🎨 Design: {product_listing['image_path']}")
    print(f"👕 Product Type: T-Shirt (Blueprint {product_listing['blueprint_id']})")
    print(f"🏭 Provider: {product_listing['print_provider_id']}")
    print(f"🎨 Colors: {', '.join(product_listing['colors'])}")
    print(f"📏 Sizes: {', '.join(product_listing['sizes'])}")
    
    try:
        # Create and publish the product
        # Set margin to 60% (0.6) and publish as draft first
        result = agent.run_batch(
            listings=[product_listing], 
            margin=0.6,  # 60% profit margin
            draft=True   # Start as draft for safety
        )
        
        print("\n✅ Product creation completed!")
        print("📊 Results:")
        
        for product_result in result.get("publication_results", []):
            print(f"  • {product_result['product_title']}")
            print(f"    Status: {product_result['status']}")
            print(f"    Printify ID: {product_result['printify_product_id']}")
            print(f"    Image ID: {product_result['image_upload_id']}")
            print(f"    Price: ${product_result['final_price']:.2f}")
            
            if product_result['error_details']:
                print(f"    ❌ Error: {product_result['error_details']}")
        
        # Summary
        summary = result.get("batch_summary", {})
        print(f"\n📈 Batch Summary:")
        print(f"  Total processed: {summary.get('total_processed', 0)}")
        print(f"  Successful: {summary.get('successful_publications', 0)}")
        print(f"  Failed: {summary.get('failed_publications', 0)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Product creation failed: {e}")
        return None

def create_multiple_products():
    """Create multiple products in a batch"""
    from helios.agents.publisher_agent import PrintifyPublisherAgent
    
    api_token = os.getenv("PRINTIFY_API_TOKEN")
    shop_id = os.getenv("PRINTIFY_SHOP_ID")
    
    if not api_token or not shop_id:
        return
    
    agent = PrintifyPublisherAgent(api_token=api_token, shop_id=shop_id)
    
    # Multiple product listings
    product_listings = [
        {
            "title": "Retro Gaming T-Shirt - Classic Edition",
            "description": "Classic retro gaming design with vintage appeal",
            "image_path": "assets/retro_gaming.png",
            "blueprint_id": 482,
            "print_provider_id": 1,
            "colors": ["white", "black"],
            "sizes": ["S", "M", "L", "XL"]
        },
        {
            "title": "Pixel Art Gaming T-Shirt - Modern Style",
            "description": "Modern pixel art gaming design for contemporary gamers",
            "image_path": "assets/pixel_art.png",
            "blueprint_id": 482,
            "print_provider_id": 1,
            "colors": ["white", "navy"],
            "sizes": ["M", "L", "XL", "2XL"]
        }
    ]
    
    print("🚀 Creating multiple products...")
    
    try:
        result = agent.run_batch(
            listings=product_listings,
            margin=0.65,  # 65% profit margin
            draft=True    # Start as drafts
        )
        
        print("✅ Batch creation completed!")
        return result
        
    except Exception as e:
        print(f"❌ Batch creation failed: {e}")
        return None

def main():
    """Main function"""
    print("🎯 Helios Printify Product Creation Example")
    print("=" * 60)
    
    print("Choose an option:")
    print("1. Create a single product")
    print("2. Create multiple products")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        create_single_product()
    elif choice == "2":
        create_multiple_products()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
