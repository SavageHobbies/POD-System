#!/usr/bin/env python3
"""
Fix Blueprint/Variant Resolution Issue
Based on IncomeStreamSurfer's working approach
"""

import os
import requests
import json
from typing import Dict, List, Any, Optional

def get_printify_shops(api_token: str) -> List[Dict[str, Any]]:
    """Get all available Printify shops"""
    print("🏪 Getting Printify shops...")
    
    url = "https://api.printify.com/v1/shops.json"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        shops = response.json()
        print(f"✅ Found {len(shops)} shops")
        for shop in shops:
            print(f"   Shop ID: {shop['id']} - {shop['title']}")
        return shops
    else:
        print(f"❌ Failed to get shops: {response.text}")
        return []

def get_blueprint_variants(api_token: str, blueprint_id: int, provider_id: int) -> Dict[str, Any]:
    """Get variants for a specific blueprint and provider combination"""
    print(f"🔍 Getting variants for blueprint {blueprint_id} + provider {provider_id}...")
    
    url = f"https://api.printify.com/v1/catalog/blueprints/{blueprint_id}/print_providers/{provider_id}/variants.json"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        variants_data = response.json()
        variants = variants_data.get('variants', [])
        print(f"✅ Found {len(variants)} variants")
        
        # Show first few variants for reference
        for i, variant in enumerate(variants[:3]):
            price = variant.get('price', 0)
            print(f"   Variant {variant['id']}: {variant['title']} - ${price/100:.2f}" if price else f"   Variant {variant['id']}: {variant['title']}")
            
        return variants_data
    else:
        print(f"❌ Failed to get variants: {response.text}")
        return {}

def test_product_creation(api_token: str, shop_id: str, upload_id: str) -> Optional[str]:
    """Test product creation with proper variant resolution"""
    print(f"🛒 Testing product creation...")
    
    # Use the working configuration from IncomeStreamSurfer's approach
    blueprint_id = 145  # Your current blueprint
    provider_id = 29    # Your current provider
    
    # First, get valid variants for this combination
    variants_data = get_blueprint_variants(api_token, blueprint_id, provider_id)
    if not variants_data or not variants_data.get('variants'):
        print("❌ No variants found - blueprint/provider combination may be invalid")
        return None
    
    # Use the first available variant (following IncomeStreamSurfer's pattern)
    available_variants = variants_data['variants']
    first_variant = available_variants[0]
    variant_id = first_variant['id']
    
    variant_price = first_variant.get('price', 1999)  # Default to $19.99 if no price
    print(f"📦 Using variant: {variant_id} - {first_variant['title']} (${variant_price/100:.2f})")
    
    # Create product payload following their exact structure
    url = f"https://api.printify.com/v1/shops/{shop_id}/products.json"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Product payload based on working patterns
    payload = {
        "title": "Test Minimalist Quote Design",
        "description": "A test product with clean minimalist design",
        "blueprint_id": blueprint_id,
        "print_provider_id": provider_id,
        "variants": [
            {
                "id": variant_id,
                "price": variant_price + 500,  # Add $5 markup
                "is_enabled": True
            }
        ],
        "print_areas": [
            {
                "variant_ids": [variant_id],
                "placeholders": [
                    {
                        "position": "front",
                        "images": [
                            {
                                "id": upload_id,
                                "x": 0.5,
                                "y": 0.5,
                                "scale": 1.0,
                                "angle": 0
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    print(f"📤 Creating product with variant {variant_id}...")
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        product = response.json()
        product_id = product['id']
        print(f"✅ Product created successfully!")
        print(f"   Product ID: {product_id}")
        print(f"   Title: {product['title']}")
        print(f"   Variants: {len(product['variants'])}")
        return product_id
    else:
        print(f"❌ Product creation failed: {response.text[:500]}")
        return None

def main():
    """Main function to test and fix blueprint/variant issues"""
    print("🔧 FIXING BLUEPRINT/VARIANT RESOLUTION ISSUE")
    print("=" * 60)
    
    # Load API credentials
    api_token = os.getenv("PRINTIFY_API_TOKEN")
    if not api_token:
        print("❌ PRINTIFY_API_TOKEN not found in environment")
        return
    
    shop_id = os.getenv("PRINTIFY_SHOP_ID", "8542090")  # Your shop ID
    
    print(f"🏪 Using shop ID: {shop_id}")
    print()
    
    # Step 1: Verify shop access
    shops = get_printify_shops(api_token)
    if not shops:
        return
    
    # Verify our shop ID is valid
    shop_ids = [str(shop['id']) for shop in shops]
    if shop_id not in shop_ids:
        print(f"❌ Shop ID {shop_id} not found in available shops: {shop_ids}")
        return
    
    print(f"✅ Shop ID {shop_id} confirmed")
    print()
    
    # Step 2: Test blueprint/variant combinations
    blueprint_id = 145
    provider_id = 29
    
    print(f"🧪 Testing blueprint {blueprint_id} + provider {provider_id}...")
    variants_data = get_blueprint_variants(api_token, blueprint_id, provider_id)
    
    if not variants_data or not variants_data.get('variants'):
        print("❌ This blueprint/provider combination has no available variants")
        print("💡 Try different blueprint_id or print_provider_id values")
        
        # Suggest alternatives
        print("\n🔍 Checking alternative providers for this blueprint...")
        blueprint_url = f"https://api.printify.com/v1/catalog/blueprints/{blueprint_id}.json"
        headers = {"Authorization": f"Bearer {api_token}"}
        response = requests.get(blueprint_url, headers=headers)
        
        if response.status_code == 200:
            blueprint_data = response.json()
            print_providers = blueprint_data.get('print_providers', [])
            print(f"📋 Available providers for blueprint {blueprint_id}:")
            for provider in print_providers[:5]:  # Show first 5
                print(f"   Provider ID: {provider['id']} - {provider['title']}")
        return
    
    print(f"✅ Blueprint/provider combination is valid!")
    print()
    
    # Step 3: Test with actual uploaded image
    upload_id = "689ee38d2596045f0dade42b"  # Our successful upload
    product_id = test_product_creation(api_token, shop_id, upload_id)
    
    if product_id:
        print(f"\n🎉 SUCCESS! Product creation working!")
        print(f"📋 Product ID: {product_id}")
        print(f"🔗 Check your Printify dashboard to see the product")
        
        # Output the working configuration
        print(f"\n⚙️  WORKING CONFIGURATION:")
        print(f"   Blueprint ID: {blueprint_id}")
        print(f"   Provider ID: {provider_id}")
        print(f"   First variant ID: {variants_data['variants'][0]['id']}")
    else:
        print(f"\n💥 Product creation still failing - need to investigate further")

if __name__ == "__main__":
    main()
