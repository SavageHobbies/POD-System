#!/usr/bin/env python3
"""
Fix Printify Integration Issues
Based on analysis of working implementations and our validation errors.
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from helios.config import load_config


class FixedPrintifyPublisher:
    """Fixed Printify Publisher based on working patterns"""
    
    BASE_URL = "https://api.printify.com/v1"
    
    def __init__(self, api_token: str, shop_id: str):
        self.api_token = api_token
        self.shop_id = shop_id
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test basic API connectivity"""
        try:
            url = f"{self.BASE_URL}/shops.json"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return {
                'status': 'success',
                'shops_count': len(data) if isinstance(data, list) else 1,
                'current_shop_id': self.shop_id,
                'api_accessible': True
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'api_accessible': False
            }
    
    def get_blueprints(self, limit: int = 5) -> Dict[str, Any]:
        """Get available blueprints - fixed version"""
        try:
            url = f"{self.BASE_URL}/catalog/blueprints.json"
            params = {"limit": limit}
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, dict) and 'data' in data:
                blueprints = data['data']
            elif isinstance(data, list):
                blueprints = data
            else:
                blueprints = []
            
            return {
                'status': 'success',
                'blueprints': blueprints[:limit],
                'count': len(blueprints)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'blueprints': []
            }
    
    def upload_image_fixed(self, image_path: Path) -> Dict[str, Any]:
        """Fixed image upload based on working patterns"""
        try:
            if not image_path.exists():
                return {
                    'status': 'error',
                    'error': f'Image file not found: {image_path}'
                }
            
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Validate image size (Printify has limits)
            if len(image_data) > 20 * 1024 * 1024:  # 20MB limit
                return {
                    'status': 'error',
                    'error': 'Image too large (>20MB)'
                }
            
            # Encode to base64
            b64_data = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare payload - key difference from our current implementation
            payload = {
                "file_name": image_path.name,
                "contents": b64_data
            }
            
            # Upload with proper headers
            url = f"{self.BASE_URL}/uploads/images.json"
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                url, 
                headers=headers, 
                data=json.dumps(payload),
                timeout=60
            )
            
            if response.status_code != 200:
                return {
                    'status': 'error',
                    'error': f'Upload failed: {response.status_code} - {response.text[:500]}'
                }
            
            result = response.json()
            
            return {
                'status': 'success',
                'upload_id': result.get('id'),
                'file_name': image_path.name,
                'response': result
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def create_simple_product(self, title: str, description: str, upload_id: str, 
                            blueprint_id: int = 145, print_provider_id: int = 29) -> Dict[str, Any]:
        """Create a simple product - working pattern"""
        try:
            # Get variants for the blueprint/provider
            variants_url = f"{self.BASE_URL}/catalog/blueprints/{blueprint_id}/print_providers/{print_provider_id}/variants.json"
            variants_response = requests.get(variants_url, headers=self.headers, timeout=30)
            
            if variants_response.status_code != 200:
                return {
                    'status': 'error',
                    'error': f'Failed to get variants: {variants_response.status_code}'
                }
            
            variants_data = variants_response.json()
            variants = variants_data.get('variants', [])
            
            if not variants:
                return {
                    'status': 'error',
                    'error': 'No variants found for blueprint/provider'
                }
            
            # Use first available variant
            variant = variants[0]
            variant_id = variant.get('id')
            
            # Create product payload
            product_payload = {
                "title": title,
                "description": description,
                "blueprint_id": blueprint_id,
                "print_provider_id": print_provider_id,
                "variants": [
                    {
                        "id": variant_id,
                        "price": 2500,  # $25.00 in cents
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
            
            # Create product
            create_url = f"{self.BASE_URL}/shops/{self.shop_id}/products.json"
            create_response = requests.post(
                create_url,
                headers=self.headers,
                data=json.dumps(product_payload),
                timeout=60
            )
            
            if create_response.status_code != 200:
                return {
                    'status': 'error',
                    'error': f'Product creation failed: {create_response.status_code} - {create_response.text[:500]}'
                }
            
            result = create_response.json()
            
            return {
                'status': 'success',
                'product_id': result.get('id'),
                'title': title,
                'response': result
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


def test_fixed_printify_integration():
    """Test the fixed Printify integration"""
    print("🧪 Testing Fixed Printify Integration")
    print("=" * 50)
    
    # Load config
    config = load_config()
    
    if not config.printify_api_token or not config.printify_shop_id:
        print("❌ Missing Printify credentials in config")
        return False
    
    # Initialize fixed publisher
    publisher = FixedPrintifyPublisher(
        api_token=config.printify_api_token,
        shop_id=config.printify_shop_id
    )
    
    # Test 1: Connection
    print("\n1. Testing API Connection...")
    connection_result = publisher.test_connection()
    if connection_result['status'] == 'success':
        print("✅ API Connection: SUCCESS")
        print(f"   Shops accessible: {connection_result['shops_count']}")
    else:
        print(f"❌ API Connection: FAILED - {connection_result['error']}")
        return False
    
    # Test 2: Blueprints
    print("\n2. Testing Blueprint Access...")
    blueprints_result = publisher.get_blueprints(limit=3)
    if blueprints_result['status'] == 'success':
        print("✅ Blueprints: SUCCESS")
        print(f"   Available blueprints: {blueprints_result['count']}")
    else:
        print(f"❌ Blueprints: FAILED - {blueprints_result['error']}")
        return False
    
    # Test 3: Image Upload (create test image if needed)
    print("\n3. Testing Image Upload...")
    test_image_path = Path("test_upload_image.png")
    
    # Create a simple test image
    try:
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (1000, 1000), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((400, 450), "TEST IMAGE", fill='black')
        img.save(test_image_path)
        print(f"   Created test image: {test_image_path}")
    except ImportError:
        print("❌ PIL not available, skipping image upload test")
        return False
    
    upload_result = publisher.upload_image_fixed(test_image_path)
    if upload_result['status'] == 'success':
        print("✅ Image Upload: SUCCESS")
        print(f"   Upload ID: {upload_result['upload_id']}")
        
        # Test 4: Product Creation (only if upload succeeded)
        print("\n4. Testing Product Creation...")
        product_result = publisher.create_simple_product(
            title="Test Product - Helios Validation",
            description="This is a test product created during Helios validation",
            upload_id=upload_result['upload_id']
        )
        
        if product_result['status'] == 'success':
            print("✅ Product Creation: SUCCESS")
            print(f"   Product ID: {product_result['product_id']}")
        else:
            print(f"❌ Product Creation: FAILED - {product_result['error']}")
            return False
    else:
        print(f"❌ Image Upload: FAILED - {upload_result['error']}")
        return False
    
    # Cleanup
    if test_image_path.exists():
        test_image_path.unlink()
        print(f"   Cleaned up test image: {test_image_path}")
    
    print("\n" + "=" * 50)
    print("🎉 ALL PRINTIFY TESTS PASSED!")
    print("✅ Your Printify integration is working correctly")
    return True


if __name__ == "__main__":
    success = test_fixed_printify_integration()
    sys.exit(0 if success else 1)
