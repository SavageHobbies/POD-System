# Printify API Integration Setup Guide

## Overview
Your Helios project already has a complete Printify API integration that can:
- Upload designs to Printify
- Create products with custom designs
- Publish products to Etsy via Printify
- Handle batch operations
- Manage pricing and variants

## Required Environment Variables

Create a `.env` file in your project root with:

```bash
# Printify API Configuration
PRINTIFY_API_TOKEN=your_printify_api_token_here
PRINTIFY_SHOP_ID=your_printify_shop_id_here

# Product Configuration (optional - will use defaults if not set)
BLUEPRINT_ID=482
PRINT_PROVIDER_ID=1
DEFAULT_COLORS=white,black
DEFAULT_SIZES=S,M,L,XL,2XL

# Business Logic
DEFAULT_MARGIN=0.5
DEFAULT_DRAFT=true
DRY_RUN=true
```

## How to Get Your Printify Credentials

1. **API Token**: 
   - Go to [Printify Dashboard](https://printify.com/app/dashboard)
   - Navigate to Settings → API Keys
   - Generate a new API key

2. **Shop ID**:
   - In your Printify dashboard, go to Settings → General
   - Your Shop ID is displayed there

## Quick Start: Create and Publish a Product

### Option 1: Use the Full Pipeline (Recommended)
```bash
# Run the complete workflow: trends → ideas → design → publish
python -m helios run --seed "vintage gaming" --num-ideas 3 --draft false --margin 0.6
```

### Option 2: Use Just the Publisher Agent
```python
from helios.agents.publisher_agent import PrintifyPublisherAgent

# Initialize the agent
agent = PrintifyPublisherAgent(
    api_token="your_token_here",
    shop_id="your_shop_id_here"
)

# Create a listing
listing = {
    "title": "Vintage Gaming T-Shirt",
    "description": "Cool vintage gaming design",
    "image_path": "path/to/your/design.png",
    "blueprint_id": 482,  # T-shirt
    "print_provider_id": 1,  # Printify's default provider
    "colors": ["white", "black"],
    "sizes": ["S", "M", "L", "XL"]
}

# Publish to Etsy via Printify
result = agent.run_batch([listing], margin=0.6, draft=False)
print(result)
```

## Product Blueprint IDs

Common product types:
- **T-Shirt**: 482
- **Hoodie**: 484
- **Mug**: 1
- **Poster**: 5

## Features

✅ **Image Upload**: Automatically uploads designs to Printify
✅ **Product Creation**: Creates products with proper variants and pricing
✅ **Etsy Publishing**: Publishes directly to Etsy via Printify
✅ **Batch Processing**: Handle multiple products at once
✅ **Error Handling**: Robust retry logic and error reporting
✅ **Pricing Control**: Automatic margin calculation and .99 pricing
✅ **Variant Management**: Smart color and size variant handling

## Testing

Start with `DRY_RUN=true` to test without actually publishing:
```bash
python -m helios run --seed "test" --num-ideas 1 --dry-run
```

## Monitoring

The system automatically logs to:
- Console output (JSON format)
- `output/run-report-*.json` files
- Google Sheets (if configured)

## Troubleshooting

1. **API Token Issues**: Verify your token has the correct permissions
2. **Shop ID**: Ensure you're using the correct shop ID
3. **Image Format**: Use PNG format for best results
4. **Rate Limits**: The system includes automatic retry logic
5. **Draft Mode**: Start with drafts to test before going live
