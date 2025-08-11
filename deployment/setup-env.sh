#!/bin/bash

# Environment Setup Script for Helios POD Business Deployment
# This script sets up all required environment variables

echo "🔧 Setting up environment variables for Helios POD Business deployment..."

# Google Cloud API Key (from the API key we just created)
export GOOGLE_CLOUD_API_KEY="AIzaSyCt55007xEJgG914f704R1iwE4vEsDrL04"

# Vertex AI Service Account Key (base64 encoded)
export VERTEX_AI_SA_KEY=$(base64 -i /tmp/helios-vertex-ai-key.json)

# Generate a simple MCP auth token (you can change this to something more secure)
export GOOGLE_MCP_AUTH_TOKEN="helios-mcp-auth-$(date +%s)"

# Printify API Token - YOU NEED TO SET THIS
echo "⚠️  IMPORTANT: You need to set your Printify API token manually!"
echo "   Get it from: https://printify.com/app/settings/api-keys"
echo "   Then run: export PRINTIFY_API_TOKEN='your-token-here'"
echo ""

# Display current environment variables
echo "✅ Environment variables set:"
echo "   GOOGLE_CLOUD_API_KEY: ${GOOGLE_CLOUD_API_KEY:0:20}..."
echo "   VERTEX_AI_SA_KEY: ${VERTEX_AI_SA_KEY:0:20}..."
echo "   GOOGLE_MCP_AUTH_TOKEN: ${GOOGLE_MCP_AUTH_TOKEN}"
echo "   PRINTIFY_API_TOKEN: [NOT SET - YOU NEED TO SET THIS]"
echo ""

# Save to a file for easy sourcing
cat > deployment/.env.deployment << EOF
# Deployment Environment Variables
# Source this file with: source deployment/.env.deployment

export GOOGLE_CLOUD_API_KEY="${GOOGLE_CLOUD_API_KEY}"
export VERTEX_AI_SA_KEY="${VERTEX_AI_SA_KEY}"
export GOOGLE_MCP_AUTH_TOKEN="${GOOGLE_MCP_AUTH_TOKEN}"
# export PRINTIFY_API_TOKEN="your-token-here"
EOF

echo "📝 Environment variables saved to deployment/.env.deployment"
echo "   To use them in future sessions, run: source deployment/.env.deployment"
echo ""
echo "🚀 Next steps:"
echo "   1. Set your Printify API token: export PRINTIFY_API_TOKEN='your-token'"
echo "   2. Run the deployment: ./deployment/deploy-production.sh"
