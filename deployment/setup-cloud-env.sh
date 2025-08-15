#!/bin/bash

# Cloud Environment Setup Script for Helios
# Based on your current .env configuration

set -e

PROJECT_ID="helios-pod-system"
REGION="us-central1"

echo "🔧 Setting up Google Cloud environment for Helios..."

# Set your environment variables for deployment
export PRINTIFY_API_TOKEN="YOUR_REAL_PRINTIFY_TOKEN_HERE"  # Replace with actual token
export GOOGLE_MCP_AUTH_TOKEN="helios_mcp_token_2024"
export GEMINI_API_KEY="AIzaSyBnAvjXfKklaH-E6US2PHWkj89GKddYl1g"

# Upload service account key to Secret Manager
echo "📋 Uploading service account key to Secret Manager..."
gcloud secrets create helios-service-account-key \
    --data-file="/Users/nizorfarukhzoda/.config/helios/helios-vertex-ai.json" \
    --project=$PROJECT_ID \
    --replication-policy="automatic" || \
gcloud secrets versions add helios-service-account-key \
    --data-file="/Users/nizorfarukhzoda/.config/helios/helios-vertex-ai.json" \
    --project=$PROJECT_ID

echo "✅ Service account key uploaded to Secret Manager"

# Create additional required secrets
echo "🔐 Creating additional secrets..."

# Printify API Token (you'll need to replace with actual token)
echo "$PRINTIFY_API_TOKEN" | gcloud secrets create printify-api-token \
    --data-file=- --replication-policy="automatic" --project=$PROJECT_ID || \
echo "$PRINTIFY_API_TOKEN" | gcloud secrets versions add printify-api-token \
    --data-file=- --project=$PROJECT_ID

# Google MCP Auth Token
echo "$GOOGLE_MCP_AUTH_TOKEN" | gcloud secrets create google-mcp-auth-token \
    --data-file=- --replication-policy="automatic" --project=$PROJECT_ID || \
echo "$GOOGLE_MCP_AUTH_TOKEN" | gcloud secrets versions add google-mcp-auth-token \
    --data-file=- --project=$PROJECT_ID

# Gemini API Key
echo "$GEMINI_API_KEY" | gcloud secrets create gemini-api-key \
    --data-file=- --replication-policy="automatic" --project=$PROJECT_ID || \
echo "$GEMINI_API_KEY" | gcloud secrets versions add gemini-api-key \
    --data-file=- --project=$PROJECT_ID

echo "✅ All secrets configured in Google Secret Manager"
echo ""
echo "🚀 Ready to deploy! Run the deployment script next:"
echo "./deploy-production.sh"
