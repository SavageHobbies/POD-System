#!/bin/bash

# Deploy only the CEO Orchestrator service
# Based on existing cloud infrastructure

set -e

PROJECT_ID="helios-pod-system"
REGION="us-central1"

echo "🚀 Deploying Helios CEO Orchestrator to existing cloud infrastructure..."

# Load environment variables from your .env file
if [ -f ".env" ]; then
    echo "📋 Loading environment variables from .env file..."
    set -a  # automatically export all variables
    source .env
    set +a  # stop auto-exporting
    echo "✅ Environment variables loaded"
else
    echo "❌ .env file not found. Please ensure it exists in the project root."
    exit 1
fi

# Verify required variables are set
if [ -z "$PRINTIFY_API_TOKEN" ] || [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ Required environment variables not found in .env file"
    echo "   Please ensure PRINTIFY_API_TOKEN and GEMINI_API_KEY are set"
    exit 1
fi

# Ensure we're using the right project
gcloud config set project $PROJECT_ID

# Create/update secrets that might be missing
echo "🔐 Ensuring all secrets are configured..."

# Gemini API Key (might be missing)
echo "$GEMINI_API_KEY" | gcloud secrets create gemini-api-key \
    --data-file=- --replication-policy="automatic" --project=$PROJECT_ID 2>/dev/null || \
echo "$GEMINI_API_KEY" | gcloud secrets versions add gemini-api-key \
    --data-file=- --project=$PROJECT_ID

echo "✅ Secrets verified"

# Build and push CEO Docker image
echo "🐳 Building CEO service Docker image..."
docker build -f deployment/docker/Dockerfile.ceo -t gcr.io/$PROJECT_ID/helios-ceo:latest .
docker push gcr.io/$PROJECT_ID/helios-ceo:latest

# Deploy CEO service to Cloud Run
echo "🚀 Deploying CEO service..."
gcloud run deploy helios-ceo \
    --image gcr.io/$PROJECT_ID/helios-ceo:latest \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 900 \
    --concurrency 80 \
    --min-instances 1 \
    --max-instances 10 \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_REGION=$REGION,GOOGLE_MCP_URL=https://helios-mcp-658997361183.us-central1.run.app,PRINTIFY_SHOP_ID=8542090,GEMINI_MODEL=gemini-2.5-flash,DRY_RUN=false,ALLOW_LIVE_PUBLISHING=true,BLUEPRINT_ID=145,PRINT_PROVIDER_ID=29,ASSETS_BUCKET=helios-product-assets-658997361183,PREFERRED_PROVIDER_NAME=Monster Digital" \
    --set-secrets="PRINTIFY_API_TOKEN=printify-api-token:latest,GOOGLE_MCP_AUTH_TOKEN=google-mcp-auth-token:latest,GEMINI_API_KEY=gemini-api-key:latest,GOOGLE_SERVICE_ACCOUNT_JSON=helios-mcp-sa-key:latest"

# Get the CEO service URL
CEO_URL=$(gcloud run services describe helios-ceo --region=$REGION --format='value(status.url)')

echo ""
echo "✅ CEO service deployed successfully!"
echo "🔗 CEO Service URL: $CEO_URL"

# Test the deployment
echo "🏥 Testing CEO service health..."
if curl -f "$CEO_URL/health" > /dev/null 2>&1; then
    echo "✅ CEO service is healthy"
else
    echo "❌ CEO service health check failed"
    echo "Check logs: gcloud logs read --service helios-ceo --region $REGION"
fi

echo ""
echo "🎉 Deployment complete! Your Helios system is now fully operational."
echo ""
echo "📊 Control your system:"
echo "   Start: curl -X POST $CEO_URL/run-async"
echo "   Health: curl $CEO_URL/health"
echo "   Logs: gcloud logs read --service helios-ceo --region $REGION"
echo ""
echo "🎛️  Or use Google Cloud Console:"
echo "   https://console.cloud.google.com/run?project=$PROJECT_ID"
