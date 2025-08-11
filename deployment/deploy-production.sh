#!/bin/bash

# Production Deployment Script for Helios POD Business
# Project: helios-pod-system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="helios-pod-system"
REGION="us-central1"
SERVICES=("helios-ceo" "helios-agents" "helios-mcp")

echo -e "${BLUE}🚀 Starting Production Deployment for Helios POD Business${NC}"
echo -e "${BLUE}Project: ${PROJECT_ID}${NC}"
echo -e "${BLUE}Region: ${REGION}${NC}"
echo ""

# Check if required environment variables are set
check_env_vars() {
    echo -e "${YELLOW}🔍 Checking environment variables...${NC}"
    
    required_vars=(
        "PRINTIFY_API_TOKEN"
        "GOOGLE_MCP_AUTH_TOKEN"
        "VERTEX_AI_SA_KEY"
        "GOOGLE_CLOUD_API_KEY"
    )
    
    missing_vars=()
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        echo -e "${RED}❌ Missing required environment variables:${NC}"
        for var in "${missing_vars[@]}"; do
            echo -e "${RED}   - ${var}${NC}"
        done
        echo ""
        echo -e "${YELLOW}Please set these variables and try again:${NC}"
        echo "export PRINTIFY_API_TOKEN='your-token'"
        echo "export GOOGLE_MCP_AUTH_TOKEN='your-token'"
        echo "export VERTEX_AI_SA_KEY='your-service-account-key'"
        echo "export GOOGLE_CLOUD_API_KEY='your-api-key'"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All required environment variables are set${NC}"
    echo ""
}

# Set up Google Cloud configuration
setup_gcloud() {
    echo -e "${YELLOW}🔧 Setting up Google Cloud configuration...${NC}"
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    echo -e "${YELLOW}Enabling required APIs...${NC}"
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable run.googleapis.com
    gcloud services enable cloudresourcemanager.googleapis.com
    gcloud services enable iam.googleapis.com
    gcloud services enable secretmanager.googleapis.com
    gcloud services enable vertexai.googleapis.com
    gcloud services enable firestore.googleapis.com
    gcloud services enable storage.googleapis.com
    gcloud services enable pubsub.googleapis.com
    gcloud services enable cloudscheduler.googleapis.com
    
    echo -e "${GREEN}✅ Google Cloud configuration complete${NC}"
    echo ""
}

# Create and configure secrets
setup_secrets() {
    echo -e "${YELLOW}🔐 Setting up secrets...${NC}"
    
    # Create secrets in Secret Manager
    echo -e "${YELLOW}Creating secrets in Secret Manager...${NC}"
    
    # Printify API Token
    echo "$PRINTIFY_API_TOKEN" | gcloud secrets create printify-api-token --data-file=- --replication-policy="automatic" || \
    echo "$PRINTIFY_API_TOKEN" | gcloud secrets versions add printify-api-token --data-file=-
    
    # Google MCP Auth Token
    echo "$GOOGLE_MCP_AUTH_TOKEN" | gcloud secrets create google-mcp-auth-token --data-file=- --replication-policy="automatic" || \
    echo "$GOOGLE_MCP_AUTH_TOKEN" | gcloud secrets versions add google-mcp-auth-token --data-file=-
    
    # Vertex AI Service Account Key
    echo "$VERTEX_AI_SA_KEY" | gcloud secrets create vertex-ai-sa-key --data-file=- --replication-policy="automatic" || \
    echo "$VERTEX_AI_SA_KEY" | gcloud secrets versions add vertex-ai-sa-key --data-file=-
    
    # Google Cloud API Key
    echo "$GOOGLE_CLOUD_API_KEY" | gcloud secrets create google-cloud-api-key --data-file=- --replication-policy="automatic" || \
    echo "$GOOGLE_CLOUD_API_KEY" | gcloud secrets versions add google-cloud-api-key --data-file=-
    
    echo -e "${GREEN}✅ Secrets configured${NC}"
    echo ""
}

# Build and push Docker images
build_images() {
    echo -e "${YELLOW}🐳 Building and pushing Docker images...${NC}"
    
    # Build CEO service
    echo -e "${YELLOW}Building helios-ceo image...${NC}"
    docker build -f deployment/docker/Dockerfile.ceo -t gcr.io/$PROJECT_ID/helios-ceo:latest .
    docker push gcr.io/$PROJECT_ID/helios-ceo:latest
    
    # Build agents service
    echo -e "${YELLOW}Building helios-agents image...${NC}"
    docker build -f deployment/docker/Dockerfile.agents -t gcr.io/$PROJECT_ID/helios-agents:latest .
    docker push gcr.io/$PROJECT_ID/helios-agents:latest
    
    # Build MCP server
    echo -e "${YELLOW}Building helios-mcp image...${NC}"
    docker build -f mcp/Dockerfile -t gcr.io/$PROJECT_ID/helios-mcp:latest ./mcp
    docker push gcr.io/$PROJECT_ID/helios-mcp:latest
    
    echo -e "${GREEN}✅ All images built and pushed${NC}"
    echo ""
}

# Deploy services to Cloud Run
deploy_services() {
    echo -e "${YELLOW}🚀 Deploying services to Cloud Run...${NC}"
    
    # Deploy MCP server first (dependency for other services)
    echo -e "${YELLOW}Deploying helios-mcp...${NC}"
    gcloud run deploy helios-mcp \
        --image gcr.io/$PROJECT_ID/helios-mcp:latest \
        --region $REGION \
        --platform managed \
        --allow-unauthenticated \
        --memory 4Gi \
        --cpu 2 \
        --timeout 1800 \
        --concurrency 80 \
        --max-instances 10
    
    # Deploy CEO service
    echo -e "${YELLOW}Deploying helios-ceo...${NC}"
    gcloud run deploy helios-ceo \
        --image gcr.io/$PROJECT_ID/helios-ceo:latest \
        --region $REGION \
        --platform managed \
        --allow-unauthenticated \
        --memory 4Gi \
        --cpu 2 \
        --timeout 900 \
        --concurrency 80 \
        --max-instances 10 \
        --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_REGION=$REGION" \
        --set-secrets="PRINTIFY_API_TOKEN=printify-api-token:latest,GOOGLE_MCP_AUTH_TOKEN=google-mcp-auth-token:latest"
    
    # Deploy agents service
    echo -e "${YELLOW}Deploying helios-agents...${NC}"
    gcloud run deploy helios-agents \
        --image gcr.io/$PROJECT_ID/helios-agents:latest \
        --region $REGION \
        --platform managed \
        --allow-unauthenticated \
        --memory 8Gi \
        --cpu 4 \
        --timeout 1800 \
        --concurrency 100 \
        --max-instances 20 \
        --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_REGION=$REGION,VERTEX_AI_PROJECT_ID=$PROJECT_ID,VERTEX_AI_LOCATION=$REGION" \
        --set-secrets="PRINTIFY_API_TOKEN=printify-api-token:latest,GOOGLE_MCP_AUTH_TOKEN=google-mcp-auth-token:latest,VERTEX_AI_SA_KEY=vertex-ai-sa-key:latest"
    
    echo -e "${GREEN}✅ All services deployed${NC}"
    echo ""
}

# Set up Cloud Scheduler jobs
setup_scheduler() {
    echo -e "${YELLOW}⏰ Setting up Cloud Scheduler jobs...${NC}"
    
    # Market research every 6 hours
    gcloud scheduler jobs create http helios-market-research \
        --schedule="0 */6 * * *" \
        --uri="$(gcloud run services describe helios-agents --region=$REGION --format='value(status.url)')/market-research" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body='{"action": "market_research"}' \
        --location=$REGION || echo "Market research job already exists"
    
    # Supplier vetting daily
    gcloud scheduler jobs create http helios-supplier-vetting \
        --schedule="0 0 */1 * *" \
        --uri="$(gcloud run services describe helios-agents --region=$REGION --format='value(status.url)')/supplier-vetting" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body='{"action": "supplier_vetting"}' \
        --location=$REGION || echo "Supplier vetting job already exists"
    
    # Copyright review every 12 hours
    gcloud scheduler jobs create http helios-copyright-review \
        --schedule="0 */12 * * *" \
        --uri="$(gcloud run services describe helios-agents --region=$REGION --format='value(status.url)')/copyright-review" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body='{"action": "copyright_review"}' \
        --location=$REGION || echo "Copyright review job already exists"
    
    echo -e "${GREEN}✅ Scheduler jobs configured${NC}"
    echo ""
}

# Run health checks
health_check() {
    echo -e "${YELLOW}🏥 Running health checks...${NC}"
    
    # Get service URLs
    CEO_URL=$(gcloud run services describe helios-ceo --region=$REGION --format='value(status.url)')
    AGENTS_URL=$(gcloud run services describe helios-agents --region=$REGION --format='value(status.url)')
    MCP_URL=$(gcloud run services describe helios-mcp --region=$REGION --format='value(status.url)')
    
    # Check CEO service
    echo -e "${YELLOW}Checking helios-ceo...${NC}"
    if curl -f "$CEO_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ helios-ceo is healthy${NC}"
    else
        echo -e "${RED}❌ helios-ceo health check failed${NC}"
    fi
    
    # Check agents service
    echo -e "${YELLOW}Checking helios-agents...${NC}"
    if curl -f "$AGENTS_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ helios-agents is healthy${NC}"
    else
        echo -e "${RED}❌ helios-agents health check failed${NC}"
    fi
    
    # Check MCP service
    echo -e "${YELLOW}Checking helios-mcp...${NC}"
    if curl -f "$MCP_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ helios-mcp is healthy${NC}"
    else
        echo -e "${RED}❌ helios-mcp health check failed${NC}"
    fi
    
    echo ""
}

# Main deployment flow
main() {
    echo -e "${BLUE}🎯 Starting production deployment...${NC}"
    echo ""
    
    check_env_vars
    setup_gcloud
    setup_secrets
    build_images
    deploy_services
    setup_scheduler
    health_check
    
    echo -e "${GREEN}🎉 Production deployment completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}📊 Service URLs:${NC}"
    echo -e "${BLUE}CEO Service: ${NC}$(gcloud run services describe helios-ceo --region=$REGION --format='value(status.url)')"
    echo -e "${BLUE}Agents Service: ${NC}$(gcloud run services describe helios-agents --region=$REGION --format='value(status.url)')"
    echo -e "${BLUE}MCP Service: ${NC}$(gcloud run services describe helios-mcp --region=$REGION --format='value(status.url)')"
    echo ""
    echo -e "${BLUE}📅 Scheduled Jobs:${NC}"
    echo -e "${BLUE}Market Research: ${NC}Every 6 hours"
    echo -e "${BLUE}Supplier Vetting: ${NC}Daily at midnight"
    echo -e "${BLUE}Copyright Review: ${NC}Every 12 hours"
    echo ""
}

# Run main function
main "$@"
