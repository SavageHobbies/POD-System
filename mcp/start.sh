#!/bin/bash

echo "🚀 Starting Helios MCP Server..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from example..."
    cp env.example .env
    echo "📝 Please edit .env file with your API keys and configuration"
    echo "   Required: GEMINI_API_KEY, GOOGLE_CLOUD_PROJECT, GOOGLE_SERVICE_ACCOUNT_JSON"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check required environment variables
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set in .env file"
    exit 1
fi

if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "❌ GOOGLE_CLOUD_PROJECT not set in .env file"
    exit 1
fi

echo "✅ Environment configured"
echo "🔑 Gemini API: ${GEMINI_API_KEY:0:10}..."
echo "☁️  Google Project: $GOOGLE_CLOUD_PROJECT"

# Start server
echo "🌐 Starting server on port 8080..."
python3 server.py
