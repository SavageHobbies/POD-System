# Helios MCP Server

AI-Powered Print-on-Demand Pipeline with Google MCP Integration

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Copy environment template
cp env.example .env

# Edit .env with your API keys
nano .env
```

### 2. Install Dependencies
```bash
# Using pip
pip install -r requirements.txt

# Or using Docker
docker build -t helios-mcp .
```

### 3. Start Server
```bash
# Direct Python
./start.sh

# Or Docker
docker run -p 8080:8080 --env-file .env helios-mcp

# Or Docker Compose
docker-compose up
```

## 🔧 Configuration

### Required Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini AI API key
- `GOOGLE_CLOUD_PROJECT`: Google Cloud project ID
- `GOOGLE_SERVICE_ACCOUNT_JSON`: Path to service account JSON or raw JSON string
- `GOOGLE_DRIVE_FOLDER_ID`: Google Drive folder ID for file operations

### Optional Environment Variables
- `SERVER_HOST`: Server host (default: 0.0.0.0)
- `SERVER_PORT`: Server port (default: 8080)
- `LOG_LEVEL`: Logging level (default: info)

## 🌐 API Endpoints

### Health Check
```bash
GET /health
```

### Execute MCP Tool
```bash
POST /execute
{
  "tool": "tool_name",
  "parameters": {...}
}
```

## 🤖 Available AI Tools

### Core AI Agents
- **orchestrator_ai**: CEO/Orchestrator for business decisions
- **trend_seeker**: Trend analysis and market research
- **ethics_ai**: Ethical screening and risk assessment
- **marketing_ai**: Marketing copy and content generation
- **creative_ai**: Design ideas and creative concepts
- **image_generation**: Image generation (placeholder)
- **multimodal_ai**: Text + image processing

### Google Services
- **google_trends_keywords**: Get trending keywords
- **social_media_scanner**: Social media trend scanning
- **google_sheets_operation**: Read/write Google Sheets
- **google_drive_operation**: File operations in Google Drive
- **vertex_ai_call**: Advanced AI tasks (placeholder)

## 🧪 Testing

### Run Test Suite
```bash
python test_client.py
```

### Manual Testing
```bash
# Health check
curl http://localhost:8080/health

# Test AI tool
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "orchestrator_ai",
    "parameters": {
      "prompt": "Analyze sustainable fashion market trends"
    }
  }'
```

## 🐳 Docker

### Build Image
```bash
docker build -t helios-mcp .
```

### Run Container
```bash
docker run -p 8080:8080 --env-file .env helios-mcp
```

### Docker Compose
```bash
docker-compose up -d
```

## 📊 Monitoring

### Health Check Response
```json
{
  "status": "healthy",
  "services": {
    "gemini": true,
    "google_sheets": true,
    "google_drive": true,
    "google_storage": true,
    "google_trends": true
  },
  "models": {
    "orchestrator": "gemini-2.0-flash-exp",
    "trend": "gemini-2.0-flash-exp",
    "ethics": "gemini-2.0-flash-exp",
    "marketing": "gemini-2.0-flash-exp",
    "creative": "gemini-2.0-flash-exp"
  }
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using port 8080
   lsof -i :8080
   
   # Kill process or change port in .env
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

3. **API Key Issues**
   - Verify GEMINI_API_KEY is correct
   - Check Google Cloud service account permissions
   - Ensure billing is enabled for Google Cloud project

4. **Docker Issues**
   ```bash
   # Clean rebuild
   docker-compose down
   docker-compose build --no-cache
   docker-compose up
   ```

### Logs
- Server logs: `mcp_server.log`
- Docker logs: `docker-compose logs helios-mcp`

## 🚀 Next Steps

1. **Integrate with Main Helios Pipeline**
   - Update `helios/mcp_client.py` to use this server
   - Test end-to-end workflow

2. **Add More AI Models**
   - Implement image generation with Imagen
   - Add Vertex AI integration
   - Expand multimodal capabilities

3. **Production Deployment**
   - Add authentication
   - Implement rate limiting
   - Add monitoring and alerting
   - Set up CI/CD pipeline

## 📚 Resources

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Google Cloud APIs](https://cloud.google.com/apis)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MCP Protocol](https://modelcontextprotocol.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is part of the Helios AI-Powered Print-on-Demand Pipeline.
