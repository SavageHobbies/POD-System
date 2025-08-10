# Google MCP Integration Setup Guide

## 🚀 Overview

Your Helios project now has a comprehensive Google MCP (Model Context Protocol) integration that provides:

- **Different Gemini models for different tasks** (as per your HTML spec)
- **Google Cloud services integration** (Vertex AI, Cloud Storage, Vision, etc.)
- **Google Sheets and Drive operations**
- **Multimodal AI capabilities** (text, image, voice)
- **Trend analysis and social media scanning**

## 🔧 Required Environment Variables

Add these to your `.env` file:

```bash
# Google MCP Configuration
GOOGLE_MCP_URL=http://localhost:8787
GOOGLE_MCP_AUTH_TOKEN=helios_mcp_token_2024

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your_google_cloud_project_id

# MCP Bearer Token (for MCP server authentication)
MCP_BEARER_TOKEN=helios_mcp_token_2024

# You already have these:
# GEMINI_API_KEY=your_gemini_api_key
# GOOGLE_SERVICE_ACCOUNT_JSON=~/.config/helios/helios-sa.json
```

## 🎯 Gemini Model Assignment (Per Your Spec)

### **Fast Tasks (Gemini 2.0 Flash):**
- **Orchestrator AI** → `gemini-2.0-flash-exp` (Fast decision making)
- **Trend Seeker** → `gemini-2.0-flash-exp` (Fast trend analysis)
- **Ethics AI** → `gemini-2.0-flash-exp` (Fast ethical screening)
- **Marketing AI** → `gemini-2.0-flash-exp` (Fast creative marketing)
- **Creative AI** → `gemini-2.0-flash-exp` (Fast design ideas)
- **Image Generation** → `gemini-2.0-flash-exp` (Fast image tasks)
- **Multimodal AI** → `gemini-2.0-flash-exp` (Fast multimodal processing)

### **Advanced Tasks (Gemini 2.0 Pro):**
- **Complex Reasoning** → `gemini-2.0-pro` (When needed for advanced analysis)

## 🛠️ Installation & Setup

### 1. Install MCP Dependencies
```bash
cd mcp
pip install -r requirements.txt
```

### 2. Start the MCP Server
```bash
cd mcp
python server.py
```

The server will run on `http://localhost:8787`

### 3. Test the Connection
```bash
curl http://localhost:8787/health
```

## 🎮 Available MCP Tools

### **AI Agents (Different Gemini Models)**
- `orchestrator_ai` - CEO/Orchestrator decisions
- `trend_seeker` - Trend analysis and keyword generation
- `ethics_ai` - Ethical screening and risk assessment
- `marketing_ai` - Marketing copy and tag generation
- `creative_ai` - Creative design ideas
- `image_generation` - Image concept generation
- `multimodal_ai` - Text, image, voice processing

### **Google Services**
- `google_trends_keywords` - Google Trends integration
- `social_media_scanner` - Social media trend scanning
- `google_sheets_operation` - Read/write Google Sheets
- `google_drive_operation` - File operations in Google Drive
- `vertex_ai_call` - Advanced Vertex AI models

## 📱 Usage Examples

### **1. Orchestrator AI (Fast Decision Making)**
```python
import requests

response = requests.post("http://localhost:8787/execute", json={
    "tool": "orchestrator_ai",
    "parameters": {
        "prompt": "Analyze this trend: 'vintage gaming' and make a business decision for print-on-demand products"
    }
})
print(response.json())
```

### **2. Trend Seeker (Fast Trend Analysis)**
```python
response = requests.post("http://localhost:8787/execute", json={
    "tool": "trend_seeker",
    "parameters": {
        "seed": "vintage gaming"
    }
})
print(response.json())
```

### **3. Marketing AI (Fast Creative Marketing)**
```python
response = requests.post("http://localhost:8787/execute", json={
    "tool": "marketing_ai",
    "parameters": {
        "product_info": {
            "name": "Vintage Gaming T-Shirt",
            "style": "retro pixel art",
            "target_audience": "gamers 18-35"
        }
    }
})
print(response.json())
```

### **4. Google Sheets Integration**
```python
response = requests.post("http://localhost:8787/execute", json={
    "tool": "google_sheets_operation",
    "parameters": {
        "operation": "read",
        "sheet_id": "your_sheet_id",
        "worksheet_name": "Sheet1"
    }
})
print(response.json())
```

## 🔐 Authentication

The MCP server uses bearer token authentication. Set the token in your requests:

```python
headers = {
    "Authorization": "Bearer helios_mcp_token_2024",
    "Content-Type": "application/json"
}

response = requests.post("http://localhost:8787/execute", 
                        json=payload, 
                        headers=headers)
```

## 🌐 Integration with Helios

### **Update your Helios MCP client configuration:**
```python
# In your Helios config or MCP client
MCP_BASE_URL = "http://localhost:8787"
MCP_AUTH_TOKEN = "helios_mcp_token_2024"
```

### **Call MCP tools from Helios agents:**
```python
# Example: CEO agent using MCP orchestrator
async def make_decision(self, trend_data):
    mcp_response = await self.mcp_client.call("orchestrator_ai", {
        "prompt": f"Analyze trend: {trend_data} and decide if we should proceed"
    })
    return mcp_response["response"]
```

## 🚀 Advanced Features

### **Multimodal AI (Text + Image + Voice)**
```python
# Send image with text prompt
with open("design.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8787/execute", json={
    "tool": "multimodal_ai",
    "parameters": {
        "prompt": "Analyze this design and suggest improvements",
        "image_data": image_data
    }
})
```

### **Vertex AI Integration**
```python
response = requests.post("http://localhost:8787/execute", json={
    "tool": "vertex_ai_call",
    "parameters": {
        "model": "gemini-2.0-pro",
        "prompt": "Complex reasoning task here"
    }
})
```

## 📊 Monitoring & Health

### **Health Check**
```bash
curl http://localhost:8787/health
```

### **Available Tools**
```bash
curl http://localhost:8787/tools
```

## 🔧 Troubleshooting

### **Common Issues:**

1. **Port already in use**: Change port in `server.py`
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Authentication errors**: Check `MCP_BEARER_TOKEN` in `.env`
4. **Google services not working**: Verify service account and project ID

### **Logs:**
The MCP server prints initialization logs. Check for:
- ✅ Gemini API initialized
- ✅ Google Cloud AI Platform initialized  
- ✅ Google Sheets service initialized

## 🎯 Next Steps

1. **Set up Google Cloud Project** and get your project ID
2. **Update your `.env` file** with the new variables
3. **Install MCP dependencies**: `cd mcp && pip install -r requirements.txt`
4. **Start the MCP server**: `python server.py`
5. **Test the integration** with the examples above
6. **Integrate with your Helios agents** using the MCP client

## 🔗 Integration Points

- **CEO Agent** → `orchestrator_ai` for business decisions
- **Trend Agent** → `trend_seeker` for trend analysis
- **Ethics Agent** → `ethics_ai` for ethical screening
- **Marketing Agent** → `marketing_ai` for copy generation
- **Creative Agent** → `creative_ai` for design ideas
- **Publishing** → Google Sheets for tracking, Google Drive for assets

This setup gives you exactly what you specified: different Gemini models for different tasks, Google Cloud integration, and a comprehensive MCP server that handles all your AI orchestration needs!
