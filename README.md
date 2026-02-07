# Intelli_GPT 🤖

A powerful Multimodal RAG (Retrieval-Augmented Generation) API built with FastAPI that enables intelligent document processing and question-answering capabilities. Upload PDF documents, process them with multimodal embeddings, and query them using natural language.

## ✨ Features

- **PDF Ingestion**: Upload and process PDF documents with automatic text and image extraction
- **Multimodal Embeddings**: Leverages both text and image embeddings using CLIP models
- **Vector Search**: Uses Qdrant vector database for efficient similarity search
- **AI-Powered Responses**: Integrates with Google Gemini for intelligent question answering
- **Cloud Storage**: AWS S3 integration for document storage
- **RESTful API**: Clean, documented FastAPI endpoints

## 🏗️ Architecture

```
app/
├── api/                    # API route handlers
│   ├── routes_ingestion.py # PDF upload endpoints
│   └── routes_query.py     # Question-answering endpoints
├── core/                   # Core configuration and settings
├── infra/                  # Infrastructure components (S3, Qdrant)
├── pipelines/              # Processing pipelines
│   ├── ingestion_pipeline.py
│   └── query_pipeline.py
├── services/               # Business logic services
└── main.py                 # Application entry point
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- AWS account (for S3 storage)
- Qdrant instance (local or cloud)
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KARTIK64-rgb/Intelli_GPT.git
   cd Intelli_GPT
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # AWS Configuration
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=your_region
   S3_BUCKET_NAME=your_bucket_name
   
   # Qdrant Configuration
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_key
   
   # Google Gemini Configuration
   GEMINI_API_KEY=your_gemini_api_key
   ```

4. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

   The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Health Check
```http
GET /health
```
Returns the API status.

### Ingest PDF
```http
POST /ingest/pdf
Content-Type: multipart/form-data
```

**Request:**
- `file`: PDF file (multipart/form-data)

**Response:**
```json
{
  "pdf_id": "abc123...",
  "status": "ingested"
}
```

### Query Documents
```http
POST /query/
Content-Type: application/json
```

**Request:**
```json
{
  "question": "What is the main topic of the document?"
}
```

**Response:**
```json
{
  "answer": "The document discusses...",
  "sources": [...],
  "confidence": 0.95
}
```

## 🛠️ Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **PyMuPDF**: PDF processing and text extraction
- **Boto3**: AWS SDK for S3 storage
- **Qdrant**: Vector database for similarity search
- **Google Gemini**: AI model for question answering
- **OpenCLIP + PyTorch**: Image and text embeddings
- **Pillow**: Image processing

## 📖 Usage Example

### Python
```python
import requests

# Upload a PDF
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/ingest/pdf',
        files={'file': f}
    )
    pdf_id = response.json()['pdf_id']

# Query the document
response = requests.post(
    'http://localhost:8000/query/',
    json={'question': 'What are the key findings?'}
)
print(response.json()['answer'])
```

### cURL
```bash
# Upload PDF
curl -X POST "http://localhost:8000/ingest/pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Query
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?"}'
```

## 🔧 Development

### Interactive API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`



## 👤 Author

**KARTIK64-rgb**
- GitHub: [@KARTIK64-rgb](https://github.com/KARTIK64-rgb)

## 🙏 Acknowledgments

- FastAPI for the amazing web framework
- Google Gemini for AI capabilities
- Qdrant for vector search
- OpenAI CLIP for multimodal embeddings

---

⭐ Star this repository if you find it helpful!
