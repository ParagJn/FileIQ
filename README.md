# FileIQ - Multi-Step Document Q&A Web Application

A sophisticated document analysis system that enables users to upload documents, generate vector embeddings, and perform intelligent Q&A using Claude Sonnet AI. Built with Next.js, FastAPI, and advanced vector search capabilities with **Case ID-based document organization**.

## 🚀 Features

### Core Functionality
- **🆔 Case ID Management**: Organize documents by Case ID for isolated processing and Q&A
- **📁 Multi-Format Document Support**: PDF, DOCX, TXT, and JSON files
- **🔍 Intelligent Vector Search**: FAISS-powered semantic search with relevance scoring
- **🤖 AI-Powered Q&A**: Claude Sonnet integration for contextual answers
- **⚡ Smart Vector Management**: Automatic detection of existing vectors to avoid re-processing
- **📊 Real-Time Progress Tracking**: Terminal-style progress modals with per-document status

### Advanced Features
- **🔄 Case-Scoped Vector Refresh**: Rebuild vectors for specific cases with progress visualization
- **🗑️ Case-Specific Vector Management**: Delete vectors by case and manage isolated document sets
- **🎯 Source Attribution**: Detailed source references with relevance scores
- **🛡️ Error Handling**: Comprehensive error handling with user-friendly messages
- **⚙️ Configuration Management**: Centralized config system for easy deployment

## 🏗️ High-Level Architecture

```mermaid
graph TD
    A[User Interface<br/>Next.js Frontend] --> B[Case ID Modal<br/>Input Validation]
    B --> C[Configuration Layer<br/>Centralized Settings]
    A --> D[FastAPI Backend<br/>Python Server]
    
    D --> E[Case-Specific Storage<br/>Document Organization]
    D --> F[Document Processing<br/>VectorDBManager]
    D --> G[Claude Sonnet API<br/>Anthropic]
    
    E --> H[Case Folders<br/>documents/{case_id}/]
    F --> I[File Loaders<br/>PDF/DOCX/TXT/JSON]
    F --> J[Text Chunking<br/>Overlapping Segments]
    F --> K[SentenceTransformers<br/>Embeddings Generation]
    F --> L[FAISS Vector DB<br/>Case-Specific Search]
    
    G --> M[Enhanced Responses<br/>Context + AI]
    
    N[Vector Storage<br/>{case_id}_{file}_vectordb] --> L
    O[Environment Config<br/>.env Files] --> C
    
    style A fill:#e1f5fe
    style B fill:#fff9c4
    style D fill:#f3e5f5
    style E fill:#e8f5e8
    style F fill:#e8f5e8
    style G fill:#fff3e0
    style L fill:#fce4ec
```

## 🔄 Application Flow

### 1. Case ID Setup & Document Upload
```
Case ID Modal → User Input Validation → Folder Selection → File Validation → 
Upload to Case Folder → Vector Check → Generate/Skip Vectors → Store in FAISS → Update UI
```

### 2. Case-Scoped Q&A Process
```
User Question → Case Vector Search → Context Retrieval → Claude API → 
Enhanced Response → Source Attribution → Display Results
```

### 3. Case-Specific Admin Operations
```
Refresh: Scan Case Files → Delete Old Case Vectors → Regenerate → Update Storage
Delete: Remove Case Vectors → Disable Q&A → Clear State
Restart: Reset UI State → Clear Case ID → Return to Initial Step
```

## 📂 Project Structure

```
FileIQ/
├── src/
│   ├── app/
│   │   ├── page.tsx                 # Main application component
│   │   ├── components/
│   │   │   ├── TerminalProgressModal.tsx  # Progress visualization
│   │   │   └── CaseIdModal.tsx      # Case ID input modal
│   │   ├── layout.tsx               # App layout
│   │   └── globals.css              # Global styles
│   ├── components/
│   │   ├── ui/                      # Shadcn/ui components
│   │   └── layout/                  # Header/Footer components
│   ├── config/
│   │   └── config.ts                # Centralized configuration
│   ├── hooks/                       # Custom React hooks
│   └── lib/                         # Utility functions
├── backend/
│   ├── fastapi_server.py            # FastAPI server & endpoints
│   ├── generate_document_vectors.py # Vector processing engine
│   ├── documents/                   # Case-organized document storage
│   │   ├── {case_id_1}/            # Documents for Case ID 1
│   │   ├── {case_id_2}/            # Documents for Case ID 2
│   │   └── vector-data/            # Case-specific vector databases
│   └── .env                         # Backend environment variables
├── .env.local                       # Frontend environment variables
└── README.md                        # This file
```

## 🛠️ Technology Stack

### Frontend
- **Framework**: Next.js 15 with TypeScript
- **UI Library**: Shadcn/ui components with Tailwind CSS
- **State Management**: React hooks (useState, useEffect)
- **Build Tool**: Turbopack for fast development

### Backend
- **API Framework**: FastAPI with automatic OpenAPI documentation
- **Vector Processing**: SentenceTransformers + FAISS
- **Document Processing**: PyPDF2, python-docx for file parsing
- **AI Integration**: Anthropic Claude Sonnet API
- **Environment**: Python with virtual environment

### Core Libraries
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **AI Model**: Claude 3.5 Haiku (Anthropic)
- **File Processing**: PyPDF2, python-docx, JSON parsing

## ⚙️ Configuration

### Environment Variables
```bash
# Frontend (.env.local)
NEXT_PUBLIC_API_BASE_URL=http://localhost:8001

# Backend (backend/.env)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Configurable Settings
All application settings are centralized in `src/config/config.ts`:
- API endpoints and base URLs
- File processing limits and timeouts
- Vector database parameters
- UI timing and behavior
- Supported file extensions

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+ with pip
- Anthropic API key

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd FileIQ
```

2. **Setup Frontend**
```bash
npm install
cp .env.local.example .env.local
# Edit .env.local with your API base URL
```

3. **Setup Backend**
```bash
cd backend

# Option 1: Create new virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Option 2: Use existing virtual environment (e.g., /Users/paragjain/dev-works/myenv)
# source /path/to/your/venv/bin/activate

pip install fastapi uvicorn anthropic sentence-transformers faiss-cpu PyPDF2 python-docx python-dotenv
cp .env.example .env
# Edit .env with your Anthropic API key
```

### Running the Application

1. **Start Backend Server**
```bash
cd backend

# Option 1: Using project virtual environment
source venv/bin/activate
python fastapi_server.py

# Option 2: Using existing virtual environment
# /path/to/your/venv/bin/python fastapi_server.py

# Server runs on http://localhost:8001
```

2. **Start Frontend Server**
```bash
npm run dev
# App available at http://localhost:3000
```

## 📋 Usage Guide

### Step 1: Case ID Setup
1. Click "Load Documents" 
2. Enter a unique Case ID in the modal (alphanumeric, dashes, underscores)
3. Case ID will be displayed in the top-right header

### Step 2: Document Upload & Processing
1. Folder selection opens automatically after Case ID is set
2. Review and select documents to process
3. Click "Process Documents & Generate Vectors"
4. Monitor progress in the terminal-style modal
5. Documents are organized by Case ID

### Step 3: Case-Scoped Q&A
1. Enter your question in the text area
2. Click "Get Answer" to receive AI-powered responses from case documents
3. Review source attributions and relevance scores
4. All answers are scoped to the current Case ID

### Step 4: Case-Specific Admin Operations
- **Refresh Vectors**: Rebuild vectors for current case only
- **Delete Vectors**: Remove vectors for current case only
- **Restart App**: Reset to initial state and clear Case ID

## 🔧 API Endpoints

### Document Processing
- `POST /upload-and-generate-vectors` - Upload and process single file with Case ID
- `POST /generate-vectors` - Generate vectors for existing file in specific case
- `POST /refresh-all-vectors` - Rebuild all document vectors for a case
- `POST /delete-all-vectors` - Remove all vector databases for a case

### Q&A System
- `POST /ask-question` - Submit question and get AI response scoped to case

## 🎯 Key Features Explained

### Case ID-Based Organization
- Each case maintains isolated document storage and vector databases
- Case ID validation ensures proper formatting (alphanumeric, dashes, underscores)
- Vector databases are named with case prefix: `{case_id}_{filename}_vectordb`
- All operations (upload, search, refresh, delete) are scoped to the active case

### Smart Vector Management
- Automatically detects existing vectors to avoid reprocessing
- Shows "Already vectorized" status for existing documents
- Provides option to refresh vectors when needed
- Case-specific vector storage prevents cross-case contamination

### Progress Visualization
- Terminal-style progress modal with real-time updates
- Per-document status tracking (pending → processing → done/error/skipped)
- Detailed progress messages for each processing stage
- Case-aware messaging throughout the process

### Intelligent Q&A
- Semantic search across case-specific document vectors only
- Contextual responses using Claude Sonnet
- Source attribution with relevance scores
- Fallback to general knowledge when no context found
- All answers scoped to current Case ID

### Error Handling
- Comprehensive error handling at all levels
- Case ID validation and requirement checks
- User-friendly error messages and recovery options
- Graceful degradation when services are unavailable

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Anthropic](https://anthropic.com) for Claude AI
- [Hugging Face](https://huggingface.co) for SentenceTransformers
- [Facebook Research](https://github.com/facebookresearch/faiss) for FAISS
- [Shadcn/ui](https://ui.shadcn.com) for beautiful UI components

---

**Built with ❤️ using Next.js, FastAPI, and Claude AI**
