# AI Research Assistant Agent 📚🔬

An enterprise-grade AI Research Assistant built with **FastAPI**, **Google GenAI / ADK**, **ChromaDB**, **BM25**, and **SQLAlchemy**. The agent ingests PDF/text documents, performs advanced hybrid search with re-ranking, executes chained multi-call RAG workflows, maintains persistent conversational memory with sliding context windows, and enforces role-based access control (RBAC).

---

## 🚀 Key Features

### 1. Multi-Strategy Document Ingestion
- **Formats Supported**: Upload PDF files (`pypdf`) or raw text documents.
- **Fixed-Size Chunking**: Character-based chunking with configurable overlap (`RecursiveCharacterTextSplitter`).
- **Semantic Chunking**: Computes consecutive sentence embedding cosine distances using Gemini embeddings and splits on statistical percentile breakpoints (default: 85th percentile).
- **Metadata Tagging**: Automatically records source filename, strategy (`fixed` or `semantic`), chunk index, page number, and ingestion timestamp.

### 2. Hybrid Search & Re-ranking
- **Sparse Retrieval**: `BM25Plus` algorithm (`rank-bm25`) with alphanumeric tokenization, handling keyword-exact queries and small corpora without negative score decay. Disk-backed at `data/bm25_index.pkl`.
- **Dense Vector Search**: Semantic vector retrieval using ChromaDB and Google Gemini embeddings (`gemini-embedding-001`).
- **Reciprocal Rank Fusion (RRF)**: Merges sparse keyword matches and dense semantic results:
  $$RRF(d) = \sum_{m \in M} \frac{w_m}{60 + r_m(d)}$$
- **Configurable Search Modes**: Query via `mode: "hybrid"`, `mode: "dense"`, or `mode: "sparse"`.

### 3. Advanced Retrieval (HyDE & Multi-Query)
- **HyDE (Hypothetical Document Embeddings)**: Zero-shot generation of an academic reference document to eliminate query-document semantic asymmetry before dense vector retrieval.
- **Multi-Query Expansion**: Generates 3 query reformulations (syntactic, domain terminology, technical focus) and fuses candidate matches with multi-list RRF to maximize recall on complex questions.

### 4. Chained RAG Pipeline
- **2-Call Chaining Requirement**:
  - **LLM Call 1 (Expansion / HyDE)**: Gemini generates the hypothetical reference text and technical query variations.
  - **Retrieval & Fusion**: Multi-list hybrid retrieval combining BM25 sparse matches and ChromaDB dense vectors.
  - **LLM Call 2 (Grounded Synthesis)**: Synthesizes evidence-based research answers strictly citing sources and page numbers (`[Source: <filename>, Page: <page_number>]`).
- Turn history automatically stored in the user's conversational session.

### 5. Conversational Memory & Sliding Window
- **Relational Persistence**: SQLite / PostgreSQL backing with SQLAlchemy ORM (`ChatSession` and `ChatMessage`).
- **Sliding Context Window**: Automatically retrieves only the last $N$ messages (e.g., 10 turns) to respect token budgets while retaining recent conversational context.
- **Multi-User Isolation**: Strictly scoped queries ensure users only access their own conversation history.
- **Cascade Deletion**: Clean session removal with cascading message deletion.

### 6. Authentication & RBAC
- **JWT Bearer Authentication**: Secure token generation (`HS256`) and verification with expiration tracking.
- **Role-Based Access Control**: Granular roles (`admin`, `researcher`) protecting sensitive endpoints (e.g. user administration).
- **Password Security**: Cryptographic password hashing using `bcrypt`.

---

## 🏗️ Architecture

```text
                                  +-----------------------------+
                                  |       Client / User         |
                                  +--------------+--------------+
                                                 |
                                     HTTP (JWT Bearer Token)
                                                 v
+-----------------------------------------------------------------------------------------------+
|                                    FastAPI Application                                        |
|  +-------------------+      +---------------------+      +---------------------------------+  |
|  |   Auth Routes     |      |  Ingestion Routes   |      |          Agent Routes           |  |
|  | (/api/v1/auth)    |      | (/api/v1/ingest)    |      | (/api/v1/agent)                 |  |
|  +---------+---------+      +----------+----------+      +----------------+----------------+  |
+------------|---------------------------|----------------------------------|-------------------+
             |                           |                                  |
             v                           v                                  v
+------------------------+  +---------------------------+  +------------------------------------+
| SQLite / PostgreSQL    |  | Ingestion & Chunking      |  | Chained RAG & Agent Execution      |
| - Users & Credentials  |  | - Fixed Splitter          |  | - LLM Call 1: HyDE / Multi-Query   |
| - Chat Sessions        |  | - Semantic Splitter       |  | - Hybrid Search (BM25 + Chroma)    |
| - Message History      |  +-------------+-------------+  | - LLM Call 2: Grounded Synthesis   |
+------------------------+                |                +-----------------+------------------+
                                          v                                  |
                        +------------------------------------+               |
                        |         Retrieval Engine           |<--------------+
                        |  - BM25Plus (Sparse Lexical)       |
                        |  - ChromaDB (Dense Embeddings)     |
                        |  - Reciprocal Rank Fusion (RRF)    |
                        +------------------------------------+
```

---

## 📋 API Endpoints

### Authentication (`/api/v1/auth`)
| Method | Path | Role | Description |
|---|---|---|---|
| `POST` | `/api/v1/auth/register` | Public | Register a new user (`researcher` or `admin`) |
| `POST` | `/api/v1/auth/login` | Public | Authenticate credentials and receive a JWT token |
| `GET` | `/api/v1/auth/me` | Authenticated | Get current authenticated user profile |
| `GET` | `/api/v1/auth/users` | Admin | List all registered users (Admin only) |

### Document Ingestion & Search (`/api/v1/ingest`)
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/ingest/text` | Ingest raw text with strategy (`fixed` or `semantic`) |
| `POST` | `/api/v1/ingest/pdf` | Upload and parse PDF file with strategy selection |
| `POST` | `/api/v1/ingest/search` | Search documents (`mode`: `hybrid`, `dense`, `sparse`, `use_hyde`) |

### Research Agent & Conversational Memory (`/api/v1/agent`)
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/agent/research` | Run 2-call Chained RAG pipeline (HyDE + Multi-Query + Grounded Synthesis) |
| `POST` | `/api/v1/agent/chat` | Chat with the Google ADK root agent |
| `GET` | `/api/v1/agent/sessions` | List all conversation sessions for the authenticated user |
| `GET` | `/api/v1/agent/sessions/{id}` | Get full chronological message history for a session |
| `DELETE` | `/api/v1/agent/sessions/{id}` | Delete a chat session and all cascading message records |

---

## 🛠️ Getting Started

### Prerequisites
- Python `3.12+`
- [uv](https://github.com/astral-sh/uv) (recommended fast Python package manager)
- Google Gemini API Key

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rejul-newformtech/AI-Research-Agent.git
   cd AI-Research-Agent
   ```

2. **Install dependencies with `uv`**:
   ```bash
   uv sync
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory (or copy from `.env.example`):
   ```ini
   APP_NAME="Research Assistant Agent"
   APP_VERSION="0.1.0"
   DEBUG=true

   # Google Gemini API Configuration
   GEMINI_API_KEY="your-gemini-api-key-here"
   GEMINI_MODEL="gemini-2.5-flash"
   EMBEDDING_MODEL="gemini-embedding-001"

   # Database Configuration (SQLite default; PostgreSQL supported)
   DATABASE_URL="sqlite:///data/research_agent.db"

   # Security & Authentication
   JWT_SECRET_KEY="generate-a-secure-random-32-byte-hex-string"
   ACCESS_TOKEN_EXPIRE_MINUTES=60

   # Ingestion & Chunking Defaults
   DEFAULT_CHUNK_SIZE=1000
   DEFAULT_CHUNK_OVERLAP=200
   DEFAULT_BREAKPOINT_PERCENTILE=85.0

   # ChromaDB Persistence
   CHROMA_PERSIST_DIR="data/chroma"
   ```

4. **Run the Development Server**:
   ```bash
   make dev
   # or
   uv run fastapi dev app/main.py
   ```
   The interactive Swagger documentation will be live at:
   👉 **http://localhost:8000/docs**

---

## 🧪 Testing & Code Quality

### Running Tests
The project includes a comprehensive test suite (27 unit and integration tests) using in-memory SQLite and mock fixtures:

```bash
uv run python -c "import unittest, sys; suite = unittest.defaultTestLoader.discover('tests'); res = unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(0 if res.wasSuccessful() else 1)"
```

### Pre-Commit Code Hygiene
Run all pre-commit linters and formatters (`ruff`, `ruff-format`, file hygiene):
```bash
uv run pre-commit run --all-files
```

---

## 🗺️ Project Roadmap

- [x] **Stage 001**: Document Ingestion (Fixed + Semantic Chunking) & Persistent Conversational Memory.
- [x] **Stage 002**: Hybrid Search (`BM25Plus` + ChromaDB) with Reciprocal Rank Fusion (RRF) Re-ranking.
- [x] **Stage 003**: Advanced Retrieval (HyDE, Multi-Query) and 2-Call Chained RAG Synthesis.
- [ ] **Stage 004**: Structured Output (Typed Pydantic Models) & Dynamic System Prompting.
- [ ] **Stage 005**: ReAct Agent Loop (Reasoning + Tool execution cycles).
- [ ] **Stage 006**: MCP Server (3 Tools + 1 Resource) & A2A Server with Agent Card.
- [ ] **Stage 007**: Evaluation Report & Benchmarks.
