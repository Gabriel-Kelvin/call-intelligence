# Call Intelligence

Call Intelligence is an evidence-first research application for analysing expert-call transcripts. Users upload transcripts and an interview guide at runtime, then receive cross-call insights and grounded answers with exact quotes, speakers, filenames, and timestamps.

This repository contains the rebuilt Python RAG version of the project. The interface is Next.js and React. The complete ingestion, retrieval, and generation backend is FastAPI, LangChain, Qdrant, and Groq.

## What it does

- Uploads up to 20 TXT, Markdown, VTT, or SRT transcripts
- Detects transcript metadata, speakers, and timestamps
- Splits speaker turns into overlapping, evidence-preserving chunks
- Embeds each chunk locally with `BAAI/bge-small-en-v1.5`
- Stores vectors and metadata in Qdrant
- Generates recurring themes and meaningful disagreements across calls
- Parses a newly uploaded interview guide and answers each question per expert
- Answers free-form questions across the corpus
- Shows exact stored excerpts rather than model-written quotations
- Validates every returned evidence ID before displaying a citation
- Deletes the corpus, vectors, and generated workspace state on request

## Architecture

```text
Browser
  |
  |  upload transcripts / guide / question
  v
Next.js + React UI
  |
  |  /api/*
  v
FastAPI
  |
  +--> transcript parser
  +--> LangChain RecursiveCharacterTextSplitter
  +--> FastEmbed embeddings
  +--> Qdrant vector database
  +--> multi-query retrieval
  +--> Groq LLM reranking and grounded generation
  |
  v
validated answer + original quote + timestamp
```

### RAG flow

1. FastAPI validates each upload.
2. The parser extracts expert, market, speaker, and timestamp metadata.
3. LangChain splits long speaker turns with overlap so passages remain understandable.
4. FastEmbed creates dense vectors and Qdrant persists them with source metadata.
5. A question is expanded into complementary search queries.
6. Qdrant retrieves semantically similar chunks with diversity across transcripts.
7. The LLM reranks candidates and receives only the selected transcript evidence.
8. The model returns an answer plus evidence IDs.
9. The backend rejects unknown IDs and hydrates citations from the original stored chunks.

The same grounding rules are used for guide answers and cross-call synthesis.

## Tech stack

### Frontend

- Next.js 16
- React 19 and TypeScript
- Tailwind CSS and custom responsive CSS
- Lucide icons

### Backend and AI

- Python 3.12
- FastAPI and Uvicorn
- LangChain document, splitter, embedding, and model integrations
- Qdrant local persistent vector database
- FastEmbed with `BAAI/bge-small-en-v1.5`
- Groq with `qwen/qwen3.8-27b`

## Project structure

```text
app/                         Next.js interface
components/                  UI components
backend/app/main.py          FastAPI routes
backend/app/rag_service.py   ingestion, retrieval, reranking, generation
backend/app/transcript_parser.py
                             transcript and interview-guide parsing
backend/app/models.py        request and domain models
backend/tests/               backend tests
render.yaml                  Python API deployment definition
next.config.ts               frontend-to-backend API proxy
```

## Run locally

Requirements:

- Python 3.12
- Node.js 22 or newer
- A Groq API key

Clone and configure the project:

```bash
git clone https://github.com/Gabriel-Kelvin/call-intelligence.git
cd call-intelligence
cp .env.example .env.local
```

Set `GROQ_API_KEY` in `.env.local`, then install and start the Python API:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

In a second terminal, start the unchanged web interface:

```bash
npm ci
npm run dev
```

Open `http://localhost:3000`. The Next.js development server proxies `/api/*` to `http://127.0.0.1:8000`.

## Tests and checks

```bash
pytest backend/tests
npm run lint
npm run build
```

FastAPI also exposes interactive API documentation at `http://127.0.0.1:8000/docs`.

## API

| Route | Method | Purpose |
| --- | --- | --- |
| `/api/health` | GET | Check the Python API and architecture |
| `/api/corpus` | GET | Read the active corpus summary |
| `/api/corpus` | POST | Parse, embed, and index uploaded transcripts |
| `/api/corpus` | DELETE | Delete corpus metadata and vectors |
| `/api/insights` | POST | Generate themes and disagreements |
| `/api/guide` | POST | Parse and answer an interview guide |
| `/api/ask` | POST | Run the multi-query RAG pipeline |

## Environment variables

| Variable | Used by | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | FastAPI | Required Groq secret |
| `GROQ_MODEL` | FastAPI | Optional model override |
| `QDRANT_PATH` | FastAPI | Local Qdrant persistence directory |
| `API_BACKEND_URL` | Next.js | Base URL of the Python API |
| `CORS_ORIGINS` | FastAPI | Comma-separated local or deployed UI origins |

## Deployment

- Deploy the frontend to Vercel and set `API_BACKEND_URL` to the Render service URL.
- Deploy `render.yaml` to Render and add `GROQ_API_KEY` as a secret.
- Render starts one Uvicorn process because local Qdrant storage is process-bound.

The free Render filesystem is ephemeral. That is acceptable for a case-study demo where users upload a fresh corpus. A production deployment should point the LangChain Qdrant integration to Qdrant Cloud or attach durable storage, add authentication, and isolate collections by workspace.

## Accuracy and safety choices

- Quotes are copied from stored chunks, never generated by the LLM.
- Model-returned evidence IDs must exist in the retrieved candidate set.
- Guide answers cannot borrow citations from another expert.
- Unanswerable questions return an evidence-gap response.
- Low temperature is used for stable analysis.
- Uploaded files and environment secrets are not committed.

## AI-assisted development disclosure

AI coding tools were used during implementation. The architecture, code, tests, deployment, and output behaviour were reviewed and verified by the repository owner. This disclosure should also be mentioned in the demo video, as requested in the assignment instructions.
