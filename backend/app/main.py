from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .models import AskRequest, CorpusUpload, GuideRequest
from .rag_service import RagService


settings = get_settings()
rag = RagService(settings)
app = FastAPI(title="Call Intelligence API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_error(_: Request, error: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"error": str(error.detail)})


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
    first = error.errors()[0] if error.errors() else {}
    message = str(first.get("msg", "The request is invalid.")).replace("Value error, ", "")
    return JSONResponse(status_code=422, content={"error": message})


def fail(error: Exception) -> HTTPException:
    status = 400 if isinstance(error, ValueError) else 500
    return HTTPException(status_code=status, detail=str(error))


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"ok": True, "service": "python-fastapi", "vectorDatabase": "qdrant", "framework": "langchain"}


@app.get("/api/corpus")
def corpus_summary() -> dict[str, object]:
    return rag.summary()


@app.post("/api/corpus")
def upload_corpus(payload: CorpusUpload) -> dict[str, object]:
    try:
        return rag.ingest([(item.filename, item.text) for item in payload.transcripts])
    except Exception as error:
        raise fail(error) from error


@app.delete("/api/corpus")
def delete_corpus() -> dict[str, object]:
    try:
        return rag.clear()
    except Exception as error:
        raise fail(error) from error


@app.post("/api/ask")
def ask(payload: AskRequest) -> dict[str, object]:
    try:
        return rag.ask(payload.question.strip())
    except Exception as error:
        raise fail(error) from error


@app.post("/api/insights")
def insights() -> dict[str, object]:
    try:
        return rag.insights()
    except Exception as error:
        raise fail(error) from error


@app.post("/api/guide")
def guide(payload: GuideRequest) -> dict[str, object]:
    try:
        return rag.answer_guide(payload.text)
    except Exception as error:
        raise fail(error) from error
