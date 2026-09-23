from typing import Literal

from pydantic import BaseModel, Field


class TranscriptUpload(BaseModel):
    filename: str = Field(min_length=1, max_length=240)
    text: str = Field(min_length=1, max_length=2_000_000)


class CorpusUpload(BaseModel):
    transcripts: list[TranscriptUpload] = Field(min_length=1, max_length=20)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=800)


class GuideRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)


class TranscriptRecord(BaseModel):
    id: str
    filename: str
    title: str
    market: str | None = None
    chunkCount: int
    characterCount: int
    createdAt: str


class TranscriptChunk(BaseModel):
    id: str
    transcriptId: str
    filename: str
    title: str
    speaker: str
    market: str | None = None
    timestamp: str
    startSeconds: int
    content: str


class Citation(BaseModel):
    evidenceId: str
    filename: str
    title: str
    speaker: str
    market: str | None = None
    timestamp: str
    quote: str


class GeneratedAnswer(BaseModel):
    answer: str = ""
    evidenceIds: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
