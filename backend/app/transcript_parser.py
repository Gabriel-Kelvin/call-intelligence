import hashlib
import re
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import TranscriptChunk, TranscriptRecord


TIMESTAMP = re.compile(r"^\s*(?:\(|\[)?((?:\d{1,2}:)?\d{1,2}:\d{2})(?:\)|\])?\s*(.*)$")
SUBTITLE_TIME = re.compile(r"^((?:\d{1,2}:)?\d{1,2}:\d{2})(?:[.,]\d{3})?\s*-->\s*")
SPEAKER = re.compile(r"^([^:]{1,80}):\s*(.+)$")


def _normalise_time(value: str) -> str:
    return ":".join(part.zfill(2) for part in value.split(":"))


def _seconds(timestamp: str) -> int:
    parts = [int(part) for part in timestamp.split(":")]
    return parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else parts[0] * 60 + parts[1]


def _slug(value: str) -> str:
    clean = re.sub(r"\.[^.]+$", "", value.lower())
    clean = re.sub(r"[^a-z0-9]+", "-", clean).strip("-")
    return clean[:48] or "transcript"


def parse_transcript(filename: str, raw: str) -> tuple[TranscriptRecord, list[TranscriptChunk], list[Document]]:
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()
    lines = text.split("\n")
    expert_line = next((line for line in lines if re.match(r"^\s*Expert\s*\d*\s*[–—-]", line, re.I)), None)
    role_line = next((line for line in lines if re.match(r"^\s*Role\s*:", line, re.I)), None)
    market_line = next((line for line in lines if re.match(r"^\s*Market\s*:", line, re.I)), None)
    title = re.sub(r"^\s*Expert\s*\d*\s*[–—-]\s*", "", expert_line, flags=re.I).strip() if expert_line else re.sub(r"[_-]+", " ", re.sub(r"\.[^.]+$", "", filename))
    market = re.sub(r"^\s*Market\s*:\s*", "", market_line, flags=re.I).strip() if market_line else None
    transcript_id = f"{_slug(filename)}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"

    turns: list[dict[str, str]] = []
    timestamp, speaker, buffer = "00:00", title, []

    def flush() -> None:
        nonlocal buffer
        content = re.sub(r"\s+", " ", " ".join(buffer)).strip()
        if content:
            turns.append({"timestamp": timestamp, "speaker": speaker, "content": content})
        buffer = []

    for source_line in lines:
        line = source_line.strip()
        if not line or line in {expert_line, role_line, market_line}:
            if not line:
                flush()
            continue
        subtitle = SUBTITLE_TIME.match(line)
        if subtitle:
            flush()
            timestamp = _normalise_time(subtitle.group(1))
            continue
        if line.isdigit() or line.upper() == "WEBVTT":
            continue
        timed = TIMESTAMP.match(line)
        if timed:
            flush()
            timestamp = _normalise_time(timed.group(1))
            remainder = timed.group(2).strip()
            labelled = SPEAKER.match(remainder)
            if labelled:
                speaker, remainder = labelled.group(1).strip(), labelled.group(2)
            if remainder:
                buffer.append(remainder)
            continue
        labelled = SPEAKER.match(line)
        if labelled:
            flush()
            speaker = labelled.group(1).strip()
            buffer.append(labelled.group(2))
        else:
            buffer.append(line)
    flush()

    if not turns and text:
        for index, paragraph in enumerate(filter(None, re.split(r"\n\s*\n", text))):
            turns.append({"timestamp": f"P{index + 1}", "speaker": title, "content": re.sub(r"\s+", " ", paragraph).strip()})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,
        chunk_overlap=180,
        separators=["\n\n", ". ", "? ", "! ", "; ", ", ", " "],
    )
    chunks: list[TranscriptChunk] = []
    documents: list[Document] = []
    for turn_index, turn in enumerate(turns):
        parts = splitter.split_text(turn["content"]) or [turn["content"]]
        for part_index, content in enumerate(parts):
            evidence_id = f"{transcript_id}-{turn_index + 1}-{part_index + 1}"
            chunk = TranscriptChunk(
                id=evidence_id,
                transcriptId=transcript_id,
                filename=filename,
                title=title,
                speaker=turn["speaker"],
                market=market,
                timestamp=turn["timestamp"],
                startSeconds=turn_index if turn["timestamp"].startswith("P") else _seconds(turn["timestamp"]),
                content=content,
            )
            chunks.append(chunk)
            documents.append(Document(page_content=content, metadata=chunk.model_dump(exclude={"content"})))

    record = TranscriptRecord(
        id=transcript_id,
        filename=filename,
        title=title,
        market=market,
        chunkCount=len(chunks),
        characterCount=len(text),
        createdAt=datetime.now(timezone.utc).isoformat(),
    )
    return record, chunks, documents


def parse_questions(text: str) -> list[str]:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n") if line.strip()]
    questions: list[str] = []
    for line in lines:
        clean = re.sub(r"^[-*•]\s*", "", line)
        clean = re.sub(r"^\d+[.)]\s*", "", clean)
        if clean.endswith("?") or re.match(r"^(how|what|why|when|where|which|who|do|does|did|is|are|can|could|would|should)\b", clean, re.I):
            if clean not in questions:
                questions.append(clean)
    if not questions:
        questions = list(dict.fromkeys(match.strip() for match in re.findall(r"[^?\n]+\?", text)))
    return questions[:12]
