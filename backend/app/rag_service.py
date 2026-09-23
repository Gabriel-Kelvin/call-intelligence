import json
import shutil
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .config import Settings
from .models import Citation, TranscriptChunk, TranscriptRecord
from .transcript_parser import parse_questions, parse_transcript


class RagService:
    """Owns ingestion, vector retrieval, grounding, and persisted corpus metadata."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.RLock()
        self.records: dict[str, TranscriptRecord] = {}
        self.chunks: dict[str, TranscriptChunk] = {}
        self._embeddings: FastEmbedEmbeddings | None = None
        self._client: QdrantClient | None = None
        self._vector_store: QdrantVectorStore | None = None
        self._load_registry()

    @property
    def registry_path(self) -> Path:
        return self.settings.qdrant_path.parent / "corpus.json"

    @property
    def embeddings(self) -> FastEmbedEmbeddings:
        if self._embeddings is None:
            self._embeddings = FastEmbedEmbeddings(model_name=self.settings.embedding_model)
        return self._embeddings

    @property
    def llm(self) -> ChatGroq:
        if not self.settings.groq_api_key:
            raise RuntimeError("The AI connection has not been configured.")
        return ChatGroq(
            api_key=self.settings.groq_api_key,
            model=self.settings.groq_model,
            temperature=0.1,
            max_retries=2,
        )

    def _ensure_store(self) -> QdrantVectorStore:
        if self._vector_store is not None:
            return self._vector_store
        self.settings.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self.settings.qdrant_path))
        if not self._client.collection_exists(self.settings.qdrant_collection):
            dimension = len(self.embeddings.embed_query("vector dimension probe"))
            self._client.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=self.settings.qdrant_collection,
            embedding=self.embeddings,
        )
        return self._vector_store

    def _load_registry(self) -> None:
        if not self.registry_path.exists():
            return
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            self.records = {item["id"]: TranscriptRecord.model_validate(item) for item in data.get("records", [])}
            self.chunks = {item["id"]: TranscriptChunk.model_validate(item) for item in data.get("chunks", [])}
        except (OSError, ValueError, KeyError):
            self.records, self.chunks = {}, {}

    def _save_registry(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": [record.model_dump() for record in self.records.values()],
            "chunks": [chunk.model_dump() for chunk in self.chunks.values()],
        }
        temporary = self.registry_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.registry_path)

    def summary(self) -> dict[str, Any]:
        records = sorted(self.records.values(), key=lambda item: item.createdAt)
        return {
            "transcripts": [record.model_dump() for record in records],
            "transcriptCount": len(records),
            "chunkCount": len(self.chunks),
            "characterCount": sum(record.characterCount for record in records),
        }

    def clear(self) -> dict[str, Any]:
        with self.lock:
            if self._client is not None:
                self._client.close()
            self._client = None
            self._vector_store = None
            self.records = {}
            self.chunks = {}
            if self.settings.qdrant_path.exists():
                shutil.rmtree(self.settings.qdrant_path)
            if self.registry_path.exists():
                self.registry_path.unlink()
            return self.summary()

    def ingest(self, uploads: list[tuple[str, str]]) -> dict[str, Any]:
        with self.lock:
            self.clear()
            try:
                documents: list[Document] = []
                ids: list[str] = []
                for filename, text in uploads:
                    record, chunks, parsed_documents = parse_transcript(filename, text)
                    self.records[record.id] = record
                    self.chunks.update({chunk.id: chunk for chunk in chunks})
                    documents.extend(parsed_documents)
                    ids.extend(str(uuid.uuid5(uuid.NAMESPACE_URL, f"call-intelligence:{chunk.id}")) for chunk in chunks)
                if not documents:
                    raise ValueError("No usable transcript content was found.")
                self._ensure_store().add_documents(documents=documents, ids=ids)
                self._save_registry()
                return self.summary()
            except Exception:
                self.clear()
                raise

    def _json_call(self, system: str, user: str, max_tokens: int = 1000) -> dict[str, Any]:
        model = self.llm.bind(response_format={"type": "json_object"}, max_tokens=max_tokens)
        response = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        content = response.content
        if isinstance(content, list):
            content = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
        clean = str(content).strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean.strip() or "{}")

    def retrieve(self, queries: list[str], limit: int = 20) -> list[tuple[TranscriptChunk, float]]:
        if not self.chunks:
            return []
        scores: dict[str, float] = {}
        for query_index, query in enumerate(queries):
            rows = self._ensure_store().similarity_search_with_relevance_scores(query, k=min(limit, len(self.chunks)))
            for document, score in rows:
                evidence_id = str(document.metadata.get("id", ""))
                if evidence_id in self.chunks:
                    weighted = float(score) / (query_index + 1)
                    scores[evidence_id] = max(scores.get(evidence_id, 0.0), weighted)
        ranked = sorted(((self.chunks[key], score) for key, score in scores.items()), key=lambda item: item[1], reverse=True)
        selected: list[tuple[TranscriptChunk, float]] = []
        per_transcript: dict[str, int] = defaultdict(int)
        for row in ranked:
            if per_transcript[row[0].transcriptId] >= 5:
                continue
            selected.append(row)
            per_transcript[row[0].transcriptId] += 1
            if len(selected) >= limit:
                break
        for row in ranked:
            if len(selected) >= limit:
                break
            if row not in selected:
                selected.append(row)
        return selected

    @staticmethod
    def _evidence_block(rows: list[tuple[TranscriptChunk, float]]) -> str:
        return "\n\n".join(
            f"[{chunk.id}] transcript={chunk.transcriptId} | file={chunk.filename} | expert={chunk.title} | "
            f"speaker={chunk.speaker} | market={chunk.market or 'unknown'} | time={chunk.timestamp} | vector_score={score:.4f}\n{chunk.content}"
            for chunk, score in rows
        )

    def _citations(self, evidence_ids: list[str], allowed: set[str] | None = None) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for evidence_id in dict.fromkeys(evidence_ids):
            if allowed is not None and evidence_id not in allowed:
                continue
            chunk = self.chunks.get(evidence_id)
            if not chunk:
                continue
            citations.append(Citation(
                evidenceId=chunk.id,
                filename=chunk.filename,
                title=chunk.title,
                speaker=chunk.speaker,
                market=chunk.market,
                timestamp=chunk.timestamp,
                quote=chunk.content,
            ).model_dump())
        return citations

    def ask(self, question: str) -> dict[str, Any]:
        if not self.chunks:
            raise ValueError("Upload transcripts before asking a question.")
        queries = [question]
        try:
            expanded = self._json_call(
                "Rewrite the research question into up to four complementary semantic-search queries. Preserve names, dates and technical terms. Do not answer. Return JSON only: {\"queries\":[\"...\"]}.",
                question,
                300,
            ).get("queries", [])
            queries = list(dict.fromkeys([question, *[str(item) for item in expanded if item]]))[:5]
        except Exception:
            pass
        candidates = self.retrieve(queries, 20)
        if not candidates:
            return {"answer": "I could not find relevant evidence in the indexed transcripts.", "citations": [], "confidence": "low", "retrieval": {"queries": queries, "candidates": 0}}
        candidate_block = self._evidence_block(candidates)
        selected = candidates[:10]
        try:
            reranked = self._json_call(
                "You are a strict evidence reranker. Select only passages that directly help answer the question. Reject generic word overlap. Return JSON only: {\"evidenceIds\":[\"exact supplied ID\"]}.",
                f"QUESTION:\n{question}\n\nCANDIDATES:\n{candidate_block}",
                450,
            ).get("evidenceIds", [])
            requested = set(str(item) for item in reranked)
            selected = [row for row in candidates if row[0].id in requested][:10]
        except Exception:
            pass
        if not selected:
            return {"answer": "The indexed transcripts do not contain enough relevant evidence to answer that question.", "citations": [], "confidence": "low", "model": self.settings.groq_model, "retrieval": {"queries": queries, "candidates": len(candidates), "selected": 0, "cited": 0, "strategy": "multi-query vector search and LLM reranking"}}
        generated = self._json_call(
            "You are an evidence-first research analyst. Answer only from the supplied transcript excerpts. Never invent facts, quotes, speakers, timestamps, or evidence IDs. Every substantive claim must be supported by supplied evidence. State clearly when evidence is insufficient. Return JSON only: {\"answer\":\"2-6 concise sentences\",\"evidenceIds\":[\"exact supplied ID\"],\"confidence\":\"high|medium|low\"}.",
            f"QUESTION:\n{question}\n\nRETRIEVED EVIDENCE:\n{self._evidence_block(selected)}",
            900,
        )
        allowed = {chunk.id for chunk, _ in selected}
        citations = self._citations([str(item) for item in generated.get("evidenceIds", [])], allowed)[:8]
        return {
            "answer": generated.get("answer") or "The evidence was not sufficient for a verified answer.",
            "citations": citations,
            "confidence": generated.get("confidence", "medium"),
            "model": f"{self.settings.groq_model} · Groq",
            "retrieval": {"queries": queries, "candidates": len(candidates), "selected": len(selected), "cited": len(citations), "strategy": "multi-query Qdrant vector search and LLM reranking"},
        }

    def insights(self) -> dict[str, Any]:
        if not self.chunks:
            raise ValueError("Upload transcripts before generating insights.")
        by_transcript: dict[str, list[TranscriptChunk]] = defaultdict(list)
        for chunk in self.chunks.values():
            by_transcript[chunk.transcriptId].append(chunk)
        selected = [chunk for chunks in by_transcript.values() for chunk in chunks[:30]]
        rows, used = [], 0
        for chunk in selected:
            row = f"[{chunk.id}] transcript={chunk.transcriptId} | expert={chunk.title} | market={chunk.market or 'unknown'} | speaker={chunk.speaker} | time={chunk.timestamp}\n{chunk.content}"
            if used + len(row) > 55_000:
                break
            rows.append(row)
            used += len(row)
        generated = self._json_call(
            "Analyse expert-call transcripts across calls. Identify only genuinely recurring themes and meaningful disagreements. Do not invent consensus. Every item must cite exact evidence IDs from at least two transcripts when possible. Return JSON only: {\"themes\":[{\"title\":\"...\",\"summary\":\"...\",\"evidenceIds\":[\"...\"]}],\"disagreements\":[{\"title\":\"...\",\"summary\":\"...\",\"evidenceIds\":[\"...\"]}]}. Return at most four themes and three disagreements.",
            "TRANSCRIPT EVIDENCE:\n" + "\n\n".join(rows),
            1400,
        )
        allowed = {chunk.id for chunk in selected}

        def hydrate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            output = []
            for item in items:
                citations = self._citations([str(value) for value in item.get("evidenceIds", [])], allowed)[:8]
                if item.get("title") and item.get("summary") and citations:
                    output.append({"title": item["title"], "summary": item["summary"], "citations": citations})
            return output

        return {"themes": hydrate(generated.get("themes", [])), "disagreements": hydrate(generated.get("disagreements", [])), "transcriptCount": len(by_transcript), "chunkCount": len(self.chunks)}

    def answer_guide(self, text: str) -> dict[str, Any]:
        questions = parse_questions(text)
        if not questions:
            raise ValueError("No interview questions were detected.")
        if not self.chunks:
            raise ValueError("Upload transcripts before answering an interview guide.")
        all_chunks = list(self.chunks.values())
        compact = sum(len(chunk.content) for chunk in all_chunks) <= 45_000
        candidate_sets: list[list[TranscriptChunk]] = []
        for question in questions:
            candidate_sets.append(all_chunks if compact else [chunk for chunk, _ in self.retrieve([question], 15)])
        sections = []
        for index, question in enumerate(questions):
            evidence = "\n\n".join(
                f"[{chunk.id}] transcript={chunk.transcriptId} | expert={chunk.title} | market={chunk.market or 'unknown'} | speaker={chunk.speaker} | time={chunk.timestamp}\n{chunk.content}"
                for chunk in candidate_sets[index]
            )
            sections.append(f"QUESTION {index + 1}: {question}\n{evidence}")
        generated = self._json_call(
            "Answer every interview-guide question separately for every expert transcript with relevant evidence. Use only supplied excerpts. Do not merge experts or cite another expert's evidence. Omit transcripts without evidence. Keep each answer to 1-3 sentences. Return JSON only: {\"answers\":[{\"questionIndex\":1,\"responses\":[{\"transcriptId\":\"exact transcript ID\",\"answer\":\"...\",\"evidenceIds\":[\"exact evidence ID\"],\"confidence\":\"high|medium|low\"}]}]}.",
            "INTERVIEW GUIDE WITH EVIDENCE:\n" + "\n\n---\n\n".join(sections),
            3000,
        )
        answer_map = {item.get("questionIndex"): item for item in generated.get("answers", [])}
        answers = []
        for index, question in enumerate(questions):
            allowed = {chunk.id for chunk in candidate_sets[index]}
            responses = []
            for response in answer_map.get(index + 1, {}).get("responses", []):
                transcript_id = str(response.get("transcriptId", ""))
                citations = [item for item in self._citations([str(value) for value in response.get("evidenceIds", [])], allowed) if self.chunks[item["evidenceId"]].transcriptId == transcript_id]
                if response.get("answer") and citations:
                    record = self.records.get(transcript_id)
                    responses.append({"transcriptId": transcript_id, "expert": record.title if record else citations[0]["title"], "market": record.market if record else citations[0].get("market"), "answer": response["answer"], "confidence": response.get("confidence", "medium"), "citations": citations})
            answers.append({"question": question, "responses": responses})
        first = next((line.strip() for line in text.splitlines() if line.strip()), "Interview guide")
        title = first if len(first) < 100 and not first.endswith("?") else "Interview guide"
        return {"title": title, "questions": questions, "answers": answers}
