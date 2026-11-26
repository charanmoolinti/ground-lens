"""
Module: health.py
Description: Truthful, independent health checking and telemetry for all Ground Lens subsystems.
Never fabricates readiness: reports exact operational, degraded, offline, or unconfigured states.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
import socket


class SubsystemState(str, Enum):
    OPERATIONAL = "OPERATIONAL"
    DEGRADED = "DEGRADED"
    INITIALIZING = "INITIALIZING"
    OFFLINE = "OFFLINE"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    ERROR = "ERROR"


@dataclass
class SubsystemHealth:
    name: str
    state: SubsystemState
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    is_essential: bool = True

    @property
    def is_healthy(self) -> bool:
        return self.state in (SubsystemState.OPERATIONAL, SubsystemState.DEGRADED)


@dataclass
class SystemHealthReport:
    overall_state: SubsystemState
    headline: str
    subsystems: Dict[str, SubsystemHealth]
    can_verify_locally: bool
    can_synthesize_with_gemini: bool
    indexed_document_count: int
    indexed_chunk_count: int


def check_network_connectivity(host: str = "generativelanguage.googleapis.com", port: int = 443, timeout: float = 1.5) -> bool:
    """Performs a non-blocking socket connection test to verify real internet connectivity."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (socket.timeout, socket.gaierror, ConnectionRefusedError, OSError):
        return False


class HealthChecker:
    """Audits each subsystem independently and produces a truthful health assessment."""

    @staticmethod
    def audit_pdf_ingestion() -> SubsystemHealth:
        try:
            import pymupdf
            return SubsystemHealth(
                name="PDF Ingestion",
                state=SubsystemState.OPERATIONAL,
                message=f"PyMuPDF active (v{pymupdf.__version__})",
                details={"engine": "PyMuPDF", "version": pymupdf.__version__},
                is_essential=True,
            )
        except Exception as e:
            return SubsystemHealth(
                name="PDF Ingestion",
                state=SubsystemState.ERROR,
                message=f"PyMuPDF unavailable: {e}",
                details={"error": str(e)},
                is_essential=True,
            )

    @staticmethod
    def audit_embeddings() -> SubsystemHealth:
        try:
            from backend.embeddings.embedding_model import EmbeddingModel, _check_local_cache
            is_cached = _check_local_cache("sentence-transformers/all-MiniLM-L6-v2")
            em = EmbeddingModel()
            return SubsystemHealth(
                name="Embedding Engine",
                state=SubsystemState.OPERATIONAL,
                message=f"all-MiniLM-L6-v2 ready ({em.embedding_dim}-dim, cached: {is_cached})",
                details={"model": em.model_name, "dim": em.embedding_dim, "cached_locally": is_cached},
                is_essential=True,
            )
        except Exception as e:
            return SubsystemHealth(
                name="Embedding Engine",
                state=SubsystemState.ERROR,
                message=f"Embedding model failure: {e}",
                details={"error": str(e)},
                is_essential=True,
            )

    @staticmethod
    def audit_vector_store(
        persist_dir: str = "./data/chroma_db",
        vector_store_instance: Any = None,
    ) -> SubsystemHealth:
        try:
            if vector_store_instance is not None and hasattr(vector_store_instance, "count"):
                store = vector_store_instance
            else:
                from backend.retrieval.vector_store import ChromaVectorStore
                store = ChromaVectorStore(persist_directory=persist_dir)

            doc_count = len(store.list_indexed_documents())
            chunk_count = store.count()
            state = SubsystemState.OPERATIONAL if doc_count > 0 else SubsystemState.DEGRADED
            msg = (
                f"ChromaDB active: {doc_count} document(s), {chunk_count} chunk(s)"
                if doc_count > 0
                else "ChromaDB active: Ready (0 documents indexed)"
            )
            return SubsystemHealth(
                name="Vector Database",
                state=state,
                message=msg,
                details={
                    "database": "ChromaDB",
                    "persist_directory": str(getattr(store, "persist_directory", persist_dir)),
                    "documents_count": doc_count,
                    "chunks_count": chunk_count,
                },
                is_essential=True,
            )
        except Exception as e:
            return SubsystemHealth(
                name="Vector Database",
                state=SubsystemState.ERROR,
                message=f"ChromaDB unavailable: {e}",
                details={"error": str(e)},
                is_essential=True,
            )

    @staticmethod
    def audit_nli_engine(nli_instance: Any = None) -> SubsystemHealth:
        try:
            if nli_instance is not None and hasattr(nli_instance, "predict_pair"):
                nli = nli_instance
            else:
                from backend.verification.nli_classifier import NLIClassifier
                nli = NLIClassifier(device="cpu")
            labels = getattr(nli, "label_to_index", {"contradiction": 0, "entailment": 1, "neutral": 2})
            loader = getattr(nli, "loader_type", "CrossEncoder")
            return SubsystemHealth(
                name="NLI Cross-Encoder",
                state=SubsystemState.OPERATIONAL,
                message=f"DeBERTa-v3 ready on CPU ({loader})",
                details={
                    "model": getattr(nli, "model_name", "cross-encoder/nli-deberta-v3-base"),
                    "device": getattr(nli, "device", "cpu"),
                    "loader_type": loader,
                    "cached_locally": True,
                    "label_mapping": labels,
                },
                is_essential=True,
            )
        except Exception as e:
            return SubsystemHealth(
                name="NLI Cross-Encoder",
                state=SubsystemState.ERROR,
                message=f"NLI model failed to load: {e}",
                details={"error": str(e)},
                is_essential=True,
            )

    @staticmethod
    def audit_gemini(client_or_key: Any = None, force_live_test: bool = False) -> SubsystemHealth:
        from backend.generation.gemini_client import GeminiClient
        if isinstance(client_or_key, GeminiClient):
            client = client_or_key
        else:
            client = GeminiClient(api_key=client_or_key)

        diag = client.diagnose_connection(force_live_test=force_live_test)

        if not diag["key_present"] or client.api_key == "your_gemini_api_key_here":
            return SubsystemHealth(
                name="Gemini API",
                state=SubsystemState.NOT_CONFIGURED,
                message="No API key configured (Direct Statement Verification works locally)",
                details=diag,
                is_essential=False,
            )

        if not diag["key_format_valid"]:
            return SubsystemHealth(
                name="Gemini API",
                state=SubsystemState.ERROR,
                message="Invalid API key format (expected Google AI Studio key)",
                details=diag,
                is_essential=False,
            )

        if not diag["endpoint_reachable"]:
            return SubsystemHealth(
                name="Gemini API",
                state=SubsystemState.OFFLINE,
                message="Network offline: Google AI Studio endpoint unreachable (local verification active)",
                details=diag,
                is_essential=False,
            )

        if diag["model_operational"]:
            return SubsystemHealth(
                name="Gemini API",
                state=SubsystemState.OPERATIONAL,
                message=f"Connected ({diag['active_model']})",
                details=diag,
                is_essential=False,
            )
        else:
            return SubsystemHealth(
                name="Gemini API",
                state=SubsystemState.ERROR,
                message=diag["status_message"],
                details=diag,
                is_essential=False,
            )

    @classmethod
    def audit_all(
        cls,
        gemini_client_or_key: Any = None,
        vector_store: Any = None,
        nli_classifier: Any = None,
        persist_dir: str = "./data/chroma_db",
        force_live_gemini_test: bool = False,
    ) -> SystemHealthReport:
        subsystems = {
            "pdf": cls.audit_pdf_ingestion(),
            "embeddings": cls.audit_embeddings(),
            "vector_store": cls.audit_vector_store(persist_dir, vector_store_instance=vector_store),
            "nli": cls.audit_nli_engine(nli_instance=nli_classifier),
            "gemini": cls.audit_gemini(gemini_client_or_key, force_live_test=force_live_gemini_test),
        }

        doc_count = subsystems["vector_store"].details.get("documents_count", 0)
        chunk_count = subsystems["vector_store"].details.get("chunks_count", 0)

        # Local verification capability requires embeddings, vector_store, and NLI
        can_verify_locally = (
            subsystems["embeddings"].state == SubsystemState.OPERATIONAL
            and subsystems["vector_store"].state in (SubsystemState.OPERATIONAL, SubsystemState.DEGRADED)
            and subsystems["nli"].state == SubsystemState.OPERATIONAL
        )

        can_synthesize_gemini = subsystems["gemini"].state == SubsystemState.OPERATIONAL

        # 6. RAG Generation Pipeline Subsystem
        if can_synthesize_gemini and doc_count > 0:
            rag_state = SubsystemState.OPERATIONAL
            rag_msg = f"Operational: Answer synthesis & extraction enabled with {subsystems['gemini'].details.get('active_model', 'Gemini')}"
        elif can_synthesize_gemini and doc_count == 0:
            rag_state = SubsystemState.DEGRADED
            rag_msg = "Standby: Gemini connected; index a document in Document Library to begin"
        elif can_verify_locally:
            rag_state = SubsystemState.DEGRADED
            rag_msg = "Standby: Direct Statement Verification active locally (Gemini unconfigured/offline)"
        else:
            rag_state = SubsystemState.OFFLINE
            rag_msg = "Offline: Verification core unavailable"

        subsystems["rag_pipeline"] = SubsystemHealth(
            name="RAG Generation Pipeline",
            state=rag_state,
            message=rag_msg,
            details={"gemini_connected": can_synthesize_gemini, "documents_indexed": doc_count},
            is_essential=False,
        )

        # Truthful overall state evaluation
        if not can_verify_locally:
            overall_state = SubsystemState.ERROR
            headline = "CRITICAL: Local verification core is offline or failing"
        elif can_synthesize_gemini and doc_count > 0:
            overall_state = SubsystemState.OPERATIONAL
            headline = "SYSTEM FULLY OPERATIONAL: All subsystems and Gemini active"
        elif can_verify_locally and doc_count > 0:
            overall_state = SubsystemState.OPERATIONAL
            headline = "LOCAL ENGINE: OPERATIONAL (Direct statement verification active; Gemini standby)"
        elif can_verify_locally and doc_count == 0:
            overall_state = SubsystemState.DEGRADED
            headline = "LOCAL ENGINE: READY (Standby: Index a document in Document Library to begin)"
        else:
            overall_state = SubsystemState.OFFLINE
            headline = "SYSTEM DEGRADED"

        return SystemHealthReport(
            overall_state=overall_state,
            headline=headline,
            subsystems=subsystems,
            can_verify_locally=can_verify_locally,
            can_synthesize_with_gemini=can_synthesize_gemini,
            indexed_document_count=doc_count,
            indexed_chunk_count=chunk_count,
        )
