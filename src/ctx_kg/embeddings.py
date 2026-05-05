from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass


DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "embeddinggemma"
DEFAULT_VOYAGE_EMBEDDING_MODEL = "voyage-code-3"


@dataclass
class EmbeddingProvider:
    provider: str
    model: str
    dimensions: int | None = None

    def embed(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        if self.provider in {"openai", "openai-compatible"}:
            return embed_openai_compatible(texts, self.model, self.provider, self.dimensions)
        if self.provider == "ollama":
            return embed_ollama(texts, self.model, self.dimensions)
        if self.provider == "voyage":
            return embed_voyage(texts, self.model, input_type or "document", self.dimensions)
        if self.provider in {"claude", "anthropic"}:
            raise ValueError("Claude/Anthropic does not provide embedding models. Use --provider voyage for Anthropic's recommended embedding provider.")
        raise ValueError(f"unsupported embedding provider: {self.provider}")


def provider_from_env(provider: str | None = None, model: str | None = None, dimensions: int | None = None) -> EmbeddingProvider:
    selected_provider = provider or os.environ.get("CTX_EMBED_PROVIDER") or "openai"
    if selected_provider in {"claude", "anthropic"}:
        raise ValueError("Claude/Anthropic does not provide embedding models. Use --provider voyage for Anthropic's recommended embedding provider.")
    if selected_provider == "ollama":
        default_model = DEFAULT_OLLAMA_EMBEDDING_MODEL
    elif selected_provider == "voyage":
        default_model = DEFAULT_VOYAGE_EMBEDDING_MODEL
    else:
        default_model = DEFAULT_OPENAI_EMBEDDING_MODEL
    selected_model = model or os.environ.get("CTX_EMBED_MODEL") or default_model
    selected_dimensions = dimensions
    if selected_dimensions is None and os.environ.get("CTX_EMBED_DIMENSIONS"):
        selected_dimensions = int(os.environ["CTX_EMBED_DIMENSIONS"])
    return EmbeddingProvider(provider=selected_provider, model=selected_model, dimensions=selected_dimensions)


def embed_openai_compatible(texts: list[str], model: str, provider: str, dimensions: int | None = None) -> list[list[float]]:
    if provider == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("OPENAI_API_KEY")
    else:
        base_url = os.environ.get("CTX_EMBED_BASE_URL")
        api_key = os.environ.get("CTX_EMBED_API_KEY")
        if not base_url:
            raise RuntimeError("CTX_EMBED_BASE_URL is required for openai-compatible embeddings")
    if not api_key:
        raise RuntimeError("an API key is required: set OPENAI_API_KEY or CTX_EMBED_API_KEY")
    payload: dict = {"model": model, "input": texts, "encoding_format": "float"}
    if dimensions:
        payload["dimensions"] = dimensions
    request = urllib.request.Request(
        base_url.rstrip("/") + "/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    return [item["embedding"] for item in sorted(body["data"], key=lambda item: item["index"])]


def embed_ollama(texts: list[str], model: str, dimensions: int | None = None) -> list[list[float]]:
    base_url = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("CTX_OLLAMA_URL") or "http://localhost:11434"
    payload: dict = {"model": model, "input": texts}
    if dimensions:
        payload["dimensions"] = dimensions
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/embed",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["embeddings"]


def embed_voyage(texts: list[str], model: str, input_type: str, dimensions: int | None = None) -> list[list[float]]:
    api_key = os.environ.get("VOYAGE_API_KEY") or os.environ.get("CTX_VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY is required for Voyage embeddings")
    payload: dict = {"model": model, "input": texts, "input_type": input_type}
    if dimensions:
        payload["output_dimension"] = dimensions
    request = urllib.request.Request(
        "https://api.voyageai.com/v1/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return [item["embedding"] for item in sorted(body["data"], key=lambda item: item["index"])]
