from __future__ import annotations

from collections.abc import Iterable
import os

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from langchain_openai import AzureOpenAIEmbeddings
from loguru import logger
from tqdm.auto import tqdm

from .chunking import ChunkRecord
from .config import ICDRAGConfig


VECTOR_FIELD_NAME = "content_vector"


def derive_azure_endpoint(openai_api_base: str) -> str:
    if openai_api_base.endswith("/openai/v1"):
        return openai_api_base[: -len("/openai/v1")]
    return openai_api_base.rstrip("/")


def build_embeddings_client(config: ICDRAGConfig) -> AzureOpenAIEmbeddings:
    original_openai_api_base = None
    if "OPENAI_API_BASE" in os.environ:
        original_openai_api_base = os.environ.pop("OPENAI_API_BASE")
    try:
        return AzureOpenAIEmbeddings(
            model=config.embedding_deployment,
            deployment=config.embedding_deployment,
            azure_endpoint=derive_azure_endpoint(config.openai_api_base),
            openai_api_key=config.openai_api_key,
            openai_api_type=config.openai_api_type,
            openai_api_version=config.openai_api_version,
        )
    finally:
        if original_openai_api_base is not None:
            os.environ["OPENAI_API_BASE"] = original_openai_api_base


def build_index_schema(index_name: str, vector_dimensions: int) -> SearchIndex:
    return SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True, filterable=True, sortable=True),
            SearchableField(name="source_type", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="document_title", type=SearchFieldDataType.String),
            SearchableField(name="chunk_title", type=SearchFieldDataType.String),
            SearchableField(name="semantic_path", type=SearchFieldDataType.String),
            SimpleField(name="code", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="source_path", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="text", type=SearchFieldDataType.String),
            SimpleField(name="estimated_tokens", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SearchField(
                name=VECTOR_FIELD_NAME,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_dimensions,
                vector_search_profile_name="vector-profile",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
            profiles=[VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw-config")],
        ),
    )


def chunked(items: list[dict[str, object]], size: int) -> Iterable[list[dict[str, object]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def create_clients(config: ICDRAGConfig) -> tuple[SearchIndexClient, SearchClient]:
    credential = AzureKeyCredential(config.search_service.api_key)
    return (
        SearchIndexClient(endpoint=config.search_service.endpoint, credential=credential),
        SearchClient(endpoint=config.search_service.endpoint, index_name=config.index_name, credential=credential),
    )


def ensure_index(index_client: SearchIndexClient, index_name: str, vector_dimensions: int, recreate: bool) -> None:
    if recreate:
        logger.info("Recreating Azure Search index {}", index_name)
        try:
            index_client.delete_index(index_name)
        except ResourceNotFoundError:
            pass
        index_client.create_index(build_index_schema(index_name, vector_dimensions))
        return

    try:
        index_client.get_index(index_name)
        logger.info("Using existing Azure Search index {}", index_name)
    except ResourceNotFoundError:
        logger.info("Creating Azure Search index {}", index_name)
        index_client.create_index(build_index_schema(index_name, vector_dimensions))


def embed_and_prepare_documents(config: ICDRAGConfig, chunks: list[ChunkRecord]) -> tuple[list[dict[str, object]], int]:
    logger.info(
        "Embedding {} chunks with batch size {} using deployment {}",
        len(chunks),
        config.embedding_batch_size,
        config.embedding_deployment,
    )
    embeddings = build_embeddings_client(config)
    documents: list[dict[str, object]] = []
    vector_dimensions = 0
    for start in tqdm(range(0, len(chunks), config.embedding_batch_size), desc="embedding batches", unit="batch"):
        batch = chunks[start : start + config.embedding_batch_size]
        vectors = embeddings.embed_documents([chunk.text for chunk in batch])
        for chunk, vector in zip(batch, vectors, strict=True):
            doc = chunk.to_document()
            doc[VECTOR_FIELD_NAME] = vector
            documents.append(doc)
            vector_dimensions = len(vector)
        logger.info("Prepared {} embedded documents so far", len(documents))
    return documents, vector_dimensions


def upload_documents(search_client: SearchClient, config: ICDRAGConfig, documents: list[dict[str, object]]) -> None:
    logger.info("Uploading {} documents to Azure Search with batch size {}", len(documents), config.upload_batch_size)
    batches = list(chunked(documents, config.upload_batch_size))
    for batch in tqdm(batches, desc="upload batches", unit="batch"):
        results = search_client.merge_or_upload_documents(batch)
        failed = [result.key for result in results if not result.succeeded]
        if failed:
            raise RuntimeError(f"Azure Search upload failed for {len(failed)} document(s): {failed[:5]}")
    logger.info("Uploaded {} documents successfully", len(documents))


def search_index(config: ICDRAGConfig, query_text: str, top_k: int) -> list[dict[str, object]]:
    logger.info("Running Azure Search query against {} with top_k={}", config.index_name, top_k)
    _, search_client = create_clients(config)
    embeddings = build_embeddings_client(config)
    query_vector = embeddings.embed_query(query_text)
    results = list(
        search_client.search(
            search_text=query_text,
            vector_queries=[
                {
                    "kind": "vector",
                    "vector": query_vector,
                    "fields": VECTOR_FIELD_NAME,
                    "k": top_k,
                }
            ],
            top=top_k,
            select=[
                "chunk_id",
                "source_type",
                "document_title",
                "chunk_title",
                "semantic_path",
                "code",
                "source_path",
                "text",
            ],
        )
    )
    payload: list[dict[str, object]] = []
    for result in results:
        payload.append(
            {
                "chunk_id": result.get("chunk_id"),
                "source_type": result.get("source_type"),
                "document_title": result.get("document_title"),
                "chunk_title": result.get("chunk_title"),
                "semantic_path": result.get("semantic_path"),
                "code": result.get("code"),
                "source_path": result.get("source_path"),
                "score": result.get("@search.score"),
                "text_preview": (result.get("text") or "")[:500],
            }
        )
    return payload