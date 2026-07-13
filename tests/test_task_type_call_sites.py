"""Model-level embed methods must tag document vs query intent."""

from unittest.mock import AsyncMock, Mock

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge, create_entity_edge_embeddings
from graphiti_core.embedder.client import EmbedderClient, TaskType
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.nodes import CommunityNode, EntityNode, create_entity_node_embeddings
from graphiti_core.tracer import NoOpTracer
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.node_operations import _semantic_candidate_search


class RecordingEmbedder(EmbedderClient):
    def __init__(self) -> None:
        self.calls: list[tuple[str, TaskType | None]] = []

    async def create(self, input_data, task_type: TaskType | None = None) -> list[float]:
        self.calls.append(('create', task_type))
        return [0.0] * 4

    async def create_batch(
        self, input_data_list, task_type: TaskType | None = None
    ) -> list[list[float]]:
        self.calls.append(('create_batch', task_type))
        return [[0.0] * 4 for _ in input_data_list]


def make_node(name: str = 'FalkorDB') -> EntityNode:
    return EntityNode(name=name, group_id='g', labels=['Entity'], created_at=utc_now())


def make_edge() -> EntityEdge:
    return EntityEdge(
        source_node_uuid='s',
        target_node_uuid='t',
        name='USES',
        fact='brent uses falkordb',
        group_id='g',
        created_at=utc_now(),
    )


def make_community_node(name: str = 'TestCommunity') -> CommunityNode:
    return CommunityNode(name=name, group_id='g', labels=[], created_at=utc_now())


def make_graphiti_clients(embedder: EmbedderClient) -> GraphitiClients:
    return GraphitiClients(
        driver=Mock(spec=GraphDriver),
        llm_client=Mock(spec=LLMClient),
        embedder=embedder,
        cross_encoder=Mock(spec=CrossEncoderClient),
        tracer=NoOpTracer(),
    )


@pytest.mark.asyncio
async def test_entity_node_name_embedding_is_document() -> None:
    embedder = RecordingEmbedder()
    await make_node().generate_name_embedding(embedder)
    assert embedder.calls == [('create', 'document')]


@pytest.mark.asyncio
async def test_entity_edge_fact_embedding_is_document() -> None:
    embedder = RecordingEmbedder()
    await make_edge().generate_embedding(embedder)
    assert embedder.calls == [('create', 'document')]


@pytest.mark.asyncio
async def test_bulk_node_embeddings_are_document() -> None:
    embedder = RecordingEmbedder()
    await create_entity_node_embeddings(embedder, [make_node(), make_node('Neo4j')])
    assert embedder.calls == [('create_batch', 'document')]


@pytest.mark.asyncio
async def test_bulk_edge_embeddings_are_document() -> None:
    embedder = RecordingEmbedder()
    await create_entity_edge_embeddings(embedder, [make_edge()])
    assert embedder.calls == [('create_batch', 'document')]


@pytest.mark.asyncio
async def test_community_node_name_embedding_is_document() -> None:
    embedder = RecordingEmbedder()
    await make_community_node().generate_name_embedding(embedder)
    assert embedder.calls == [('create', 'document')]


@pytest.mark.asyncio
async def test_semantic_candidate_search_batch_path_is_query(monkeypatch) -> None:
    embedder = RecordingEmbedder()
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.node_similarity_search',
        AsyncMock(return_value=[]),
    )
    clients = make_graphiti_clients(embedder)

    await _semantic_candidate_search(clients, [make_node('Foo')])

    assert embedder.calls == [('create_batch', 'query')]


@pytest.mark.asyncio
async def test_semantic_candidate_search_falls_back_to_create_with_query(monkeypatch) -> None:
    class RaisingBatchEmbedder(RecordingEmbedder):
        async def create_batch(
            self, input_data_list, task_type: TaskType | None = None
        ) -> list[list[float]]:
            raise NotImplementedError()

    embedder = RaisingBatchEmbedder()
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.node_similarity_search',
        AsyncMock(return_value=[]),
    )
    clients = make_graphiti_clients(embedder)

    await _semantic_candidate_search(clients, [make_node('Foo'), make_node('Bar')])

    assert embedder.calls == [('create', 'query'), ('create', 'query')]
