"""Model-level embed methods must tag document vs query intent."""

import pytest

from graphiti_core.edges import EntityEdge, create_entity_edge_embeddings
from graphiti_core.embedder.client import EmbedderClient, TaskType
from graphiti_core.nodes import EntityNode, create_entity_node_embeddings
from graphiti_core.utils.datetime_utils import utc_now


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
