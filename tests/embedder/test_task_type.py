"""Pass-through embedders must accept task_type and ignore it (no input mutation)."""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.embedder.voyage import VoyageAIEmbedder, VoyageAIEmbedderConfig
from tests.embedder.embedder_fixtures import create_embedding_values


def payload_values(mock_call) -> list:
    """All positional + keyword argument values of the recorded API call."""
    return list(mock_call.args) + list(mock_call.kwargs.values())


@pytest.fixture
def mock_openai_client() -> Generator[Any, Any, None]:
    with patch('openai.AsyncOpenAI') as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.embeddings = MagicMock()
        mock_instance.embeddings.create = AsyncMock()
        yield mock_instance


@pytest.mark.asyncio
async def test_openai_accepts_and_ignores_task_type(mock_openai_client: Any) -> None:
    response = MagicMock()
    item = MagicMock()
    item.embedding = create_embedding_values()
    response.data = [item]
    mock_openai_client.embeddings.create.return_value = response

    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key='k'))
    embedder.client = mock_openai_client

    result = await embedder.create('hello world', task_type='document')

    assert len(result) > 0
    values = payload_values(mock_openai_client.embeddings.create.call_args)
    assert any(v == 'hello world' or v == ['hello world'] for v in values), (
        'input must reach the API unmodified (no prefix)'
    )


@pytest.mark.asyncio
async def test_azure_accepts_and_ignores_task_type() -> None:
    azure_client = MagicMock()
    response = MagicMock()
    item = MagicMock()
    item.embedding = create_embedding_values()
    response.data = [item]
    azure_client.embeddings = MagicMock()
    azure_client.embeddings.create = AsyncMock(return_value=response)

    embedder = AzureOpenAIEmbedderClient(azure_client=azure_client)
    result = await embedder.create('hello world', task_type='query')

    assert len(result) > 0
    values = payload_values(azure_client.embeddings.create.call_args)
    assert any(v == 'hello world' or v == ['hello world'] for v in values)


@pytest.mark.asyncio
async def test_voyage_accepts_and_ignores_task_type() -> None:
    with patch('voyageai.AsyncClient') as mock_client_cls:
        mock_instance = mock_client_cls.return_value
        response = MagicMock()
        response.embeddings = [create_embedding_values()]
        mock_instance.embed = AsyncMock(return_value=response)

        embedder = VoyageAIEmbedder(config=VoyageAIEmbedderConfig(api_key='k'))
        embedder.client = mock_instance
        result = await embedder.create('hello world', task_type='document')

    assert len(result) > 0
    values = payload_values(mock_instance.embed.call_args)
    assert any(v == ['hello world'] or v == 'hello world' for v in values)
