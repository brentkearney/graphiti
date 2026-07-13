"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google import genai
    from google.genai import types
else:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            'google-genai is required for GeminiEmbedder. '
            'Install it with: pip install graphiti-core[google-genai]'
        ) from None

from pydantic import Field

from .client import EmbedderClient, EmbedderConfig, TaskType

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'text-embedding-001'  # gemini-embedding-001 or text-embedding-005

DEFAULT_BATCH_SIZE = 100

# Task-aware embedding. gemini-embedding-2 and newer express retrieval intent
# as an input prompt prefix; gemini-embedding-001 and text-embedding-* take a
# native EmbedContentConfig.task_type instead. https://ai.google.dev/gemini-api/docs/embeddings
DOCUMENT_PREFIX = 'title: none | text: '
QUERY_PREFIX = 'task: search result | query: '
PREFIXES = {'document': DOCUMENT_PREFIX, 'query': QUERY_PREFIX}
NATIVE_TASK_TYPE = {'document': 'RETRIEVAL_DOCUMENT', 'query': 'RETRIEVAL_QUERY'}


class GeminiEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    api_key: str | None = None


class GeminiEmbedder(EmbedderClient):
    """
    Google Gemini Embedder Client
    """

    def __init__(
        self,
        config: GeminiEmbedderConfig | None = None,
        client: 'genai.Client | None' = None,
        batch_size: int | None = None,
    ):
        """
        Initialize the GeminiEmbedder with the provided configuration and client.

        Args:
            config (GeminiEmbedderConfig | None): The configuration for the GeminiEmbedder, including API key, model, base URL, temperature, and max tokens.
            client (genai.Client | None): An optional async client instance to use. If not provided, a new genai.Client is created.
            batch_size (int | None): An optional batch size to use. If not provided, the default batch size will be used.
        """
        if config is None:
            config = GeminiEmbedderConfig()

        self.config = config

        if client is None:
            self.client = genai.Client(api_key=config.api_key)
        else:
            self.client = client

        if batch_size is None and self.config.embedding_model.startswith('gemini-embedding-'):
            # Gemini embedding models (`gemini-embedding-001`, `gemini-embedding-2`, etc.)
            # are observed to return mismatched embedding counts when called in larger
            # batches — fewer embeddings than inputs, with no error. Downstream
            # `zip(filtered_nodes, name_embeddings, strict=True)` in nodes.py / edges.py
            # then raises ValueError("zip() argument 2 is shorter than argument 1"),
            # which the QueueService swallows, silently dropping the episode.
            #
            # Reference: Gemini API per-request instance limit
            # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api
            #
            # Force batch_size=1 for all Gemini embedding models until upstream
            # confirms a reliable batch endpoint. Callers can override via the
            # `batch_size` argument if they have empirical proof a larger size works
            # for their model + region.
            self.batch_size = 1
        elif batch_size is None:
            self.batch_size = DEFAULT_BATCH_SIZE
        else:
            self.batch_size = batch_size

    def _model(self) -> str:
        return self.config.embedding_model or DEFAULT_EMBEDDING_MODEL

    def _uses_prompt_prefixes(self) -> bool:
        model = self._model()
        return model.startswith('gemini-embedding-') and not model.startswith(
            'gemini-embedding-001'
        )

    def _apply_prefix(self, text: str, task_type: TaskType | None) -> str:
        if task_type is None or not self._uses_prompt_prefixes():
            return text
        return f'{PREFIXES[task_type]}{text}'

    def _prefix_input(self, input_data, task_type: TaskType | None):
        """Prefix string content, preserving the input's shape. Token iterables pass through."""
        if isinstance(input_data, str):
            return self._apply_prefix(input_data, task_type)
        if isinstance(input_data, list):
            return [
                self._apply_prefix(i, task_type) if isinstance(i, str) else i for i in input_data
            ]
        return input_data

    def _embed_config(self, task_type: TaskType | None) -> 'types.EmbedContentConfig':
        kwargs: dict = {'output_dimensionality': self.config.embedding_dim}
        if task_type is not None and not self._uses_prompt_prefixes():
            kwargs['task_type'] = NATIVE_TASK_TYPE[task_type]
        return types.EmbedContentConfig(**kwargs)

    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
        task_type: TaskType | None = None,
    ) -> list[float]:
        """
        Create embeddings for the given input data using Google's Gemini embedding model.

        Args:
            input_data: The input data to create embeddings for. Can be a string, list of strings,
                       or an iterable of integers or iterables of integers.
            task_type: Optional retrieval intent ('document' or 'query'). Applied as a prompt
                       prefix or native EmbedContentConfig.task_type depending on model family.

        Returns:
            A list of floats representing the embedding vector.
        """
        # Generate embeddings
        result = await self.client.aio.models.embed_content(
            model=self.config.embedding_model or DEFAULT_EMBEDDING_MODEL,
            contents=[self._prefix_input(input_data, task_type)],  # type: ignore[arg-type]  # mypy fails on broad union type
            config=self._embed_config(task_type),
        )

        if not result.embeddings or len(result.embeddings) == 0 or not result.embeddings[0].values:
            raise ValueError('No embeddings returned from Gemini API in create()')

        return result.embeddings[0].values

    async def create_batch(
        self, input_data_list: list[str], task_type: TaskType | None = None
    ) -> list[list[float]]:
        """
        Create embeddings for a batch of input data using Google's Gemini embedding model.

        This method handles batching to respect the Gemini API's limits on the number
        of instances that can be processed in a single request.

        Args:
            input_data_list: A list of strings to create embeddings for.
            task_type: Optional retrieval intent ('document' or 'query'). Applied as a prompt
                       prefix or native EmbedContentConfig.task_type depending on model family.

        Returns:
            A list of embedding vectors (each vector is a list of floats).
        """
        if not input_data_list:
            return []

        input_data_list = [self._apply_prefix(i, task_type) for i in input_data_list]

        batch_size = self.batch_size
        all_embeddings = []

        # Process inputs in batches
        for i in range(0, len(input_data_list), batch_size):
            batch = input_data_list[i : i + batch_size]

            try:
                # Generate embeddings for this batch
                result = await self.client.aio.models.embed_content(
                    model=self.config.embedding_model or DEFAULT_EMBEDDING_MODEL,
                    contents=batch,  # type: ignore[arg-type]  # mypy fails on broad union type
                    config=self._embed_config(task_type),
                )

                if not result.embeddings or len(result.embeddings) == 0:
                    raise Exception('No embeddings returned')

                # Process embeddings from this batch
                for embedding in result.embeddings:
                    if not embedding.values:
                        raise ValueError('Empty embedding values returned')
                    all_embeddings.append(embedding.values)

            except Exception as e:
                # If batch processing fails, fall back to individual processing
                logger.warning(
                    f'Batch embedding failed for batch {i // batch_size + 1}, falling back to individual processing: {e}'
                )

                for item in batch:
                    try:
                        # Process each item individually
                        result = await self.client.aio.models.embed_content(
                            model=self.config.embedding_model or DEFAULT_EMBEDDING_MODEL,
                            contents=[item],  # type: ignore[arg-type]  # mypy fails on broad union type
                            config=self._embed_config(task_type),
                        )

                        if not result.embeddings or len(result.embeddings) == 0:
                            raise ValueError('No embeddings returned from Gemini API')
                        if not result.embeddings[0].values:
                            raise ValueError('Empty embedding values returned')

                        all_embeddings.append(result.embeddings[0].values)

                    except Exception as individual_error:
                        logger.error(f'Failed to embed individual item: {individual_error}')
                        raise individual_error

        # Gemini's batch embed endpoint can silently return fewer embeddings than
        # inputs. A short result would propagate to zip(..., strict=True) downstream
        # and get swallowed, silently dropping data. Fail loudly instead.
        if len(all_embeddings) != len(input_data_list):
            raise ValueError(
                f'Gemini API returned {len(all_embeddings)} embeddings for '
                f'{len(input_data_list)} inputs'
            )

        return all_embeddings
