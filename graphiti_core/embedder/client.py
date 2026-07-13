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

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', 1024))

# 'document' tags content being stored/indexed; 'query' tags search input at retrieval time.
TaskType = Literal['document', 'query']


class EmbedderConfig(BaseModel):
    embedding_dim: int = Field(default=EMBEDDING_DIM, frozen=True)


class EmbedderClient(ABC):
    @abstractmethod
    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
        task_type: TaskType | None = None,
    ) -> list[float]:
        """Create an embedding for a single input.

        Args:
            input_data: The content to embed.
            task_type: Retrieval intent for the embedded content: 'document' for stored
                content being indexed, 'query' for search input at retrieval time.
                Implementations MAY use this to improve asymmetric retrieval (e.g. an
                instruction prefix or a provider-native task-type parameter).
                Implementations that don't support task-aware embedding must still accept
                this argument and leave behavior unchanged. `None` preserves legacy
                symmetric behavior (no task-specific handling).

        Returns:
            The embedding vector.
        """
        pass

    async def create_batch(
        self, input_data_list: list[str], task_type: TaskType | None = None
    ) -> list[list[float]]:
        """Create embeddings for a batch of inputs.

        Args:
            input_data_list: The batch of content to embed.
            task_type: Retrieval intent for the embedded content: 'document' for stored
                content being indexed, 'query' for search input at retrieval time.
                Implementations MAY use this to improve asymmetric retrieval.
                Implementations that don't support task-aware embedding must still accept
                this argument and leave behavior unchanged. `None` preserves legacy
                symmetric behavior (no task-specific handling).

        Returns:
            One embedding vector per input, in order.
        """
        raise NotImplementedError()
