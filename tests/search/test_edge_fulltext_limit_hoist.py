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

# Regression coverage for the FalkorDB edge-fulltext LIMIT hoist.
#
# FalkorDB's db.idx.fulltext.queryRelationships procedure takes no {limit}
# option (unlike Neo4j), so without an early LIMIT the per-row
# `MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)` expansion runs
# once per fulltext hit and times out on broad-tokened queries against large
# graphs. The fix inserts `WITH ... ORDER BY score DESC LIMIT $limit_with_buffer`
# between the YIELD and the MATCH, FalkorDB-only, with limit_with_buffer == limit*4.
#
# These tests drive the live path (search_utils.edge_fulltext_search) and the
# dormant copy (FalkorSearchOperations.edge_fulltext_search) with mocked drivers,
# capturing the emitted Cypher without touching a database.

from unittest.mock import AsyncMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.operations.search_ops import FalkorSearchOperations
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import edge_fulltext_search

pytestmark = pytest.mark.asyncio

RELATES_TO_EXPANSION = 'MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]'
LIMIT_WITH_BUFFER = 'LIMIT $limit_with_buffer'


def _make_driver(provider: GraphProvider) -> AsyncMock:
    """A minimal driver double for edge_fulltext_search.

    Only the attributes the function touches on the non-Neptune path are set;
    execute_query records the Cypher and params and returns no rows.
    """
    driver = AsyncMock()
    driver.provider = provider
    driver.search_interface = None
    driver.fulltext_syntax = '@'
    driver.build_fulltext_query = lambda query, group_ids, max_len: '(merchandise)'
    driver.execute_query = AsyncMock(return_value=([], None, None))
    return driver


async def _run_search(provider: GraphProvider, limit: int = 5):
    driver = _make_driver(provider)
    await edge_fulltext_search(
        driver,
        'merchandise',
        SearchFilters(),
        group_ids=['test_group'],
        limit=limit,
    )
    call = driver.execute_query.call_args
    cypher = call.args[0]
    return cypher, call.kwargs


async def test_falkordb_hoists_limit_before_relates_to_expansion():
    """A. Regression: FalkorDB emits LIMIT $limit_with_buffer before the per-row MATCH."""
    cypher, kwargs = await _run_search(GraphProvider.FALKORDB, limit=5)

    assert LIMIT_WITH_BUFFER in cypher
    assert RELATES_TO_EXPANSION in cypher
    assert cypher.index(LIMIT_WITH_BUFFER) < cypher.index(RELATES_TO_EXPANSION), (
        'the buffered LIMIT must be hoisted ahead of the RELATES_TO expansion'
    )
    assert kwargs['limit_with_buffer'] == 5 * 4


@pytest.mark.parametrize('provider', [GraphProvider.NEO4J, GraphProvider.KUZU])
async def test_other_providers_are_untouched(provider):
    """B. Provider isolation: non-FalkorDB queries carry no buffered LIMIT."""
    cypher, kwargs = await _run_search(provider, limit=5)

    assert LIMIT_WITH_BUFFER not in cypher
    assert 'limit_with_buffer' not in kwargs


async def test_falkordb_final_order_by_limit_still_terminates():
    """C. Correctness: the final ORDER BY score DESC / LIMIT $limit still bounds the query."""
    cypher, kwargs = await _run_search(GraphProvider.FALKORDB, limit=5)

    assert cypher.rstrip().endswith('LIMIT $limit')
    # The terminating LIMIT must come after the RELATES_TO expansion (i.e. it is
    # a distinct, later clause from the hoisted buffered LIMIT).
    assert cypher.rindex('LIMIT $limit') > cypher.index(RELATES_TO_EXPANSION)
    assert kwargs['limit'] == 5


async def test_falkor_search_ops_dormant_path_also_hoists_limit():
    """D. The dormant FalkorSearchOperations copy hoists the LIMIT identically."""
    executor = AsyncMock()
    executor.execute_query = AsyncMock(return_value=([], None, None))

    ops = FalkorSearchOperations()
    await ops.edge_fulltext_search(
        executor,
        'merchandise',
        SearchFilters(),
        group_ids=['test_group'],
        limit=5,
    )

    call = executor.execute_query.call_args
    cypher = call.args[0]
    kwargs = call.kwargs

    assert LIMIT_WITH_BUFFER in cypher
    assert RELATES_TO_EXPANSION in cypher
    assert cypher.index(LIMIT_WITH_BUFFER) < cypher.index(RELATES_TO_EXPANSION)
    assert kwargs['limit_with_buffer'] == 5 * 4
