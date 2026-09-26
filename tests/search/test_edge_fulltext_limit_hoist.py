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

# FalkorDB edge-fulltext query shape.
#
# Live path (search_utils.edge_fulltext_search): upstream #1711 reads a hit's
# endpoints off the yielded relationship instead of re-matching
# `(n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)`, which FalkorDB plans as
# a full :Entity scan per hit. The fork's earlier LIMIT hoist on this path was
# superseded by #1711 and, left in place, silently overrode it.
#
# Dormant copy (FalkorSearchOperations.edge_fulltext_search): #1711 did not touch
# it, so it keeps the fork's hoist — `WITH ... ORDER BY score DESC LIMIT
# $limit_with_buffer` between the YIELD and the MATCH, with limit_with_buffer ==
# limit*4.
#
# Both are driven with mocked drivers, capturing the emitted Cypher without
# touching a database.

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


async def test_falkordb_live_path_reads_endpoints_without_per_hit_scan():
    """A. FalkorDB takes endpoints from the yielded relationship (#1711), no per-hit MATCH."""
    cypher, kwargs = await _run_search(GraphProvider.FALKORDB, limit=5)

    assert 'startNode(rel) AS n' in cypher
    assert 'endNode(rel) AS m' in cypher
    assert RELATES_TO_EXPANSION not in cypher
    assert cypher.rstrip().endswith('LIMIT $limit')
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
