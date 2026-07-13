"""One-off migration: re-embed every stored vector with document task prefixes.

Runs INSIDE the graphiti MCP container (docker exec), so it uses the exact
embedder code and config the server uses. Rewrites EntityNode.name_embedding,
EntityEdge.fact_embedding, and CommunityNode.name_embedding for one group_id.

All-or-nothing: the AtvenuEmbeddingMeta marker is written only when every
object was re-embedded and saved. Re-running is idempotent (embeddings are
regenerated from source text, never from old vectors).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

import yaml

from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.errors import GroupsEdgesNotFoundError
from graphiti_core.nodes import CommunityNode, EntityNode

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(levelname)s %(message)s')
logger = logging.getLogger('reembed')

MARKER_VERSION = 1
DEFAULT_CONFIG = '/app/mcp_server/config/config-docker-falkordb-gemini.yaml'


def build_driver(graph: str) -> FalkorDriver:
    uri = urlparse(os.environ.get('FALKORDB_URI', 'redis://falkordb:6379'))
    return FalkorDriver(
        host=uri.hostname or 'falkordb',
        port=uri.port or 6379,
        password=os.environ.get('FALKORDB_PASSWORD') or None,
        database=graph,
    )


def build_embedder(config_path: str) -> GeminiEmbedder:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    api_key = os.environ['GEMINI_API_KEY']
    return GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=api_key,
            embedding_model=cfg['embedder']['model'],
            embedding_dim=int(cfg['embedder']['dimensions']),
        ),
        batch_size=1,
    )


# get_by_group_ids' uuid_cursor pagination relies on a `x.uuid < $cursor` predicate
# that FalkorDB v4.18.10 mis-evaluates (it returns all rows regardless of the cursor,
# so the cursor never advances and the loader spins forever). We instead load every
# object in a single unpaginated query (limit=None avoids the broken predicate) and
# verify against a COUNT that nothing was silently truncated at RESULTSET_SIZE.
COUNT_CYPHER = {
    EntityNode: 'MATCH (n:Entity) WHERE n.group_id IN $gids RETURN count(n) AS c',
    EntityEdge: 'MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) '
                'WHERE e.group_id IN $gids RETURN count(e) AS c',
    CommunityNode: 'MATCH (c:Community) WHERE c.group_id IN $gids RETURN count(c) AS c',
}


async def count_objects(cls, driver, group_id: str) -> int:
    records, _, _ = await driver.execute_query(COUNT_CYPHER[cls], gids=[group_id])
    return int(records[0]['c']) if records else 0


async def load_all(cls, driver, group_id: str, page_size: int) -> list:
    try:
        items = await cls.get_by_group_ids(driver, [group_id], limit=None)
    except GroupsEdgesNotFoundError:
        # EntityEdge.get_by_group_ids raises instead of returning [] when a group has
        # zero edges (nodes/communities return []); treat as empty so sparse graphs
        # still migrate and get marked.
        items = []
    expected = await count_objects(cls, driver, group_id)
    if len(items) != expected:
        raise RuntimeError(
            f'{cls.__name__}: loaded {len(items)} of {expected} objects '
            f'(likely RESULTSET_SIZE truncation) — aborting before any write'
        )
    return items


async def read_marker(driver) -> dict | None:
    records, _, _ = await driver.execute_query(
        'MATCH (m:AtvenuEmbeddingMeta) RETURN m.version AS version, m.run AS run, '
        'm.applied_at AS applied_at, m.stats AS stats ORDER BY m.version DESC, m.run DESC LIMIT 1'
    )
    return dict(records[0]) if records else None


async def write_marker(driver, run: int, stats: dict) -> None:
    await driver.execute_query(
        'CREATE (:AtvenuEmbeddingMeta {version: $version, applied_at: $ts, run: $run, '
        'stats: $stats})',
        version=MARKER_VERSION,
        ts=datetime.now(timezone.utc).isoformat(),
        run=run,
        stats=json.dumps(stats),
    )


async def embed_all(objs: list, regenerate, concurrency: int) -> None:
    """Regenerate embeddings for objs. Any embed exception propagates so the caller
    aborts before a single write (all-or-nothing across ALL types)."""
    sem = asyncio.Semaphore(concurrency)

    async def guarded(obj):
        async with sem:
            await regenerate(obj)

    await asyncio.gather(*(guarded(o) for o in objs))


async def save_all(objs: list, save, concurrency: int) -> list[str]:
    """Save objs, collecting per-object save failures (returned as uuids). A save
    failure is reported, never raised, so one bad write can't leave the run in an
    undefined state — the failures list gates the marker."""
    sem = asyncio.Semaphore(concurrency)
    failures: list[str] = []

    async def save_one(obj):
        try:
            async with sem:
                await save(obj)
        except Exception as e:  # noqa: BLE001 - report, don't crash the run
            logger.error(f'save failed uuid={obj.uuid}: {e}')
            failures.append(obj.uuid)

    await asyncio.gather(*(save_one(o) for o in objs))
    return failures


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--graph', required=True, help='FalkorDB graph key (database name)')
    p.add_argument('--group-id', required=True, help='group_id property value to migrate')
    p.add_argument('--apply', action='store_true')
    p.add_argument('--force', action='store_true', help='apply even when marker exists')
    p.add_argument('--page-size', type=int, default=200,
                   help='accepted for CLI-contract compatibility; unused '
                        '(objects load in a single query, see load_all)')
    p.add_argument('--concurrency', type=int, default=10)
    p.add_argument('--config', default=DEFAULT_CONFIG)
    args = p.parse_args()

    out = {'ok': False, 'mode': 'apply' if args.apply else 'status', 'graph': args.graph,
           'counts': {}, 'marker': None, 'reembedded': None, 'failures': [], 'error': None}
    driver = None
    try:
        driver = build_driver(args.graph)
        nodes = await load_all(EntityNode, driver, args.group_id, args.page_size)
        edges = await load_all(EntityEdge, driver, args.group_id, args.page_size)
        communities = await load_all(CommunityNode, driver, args.group_id, args.page_size)
        out['counts'] = {'entities': len(nodes), 'edges': len(edges),
                         'communities': len(communities)}
        out['marker'] = await read_marker(driver)

        if not args.apply:
            out['ok'] = True
            return 0

        if out['marker'] and out['marker'].get('version', 0) >= MARKER_VERSION and not args.force:
            out['error'] = 'marker present; already migrated (use --force to re-run)'
            return 0

        embedder = build_embedder(args.config)

        # Phase 1: embed EVERY type first. Any embed failure raises here, before a
        # single write, so a partial run never leaves some types saved and others not.
        await embed_all(nodes, lambda n: n.generate_name_embedding(embedder), args.concurrency)
        await embed_all(edges, lambda e: e.generate_embedding(embedder), args.concurrency)
        await embed_all(communities, lambda c: c.generate_name_embedding(embedder),
                        args.concurrency)

        # Phase 2: save every type, collecting per-object save failures. reembedded
        # reflects what actually persisted (attempted minus failed), even on error.
        node_fail = await save_all(nodes, lambda n: n.save(driver), args.concurrency)
        edge_fail = await save_all(edges, lambda e: e.save(driver), args.concurrency)
        comm_fail = await save_all(communities, lambda c: c.save(driver), args.concurrency)
        failures = node_fail + edge_fail + comm_fail

        out['failures'] = failures
        out['reembedded'] = {
            'entities': len(nodes) - len(node_fail),
            'edges': len(edges) - len(edge_fail),
            'communities': len(communities) - len(comm_fail),
        }
        if failures:
            out['error'] = f'{len(failures)} objects failed to save; marker NOT written; re-run'
            return 0
        prior_run = (out['marker'] or {}).get('run', 0) or 0
        await write_marker(driver, prior_run + 1, out['reembedded'])
        out['marker'] = await read_marker(driver)
        out['ok'] = True
        return 0
    except Exception as e:  # noqa: BLE001 - single JSON error contract
        logger.exception('reembed failed')
        out['error'] = str(e)
        return 0
    finally:
        if driver is not None:
            await driver.close()
        print(json.dumps(out))


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
