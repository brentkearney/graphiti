"""The RELATION TYPE RULES block is an atVenu patch. Guard its intent."""

from graphiti_core.prompts.extract_edges import edge


def _user_message() -> str:
    context = {
        'episode_content': 'VFAServer runs Rails 6.1 in staging3.',
        'nodes': [{'name': 'VFAServer', 'entity_types': ['Service']}],
        'previous_episodes': [],
        'reference_time': '2026-08-05T00:00:00Z',
        'edge_types': [
            {
                'fact_type_name': 'DEPLOYED_TO',
                'fact_type_signatures': [],
                'fact_type_description': 'x',
            }
        ],
        'custom_extraction_instructions': '',
    }
    return edge(context)[-1].content


def test_coining_is_a_last_resort_not_a_peer_option():
    content = _user_message()
    assert 'Otherwise, derive a `relation_type`' not in content
    assert 'last resort' in content


def test_verbatim_reuse_is_demanded():
    assert 'VERBATIM' in _user_message()


def test_inverses_are_expressed_by_swapping_endpoints():
    content = _user_message()
    assert 'SWAP' in content
    assert 'Never coin an inverse' in content


def test_screaming_snake_case_is_still_the_format_for_coined_names():
    assert 'SCREAMING_SNAKE_CASE' in _user_message()


def test_fact_types_section_still_renders():
    assert 'DEPLOYED_TO' in _user_message()
