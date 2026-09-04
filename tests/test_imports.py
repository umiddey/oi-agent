"""Import smoke tests for retained runtime modules."""


def test_retained_modules_import():
    """The OpenCode runtime and Discord rails import without legacy modules."""
    import oi_agent.cli
    import oi_agent.config
    import oi_agent.daemon
    import oi_agent.opencode.bootstrap
    import oi_agent.opencode.policy
    import oi_agent.opencode.prompt
    import oi_agent.opencode.runner
    import oi_agent.poster
    import oi_agent.reflection.engine
    import oi_agent.store
    import oi_agent.watch.discord_client

    assert oi_agent.watch.discord_client is not None
    assert oi_agent.reflection.engine is not None
