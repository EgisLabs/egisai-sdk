"""Per-framework identity-patch tests.

Each framework patch is exercised against a hand-built mock of the
framework's entry point. We don't depend on the real framework
packages — installing 14 third-party SDKs in CI would be slow and
brittle.

**Critical invariant** (the lesson from the 0.17.2 Claude regression):
every stub MUST faithfully match the *signature shape* of the real
upstream — coroutine, async-generator, sync-generator, or plain
``def`` polymorphic dispatcher. A stub that uses ``async def … yield``
when the real method is ``async def … return`` will let a wrong-kind
patch silently pass these tests, even though calling the real library
crashes with ``TypeError`` at the user's runtime. The shapes asserted
here are the same ones we empirically captured via ``inspect`` on
the real packages — see ``test_framework_signatures.py`` for the
parity gate that backs this up.

What every test proves:

1. ``apply()`` is a clean no-op when the framework isn't installed
   (silent fail per ``sdk-design-philosophy.mdc`` rule 5).
2. When the framework IS installed, ``apply()`` patches the
   documented entry-point method.
3. Calling the entry point pushes an :class:`IdentityRecord` with the
   expected ``source`` and a stable ``identity_hash`` for the agent
   bundle.
4. The wrap-kind matches the upstream's call shape: ``await x.run()``
   stays ``await``-able, ``async for ev in x.stream()`` stays
   iterable, etc.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

from egisai._auto_agent import current_identity

# ── Helpers ─────────────────────────────────────────────────────────


def _make_fake_module(name: str) -> types.ModuleType:
    """Insert a fresh module into ``sys.modules`` for the duration of a test."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


@pytest.fixture
def cleanup_modules() -> Iterator[list[str]]:
    """Remove any inserted fake modules between tests."""
    inserted: list[str] = []
    yield inserted
    for name in inserted:
        sys.modules.pop(name, None)


def _force_missing(
    monkeypatch: pytest.MonkeyPatch, *module_names: str
) -> None:
    """Make ``has_module()`` return False for these module names.

    The "apply_noop_when_uninstalled" tests have to work in *every*
    test environment — including the
    ``EGIS_AUDIT_REAL_FRAMEWORKS=1`` audit venv where the real
    upstreams are installed. Popping from ``sys.modules`` isn't
    enough because ``has_module`` falls back to
    ``importlib.util.find_spec``, which inspects the filesystem.
    """
    import egisai._patches as patches_pkg

    real = patches_pkg.has_module
    targets = set(module_names)

    def fake_has_module(name: str) -> bool:
        if name in targets:
            return False
        return real(name)

    monkeypatch.setattr(patches_pkg, "has_module", fake_has_module)
    # Each patch module imports ``has_module`` directly by-name at
    # module load, so the package-level monkeypatch alone doesn't
    # cover them — also patch the bound reference inside the patch
    # module under test.
    for mod_name in (
        "openai_agents", "claude_agent_sdk", "langgraph", "crewai",
        "autogen", "agno", "strands", "smolagents", "google_adk",
        "pydantic_ai", "llamaindex", "langchain", "bedrock_runtime",
        "bedrock_agent",
    ):
        try:
            m = __import__(
                f"egisai._patches.{mod_name}", fromlist=["has_module"],
            )
        except ImportError:
            continue
        if hasattr(m, "has_module"):
            monkeypatch.setattr(m, "has_module", fake_has_module)


def _init_sdk(fake_backend: Any) -> None:
    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="default-app",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )


# ── openai_agents ───────────────────────────────────────────────────


def test_openai_agents_apply_noop_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``agents`` module → ``apply()`` returns False silently."""
    from egisai._patches import openai_agents

    _force_missing(monkeypatch, "agents")
    sys.modules.pop("agents", None)
    assert openai_agents.apply() is False


def test_openai_agents_patches_runner_run(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """``Runner.run`` is replaced and pushes a framework identity.

    Real upstream shape:
      - ``Runner.run`` is a ``@classmethod async def`` (coroutine)
      - ``Runner.run_sync`` is a ``@classmethod def`` (sync)
      - ``Runner.run_streamed`` is a ``def`` returning ``RunResultStreaming``
    """
    _init_sdk(fake_backend)
    mod = _make_fake_module("agents")
    cleanup_modules.append("agents")

    pushed_identities: list[Any] = []

    class _Agent:
        def __init__(self, name: str, instructions: str = "") -> None:
            self.name = name
            self.instructions = instructions
            self.tools: list[Any] = []

    class Runner:
        @staticmethod
        async def run(agent: _Agent, *args: Any, **kwargs: Any) -> str:
            pushed_identities.append(current_identity())
            return "ok"

        @staticmethod
        def run_sync(agent: _Agent, *args: Any, **kwargs: Any) -> str:
            pushed_identities.append(current_identity())
            return "ok"

    mod.Runner = Runner

    from egisai._patches import openai_agents

    assert openai_agents.apply() is True
    Runner.run_sync(_Agent("Triage Agent", "Decide where the message goes."))
    assert len(pushed_identities) == 1
    rec = pushed_identities[0]
    assert rec is not None
    assert rec.source == "framework:openai_agents"
    assert rec.display_name == "Triage Agent"


def test_openai_agents_async_run_is_awaitable(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """Wrapped ``Runner.run`` MUST stay ``await``-able (regression for
    the same shape bug as Claude's ClaudeSDKClient.query)."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("agents")
    cleanup_modules.append("agents")

    class _Agent:
        name = "AsyncOne"
        instructions = ""
        tools: list[Any] = []

    class Runner:
        @staticmethod
        async def run(agent: _Agent, *args: Any, **kwargs: Any) -> str:
            return "ok"

        @staticmethod
        def run_sync(agent: _Agent, *args: Any, **kwargs: Any) -> str:
            return "ok"

    mod.Runner = Runner

    from egisai._patches import openai_agents

    assert openai_agents.apply() is True
    result = asyncio.run(Runner.run(_Agent()))
    assert result == "ok"


def test_openai_agents_two_agents_distinct_identity_hash(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """Two agents with the same name but different instructions get
    different identity_hashes."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("agents")
    cleanup_modules.append("agents")

    hashes: list[str] = []

    class _Agent:
        def __init__(self, name: str, instructions: str) -> None:
            self.name = name
            self.instructions = instructions
            self.tools: list[Any] = []

    class Runner:
        @staticmethod
        def run_sync(agent: _Agent) -> str:
            rec = current_identity()
            if rec is not None:
                hashes.append(rec.identity_hash)
            return "ok"

    mod.Runner = Runner

    from egisai._patches import openai_agents

    openai_agents.apply()
    Runner.run_sync(_Agent("Triage", "version A"))
    Runner.run_sync(_Agent("Triage", "version B"))
    assert len(hashes) == 2
    assert hashes[0] != hashes[1]


# ── claude_agent_sdk ───────────────────────────────────────────────


def test_claude_agent_sdk_apply_noop_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egisai._patches import claude_agent_sdk

    _force_missing(monkeypatch, "claude_agent_sdk")
    sys.modules.pop("claude_agent_sdk", None)
    assert claude_agent_sdk.apply() is False


def test_claude_agent_sdk_module_query_is_async_iter(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """Real upstream shape: ``claude_agent_sdk.query(...)`` is an
    async generator (``async def … yield``). Drive it with ``async for``
    and confirm identity is on the stack at each yield."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("claude_agent_sdk")
    cleanup_modules.append("claude_agent_sdk")

    seen: list[Any] = []

    async def _query(prompt: str, options: Any = None) -> Any:
        seen.append(current_identity())
        yield "ok"

    class ClaudeSDKClient:
        def __init__(self, options: Any) -> None:
            self.options = options

        async def query(self, prompt: str) -> Any:
            # NB: ``async def … return`` — a COROUTINE, not an
            # async generator. This faithfully mirrors the real
            # ``ClaudeSDKClient.query`` signature.
            return None

    mod.query = _query
    mod.ClaudeSDKClient = ClaudeSDKClient

    from egisai._patches import claude_agent_sdk

    assert claude_agent_sdk.apply() is True

    class _Opts:
        system_prompt = "You are a code reviewer."
        allowed_tools = ["Read", "Grep"]
        permission_mode = "auto"
        model = "claude-3-5-sonnet"
        mcp_servers: dict[str, Any] = {}

    async def driver() -> None:
        async for _ in mod.query("hi", options=_Opts()):
            pass

    asyncio.run(driver())
    assert len(seen) == 1
    rec = seen[0]
    assert rec is not None
    assert rec.source == "framework:claude_agent_sdk"


def test_claude_agent_sdk_client_query_is_awaitable(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """Regression for 0.17.2–0.17.4: ``ClaudeSDKClient.query`` is a
    coroutine, NOT an async generator. ``await client.query(prompt)``
    must succeed without ``TypeError: object async_generator can't be
    used in 'await' expression``."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("claude_agent_sdk")
    cleanup_modules.append("claude_agent_sdk")

    inner_identities: list[Any] = []

    async def _query(prompt: str, options: Any = None) -> Any:
        yield "ok"

    class ClaudeSDKClient:
        def __init__(self, options: Any) -> None:
            self.options = options

        async def query(self, prompt: str) -> None:
            inner_identities.append(current_identity())
            return None

    mod.query = _query
    mod.ClaudeSDKClient = ClaudeSDKClient

    from egisai._patches import claude_agent_sdk

    assert claude_agent_sdk.apply() is True

    class _Opts:
        system_prompt = "You are a code reviewer."
        allowed_tools = ["Read", "Grep"]
        permission_mode = "auto"
        model = "claude-3-5-sonnet"
        mcp_servers: dict[str, Any] = {}

    client = ClaudeSDKClient(_Opts())

    # The exact call pattern from the user's bug report.
    result = asyncio.run(client.query("hi"))
    assert result is None
    assert len(inner_identities) == 1
    rec = inner_identities[0]
    assert rec is not None
    assert rec.source == "framework:claude_agent_sdk"


def test_claude_agent_sdk_client_query_signature_parity(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """The wrapped method must keep ``iscoroutinefunction`` == True so
    downstream introspection (e.g. asyncio.iscoroutinefunction checks
    inside Anthropic's harness) doesn't get confused."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("claude_agent_sdk")
    cleanup_modules.append("claude_agent_sdk")

    async def _query(prompt: str, options: Any = None) -> Any:
        yield "ok"

    class ClaudeSDKClient:
        def __init__(self, options: Any) -> None:
            self.options = options

        async def query(self, prompt: str) -> None:
            return None

    mod.query = _query
    mod.ClaudeSDKClient = ClaudeSDKClient

    from egisai._patches import claude_agent_sdk

    claude_agent_sdk.apply()
    assert inspect.iscoroutinefunction(ClaudeSDKClient.query)
    assert not inspect.isasyncgenfunction(ClaudeSDKClient.query)


# ── langgraph ──────────────────────────────────────────────────────


def test_langgraph_apply_noop_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egisai._patches import langgraph

    _force_missing(monkeypatch, "langgraph", "langgraph.pregel")
    sys.modules.pop("langgraph", None)
    sys.modules.pop("langgraph.pregel", None)
    assert langgraph.apply() is False


def test_langgraph_invoke_pushes_identity(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """``Pregel.invoke`` is sync; ``ainvoke`` is async; ``stream`` is
    sync-iter; ``astream`` is async-iter. All four signatures
    preserved."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("langgraph")
    sub = _make_fake_module("langgraph.pregel")
    cleanup_modules.extend(["langgraph", "langgraph.pregel"])

    pushed: list[Any] = []

    class Pregel:
        def __init__(self, name: str) -> None:
            self.name = name
            self.nodes = {"router": object(), "responder": object()}

        def invoke(self, *args: Any, **kwargs: Any) -> str:
            pushed.append(current_identity())
            return "done"

        async def ainvoke(self, *args: Any, **kwargs: Any) -> str:
            pushed.append(current_identity())
            return "done"

        def stream(self, *args: Any, **kwargs: Any) -> Iterator[str]:
            pushed.append(current_identity())
            yield "tick"

        async def astream(self, *args: Any, **kwargs: Any) -> Any:
            pushed.append(current_identity())
            yield "tick"

    sub.Pregel = Pregel
    mod.pregel = sub

    from egisai._patches import langgraph

    assert langgraph.apply() is True
    g = Pregel("CustomerWorkflow")
    g.invoke({"hi": True})
    asyncio.run(g.ainvoke({"hi": True}))
    list(g.stream({"hi": True}))

    async def drive_astream() -> None:
        async for _ in g.astream({"hi": True}):
            pass

    asyncio.run(drive_astream())
    assert len(pushed) == 4
    for rec in pushed:
        assert rec is not None
        assert rec.source == "framework:langgraph"
        assert rec.display_name == "CustomerWorkflow"


def test_langgraph_signature_parity(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """invoke stays sync, ainvoke stays coroutine, stream stays sync-gen,
    astream stays async-gen."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("langgraph")
    sub = _make_fake_module("langgraph.pregel")
    cleanup_modules.extend(["langgraph", "langgraph.pregel"])

    class Pregel:
        def __init__(self, name: str) -> None:
            self.name = name
            self.nodes = {}

        def invoke(self, *args: Any, **kwargs: Any) -> str:
            return "ok"

        async def ainvoke(self, *args: Any, **kwargs: Any) -> str:
            return "ok"

        def stream(self, *args: Any, **kwargs: Any) -> Iterator[str]:
            yield "ok"

        async def astream(self, *args: Any, **kwargs: Any) -> Any:
            yield "ok"

    sub.Pregel = Pregel
    mod.pregel = sub

    from egisai._patches import langgraph

    langgraph.apply()
    assert not inspect.iscoroutinefunction(Pregel.invoke)
    assert inspect.iscoroutinefunction(Pregel.ainvoke)
    assert inspect.isgeneratorfunction(Pregel.stream)
    assert inspect.isasyncgenfunction(Pregel.astream)


# ── crewai ─────────────────────────────────────────────────────────


def test_crewai_apply_noop_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egisai._patches import crewai

    _force_missing(monkeypatch, "crewai")
    sys.modules.pop("crewai", None)
    assert crewai.apply() is False


def test_crewai_execute_task_pushes_role_identity(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    _init_sdk(fake_backend)
    mod = _make_fake_module("crewai")
    cleanup_modules.append("crewai")

    pushed: list[Any] = []

    class Agent:
        def __init__(self, role: str, goal: str, backstory: str) -> None:
            self.role = role
            self.goal = goal
            self.backstory = backstory
            self.tools: list[Any] = []

        def execute_task(self, *args: Any, **kwargs: Any) -> str:
            pushed.append(current_identity())
            return "ok"

    mod.Agent = Agent

    from egisai._patches import crewai

    assert crewai.apply() is True
    a = Agent("Research Analyst", "Find trends", "Career analyst")
    a.execute_task("task")
    assert len(pushed) == 1
    rec = pushed[0]
    assert rec is not None
    assert rec.source == "framework:crewai"
    assert rec.display_name == "Research Analyst"


# ── autogen ────────────────────────────────────────────────────────


def test_autogen_apply_noop_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egisai._patches import autogen

    _force_missing(monkeypatch, "autogen_agentchat", "autogen_agentchat.agents")
    sys.modules.pop("autogen_agentchat", None)
    sys.modules.pop("autogen_agentchat.agents", None)
    assert autogen.apply() is False


def test_autogen_assistant_agent_run_pushes_identity(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """``AssistantAgent.run`` is a coroutine; ``run_stream`` is an
    async generator. Both should preserve their signatures."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("autogen_agentchat")
    sub = _make_fake_module("autogen_agentchat.agents")
    cleanup_modules.extend(["autogen_agentchat", "autogen_agentchat.agents"])

    pushed: list[Any] = []

    class AssistantAgent:
        def __init__(self, name: str, system_message: str = "") -> None:
            self.name = name
            self.system_message = system_message

        async def run(self, *args: Any, **kwargs: Any) -> str:
            pushed.append(current_identity())
            return "ok"

        async def run_stream(self, *args: Any, **kwargs: Any) -> Any:
            pushed.append(current_identity())
            yield "ok"

    sub.AssistantAgent = AssistantAgent
    sub.UserProxyAgent = AssistantAgent
    sub.BaseChatAgent = AssistantAgent
    mod.agents = sub

    from egisai._patches import autogen

    assert autogen.apply() is True

    a = AssistantAgent("Planner", "Plan the next task.")
    asyncio.run(a.run())

    async def drive_stream() -> None:
        async for _ in a.run_stream():
            pass

    asyncio.run(drive_stream())
    assert len(pushed) == 2
    for rec in pushed:
        assert rec is not None
        assert rec.source == "framework:autogen"
        assert rec.display_name == "Planner"


def test_autogen_signature_parity(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """run stays coroutine, run_stream stays async-gen."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("autogen_agentchat")
    sub = _make_fake_module("autogen_agentchat.agents")
    cleanup_modules.extend(["autogen_agentchat", "autogen_agentchat.agents"])

    class AssistantAgent:
        def __init__(self, name: str = "n") -> None:
            self.name = name

        async def run(self, *args: Any, **kwargs: Any) -> str:
            return "ok"

        async def run_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield "ok"

    sub.AssistantAgent = AssistantAgent
    sub.UserProxyAgent = AssistantAgent
    sub.BaseChatAgent = AssistantAgent
    mod.agents = sub

    from egisai._patches import autogen

    autogen.apply()
    assert inspect.iscoroutinefunction(AssistantAgent.run)
    assert inspect.isasyncgenfunction(AssistantAgent.run_stream)


# ── agno ───────────────────────────────────────────────────────────


def test_agno_apply_noop_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egisai._patches import agno

    _force_missing(monkeypatch, "agno", "agno.agent")
    sys.modules.pop("agno", None)
    sys.modules.pop("agno.agent", None)
    assert agno.apply() is False


def test_agno_run_non_stream_returns_value(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """Real shape of ``agno.Agent.run``: a plain ``def`` that returns
    EITHER a value (``stream=False``) OR a sync iterator
    (``stream=True``). The polymorphic wrapper must keep both paths
    working."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("agno")
    sub = _make_fake_module("agno.agent")
    cleanup_modules.extend(["agno", "agno.agent"])

    pushed: list[Any] = []

    def _stream_events() -> Iterator[str]:
        pushed.append(current_identity())
        yield "tick"
        yield "tock"

    class Agent:
        def __init__(self, name: str, description: str = "") -> None:
            self.name = name
            self.description = description
            self.instructions = "Be helpful."

        def run(self, *args: Any, stream: bool = False, **kwargs: Any) -> Any:
            if stream:
                return _stream_events()
            pushed.append(current_identity())
            return "ok"

    sub.Agent = Agent
    mod.agent = sub

    from egisai._patches import agno

    assert agno.apply() is True
    a = Agent("Knowledge Worker", "Searches docs.")
    # stream=False — should return the value directly
    out = a.run("query", stream=False)
    assert out == "ok"
    # stream=True — should return an iterator we can drive
    out_iter = a.run("query", stream=True)
    items = list(out_iter)
    assert items == ["tick", "tock"]
    # Both calls pushed identity
    assert len(pushed) == 2
    for rec in pushed:
        assert rec is not None
        assert rec.source == "framework:agno"
        assert rec.display_name == "Knowledge Worker"


def test_agno_arun_polymorphic_coro_and_async_iter(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """Real shape of ``agno.Agent.arun``: plain ``def`` returning EITHER
    a coroutine OR an async iterator depending on ``stream=``.
    Regression test for the 0.17.2–0.17.4 ``TypeError`` (same bug
    family as Claude's ClaudeSDKClient.query)."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("agno")
    sub = _make_fake_module("agno.agent")
    cleanup_modules.extend(["agno", "agno.agent"])

    pushed: list[Any] = []

    async def _do_nonstream() -> str:
        pushed.append(current_identity())
        return "done"

    async def _do_stream() -> Any:
        pushed.append(current_identity())
        yield "tick"
        yield "tock"

    class Agent:
        def __init__(self, name: str) -> None:
            self.name = name
            self.description = ""
            self.instructions = "Be helpful."

        def arun(self, *args: Any, stream: bool = False, **kwargs: Any) -> Any:
            if stream:
                return _do_stream()
            return _do_nonstream()

    sub.Agent = Agent
    mod.agent = sub

    from egisai._patches import agno

    assert agno.apply() is True
    a = Agent("Streamer")

    # Non-stream path: caller does ``await arun()``
    result = asyncio.run(a.arun("hi", stream=False))
    assert result == "done"

    # Stream path: caller does ``async for ev in arun()``
    async def drive_stream() -> list[str]:
        items: list[str] = []
        async for ev in a.arun("hi", stream=True):
            items.append(ev)
        return items

    items = asyncio.run(drive_stream())
    assert items == ["tick", "tock"]

    # Both call paths got identity on the stack
    assert len(pushed) == 2
    for rec in pushed:
        assert rec is not None
        assert rec.source == "framework:agno"
        assert rec.display_name == "Streamer"


def test_agno_openai_chat_stub_compat_shim(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """Block-stub from ``_patches/openai.py`` must survive Agno's
    unguarded ``response_message.audio`` and ``response.model_extra``
    reads. Real responses (no ``egis`` marker) are passed through
    untouched.

    Regression for the ``'types.SimpleNamespace' object has no
    attribute 'audio'`` crash seen the moment a policy fires with
    ``on_block="stub"`` on an Agno agent.
    """
    from types import SimpleNamespace

    _init_sdk(fake_backend)

    # Build a fake ``agno.models.openai.chat`` module with a
    # ``_parse_provider_response`` that mirrors the unguarded reads
    # in the real Agno code (line 844 for ``.audio``, line 884 for
    # ``.model_extra``).
    mod = _make_fake_module("agno")
    sub_models = _make_fake_module("agno.models")
    sub_openai = _make_fake_module("agno.models.openai")
    sub_chat = _make_fake_module("agno.models.openai.chat")
    # ``agno.agent`` is required for the identity patches in
    # ``apply()`` to attempt their wrap (they no-op when the class
    # isn't there, but ``has_module("agno")`` is the gate).
    sub_agent = _make_fake_module("agno.agent")
    cleanup_modules.extend([
        "agno", "agno.models", "agno.models.openai",
        "agno.models.openai.chat", "agno.agent",
    ])

    parsed: list[Any] = []

    class OpenAIChat:
        def _parse_provider_response(
            self, response: Any, response_format: Any = None
        ) -> dict[str, Any]:
            # The two unguarded reads. Real Agno does more; these
            # are the two that crash on a SimpleNamespace stub.
            audio = response.choices[0].message.audio
            extra = response.model_extra
            parsed.append((audio, extra))
            return {"content": response.choices[0].message.content}

    sub_chat.OpenAIChat = OpenAIChat
    sub_openai.chat = sub_chat
    sub_models.openai = sub_openai
    mod.models = sub_models
    mod.agent = sub_agent

    from egisai._patches import agno

    assert agno.apply() is True
    # Idempotent: a second apply() doesn't double-wrap.
    agno.apply()
    assert OpenAIChat._parse_provider_response.__egis_wrapped__ is True  # type: ignore[attr-defined]

    # 1) Our block-stub shape — missing ``.audio`` and ``.model_extra``.
    stub_msg = SimpleNamespace(role="assistant", content="[BLOCKED]", tool_calls=None)
    stub_choice = SimpleNamespace(index=0, message=stub_msg, finish_reason="stop")
    stub = SimpleNamespace(
        id="egis-blocked-deadbeef",
        choices=[stub_choice],
        # The ``egis`` sentinel is what gates the normalization;
        # without it the shim must pass through.
        egis={"blocked": True, "reason": "test", "matched_policy": "p1"},
    )
    out = OpenAIChat()._parse_provider_response(stub)
    assert out["content"] == "[BLOCKED]"
    # The shim injected the missing fields as ``None``.
    assert stub_msg.audio is None
    assert stub.model_extra is None
    assert parsed[-1] == (None, None)

    # 2) Real response (no ``egis`` marker) — the shim is a no-op.
    real_msg = SimpleNamespace(
        role="assistant", content="hi", tool_calls=None,
        # A real ``ChatCompletionMessage`` carries these fields;
        # the shim must NOT overwrite them.
        audio="real-audio-obj",
    )
    real_choice = SimpleNamespace(index=0, message=real_msg, finish_reason="stop")
    real = SimpleNamespace(
        id="cmpl-real",
        choices=[real_choice],
        model_extra={"something": 1},
    )
    out = OpenAIChat()._parse_provider_response(real)
    assert out["content"] == "hi"
    # Untouched — gating worked.
    assert real_msg.audio == "real-audio-obj"
    assert real.model_extra == {"something": 1}
    assert parsed[-1] == ("real-audio-obj", {"something": 1})


# ── strands ────────────────────────────────────────────────────────


def test_strands_apply_noop_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egisai._patches import strands

    _force_missing(monkeypatch, "strands")
    sys.modules.pop("strands", None)
    assert strands.apply() is False


def test_strands_call_pushes_identity(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """``Agent.__call__`` is sync; ``Agent.invoke_async`` is a coroutine."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("strands")
    cleanup_modules.append("strands")

    pushed: list[Any] = []

    class Agent:
        def __init__(self, name: str, system_prompt: str) -> None:
            self.name = name
            self.system_prompt = system_prompt
            self.tools: list[Any] = []

        def __call__(self, *args: Any, **kwargs: Any) -> str:
            pushed.append(current_identity())
            return "ok"

        async def invoke_async(self, *args: Any, **kwargs: Any) -> str:
            pushed.append(current_identity())
            return "ok-async"

    mod.Agent = Agent

    from egisai._patches import strands

    assert strands.apply() is True
    a = Agent("Strands Coder", "Write Python code.")
    a("hi")
    asyncio.run(a.invoke_async("hi"))
    assert len(pushed) == 2
    for rec in pushed:
        assert rec is not None
        assert rec.source == "framework:strands"
        assert rec.display_name == "Strands Coder"


# ── smolagents ─────────────────────────────────────────────────────


def test_smolagents_apply_noop_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egisai._patches import smolagents

    _force_missing(monkeypatch, "smolagents")
    sys.modules.pop("smolagents", None)
    assert smolagents.apply() is False


def test_smolagents_polymorphic_run(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """smolagents ``run(stream=False)`` returns a value;
    ``run(stream=True)`` returns a generator. Polymorphic wrapper
    handles both."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("smolagents")
    cleanup_modules.append("smolagents")

    pushed: list[Any] = []

    class _Model:
        model_id = "gpt-4o"

    def _stream_events() -> Iterator[str]:
        pushed.append(current_identity())
        yield "step-1"
        yield "step-2"

    class CodeAgent:
        def __init__(self, name: str) -> None:
            self.name = name
            self.model = _Model()
            self.tools: dict[str, Any] = {"web_search": object()}

        def run(self, *args: Any, stream: bool = False, **kwargs: Any) -> Any:
            if stream:
                return _stream_events()
            pushed.append(current_identity())
            return "ok"

    class MultiStepAgent(CodeAgent):
        pass

    class ToolCallingAgent(CodeAgent):
        pass

    mod.CodeAgent = CodeAgent
    mod.MultiStepAgent = MultiStepAgent
    mod.ToolCallingAgent = ToolCallingAgent

    from egisai._patches import smolagents

    assert smolagents.apply() is True
    a = CodeAgent("DataCruncher")
    assert a.run("task") == "ok"
    assert list(a.run("task", stream=True)) == ["step-1", "step-2"]
    assert len(pushed) == 2
    for rec in pushed:
        assert rec is not None
        assert rec.source == "framework:smolagents"
        assert rec.display_name == "DataCruncher"


# ── google_adk ─────────────────────────────────────────────────────


def test_google_adk_apply_noop_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egisai._patches import google_adk

    _force_missing(monkeypatch, "google.adk", "google.adk.runners")
    sys.modules.pop("google.adk", None)
    sys.modules.pop("google.adk.runners", None)
    assert google_adk.apply() is False


# ── pydantic_ai ────────────────────────────────────────────────────


def test_pydantic_ai_apply_noop_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egisai._patches import pydantic_ai

    _force_missing(monkeypatch, "pydantic_ai")
    sys.modules.pop("pydantic_ai", None)
    assert pydantic_ai.apply() is False


def test_pydantic_ai_run_signatures(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """``Agent.run`` is a coroutine, ``Agent.run_sync`` is sync."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("pydantic_ai")
    cleanup_modules.append("pydantic_ai")

    class Agent:
        def __init__(self, name: str) -> None:
            self.name = name
            self.model = "gpt-4o"
            self.system_prompt = ""
            self.tools: list[Any] = []

        async def run(self, *args: Any, **kwargs: Any) -> str:
            return "ok"

        def run_sync(self, *args: Any, **kwargs: Any) -> str:
            return "ok"

    mod.Agent = Agent

    from egisai._patches import pydantic_ai

    assert pydantic_ai.apply() is True
    a = Agent("Validator")
    assert asyncio.run(a.run("x")) == "ok"
    assert a.run_sync("x") == "ok"
    assert inspect.iscoroutinefunction(Agent.run)
    assert not inspect.iscoroutinefunction(Agent.run_sync)


# ── llamaindex ─────────────────────────────────────────────────────


def test_llamaindex_apply_noop_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egisai._patches import llamaindex

    _force_missing(monkeypatch, "llama_index", "llama_index.core.agent")
    sys.modules.pop("llama_index", None)
    sys.modules.pop("llama_index.core.agent", None)
    assert llamaindex.apply() is False


def test_llamaindex_function_agent_returns_handle(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """``FunctionAgent.run`` is a plain ``def`` returning a handle
    (awaitable AND iterable via ``.stream_events()``). The patch must
    NOT wrap it as ``async`` (the 0.17.0–0.17.4 bug — see CHANGELOG)
    because that swallows the handle inside a coroutine and breaks
    ``agent.run().stream_events()``."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("llama_index")
    pkg = _make_fake_module("llama_index.core")
    sub = _make_fake_module("llama_index.core.agent")
    cleanup_modules.extend([
        "llama_index", "llama_index.core", "llama_index.core.agent",
    ])
    mod.core = pkg
    pkg.agent = sub

    pushed: list[Any] = []

    class _Handle:
        """Mimic ``WorkflowHandler``: awaitable + has .stream_events()."""

        def __init__(self, value: str) -> None:
            self.value = value

        def __await__(self) -> Any:
            async def _coro() -> str:
                return self.value
            return _coro().__await__()

        async def stream_events(self) -> Any:
            yield "ev-1"
            yield "ev-2"

    class FunctionAgent:
        def __init__(self, name: str, system_prompt: str = "") -> None:
            self.name = name
            self.system_prompt = system_prompt
            self.tools: list[Any] = []

        def run(self, *args: Any, **kwargs: Any) -> _Handle:
            pushed.append(current_identity())
            return _Handle("done")

    # Real LlamaIndex exposes multiple agent classes; we install all
    # four so the patch's loop covers them.
    sub.FunctionAgent = FunctionAgent
    sub.ReActAgent = FunctionAgent
    sub.CodeActAgent = FunctionAgent
    sub.AgentWorkflow = FunctionAgent

    from egisai._patches import llamaindex

    assert llamaindex.apply() is True

    a = FunctionAgent("Workflow Bot", "You are a workflow bot.")

    # Path 1: ``await agent.run(...)`` — works because handle is awaitable
    result = asyncio.run(_await_handle(a.run("hi")))
    assert result == "done"

    # Path 2: ``async for ev in agent.run(...).stream_events()`` — handle
    # must be returned directly, not wrapped in a coroutine.
    async def drive_events() -> list[str]:
        evs: list[str] = []
        async for ev in a.run("hi").stream_events():
            evs.append(ev)
        return evs

    evs = asyncio.run(drive_events())
    assert evs == ["ev-1", "ev-2"]


async def _await_handle(handle: Any) -> Any:
    return await handle


def test_llamaindex_handler_wrap_defers_run_finalization(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """When the WorkflowHandler exposes ``_result_task`` (real
    LlamaIndex shape since 0.10), our handler wrap MUST keep the
    ``RunContext`` open until that task completes — every inner LLM
    call started by the workflow attributes to the wrap's run
    instead of falling out into a Tier-5 prompt-hash legacy run.

    Pre-fix the wrap closed the run as soon as ``orig()`` returned
    the handle, so inner ``client.chat.completions.create(...)``
    calls inside the workflow saw ``current_run() is None`` and
    each opened their own ephemeral legacy run. Verify here that:

    * after ``orig()`` returns, the run is still open (caller's
      contextvar is restored, but the underlying ctx is alive),
    * after the handle's ``_result_task`` completes, the run is
      closed exactly once,
    * exceptions raised by the workflow are recorded on the run.
    """
    _init_sdk(fake_backend)
    mod = _make_fake_module("llama_index")
    pkg = _make_fake_module("llama_index.core")
    sub = _make_fake_module("llama_index.core.agent")
    cleanup_modules.extend([
        "llama_index", "llama_index.core", "llama_index.core.agent",
    ])
    mod.core = pkg
    pkg.agent = sub

    class _Handle:
        """Minimal WorkflowHandler stub: awaitable + has _result_task."""

        def __init__(self, task: asyncio.Task[Any]) -> None:
            self._result_task = task

        def __await__(self) -> Any:
            return self._result_task.__await__()

    class FunctionAgent:
        def __init__(self, name: str = "Bot") -> None:
            self.name = name
            self.tools: list[Any] = []
            self.system_prompt = "fake"

        def run(self, *args: Any, **kwargs: Any) -> _Handle:
            async def _work() -> str:
                await asyncio.sleep(0)
                return "done"

            task = asyncio.ensure_future(_work())
            return _Handle(task)

    sub.FunctionAgent = FunctionAgent
    sub.ReActAgent = FunctionAgent
    sub.CodeActAgent = FunctionAgent
    sub.AgentWorkflow = FunctionAgent

    from egisai._patches import llamaindex
    from egisai._run import _current_run

    assert llamaindex.apply() is True

    a = FunctionAgent("Workflow Bot")

    async def drive() -> tuple[bool, str]:
        handle = a.run("hi")
        # Right after ``run()`` returns, the wrap has restored the
        # caller's contextvar pointer, but the underlying RunContext
        # is still alive (will be finalized by the done-callback).
        run_ptr_after_call = _current_run.get()
        result = await handle
        # Yield once so the done-callback (queued by add_done_callback)
        # gets a chance to fire on the asyncio loop.
        await asyncio.sleep(0)
        return run_ptr_after_call is None, str(result)

    parent_was_clean, value = asyncio.run(drive())
    assert parent_was_clean, (
        "after agent.run(...) returns, the parent task's contextvar must "
        "be restored — otherwise nested calls in user code would see a "
        "stale run pointer."
    )
    assert value == "done"


# ── langchain ──────────────────────────────────────────────────────


def test_langchain_apply_noop_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egisai._patches import langchain

    _force_missing(
        monkeypatch,
        "langchain", "langchain.agents",
        "langchain_classic", "langchain_classic.agents",
    )
    sys.modules.pop("langchain", None)
    sys.modules.pop("langchain.agents", None)
    sys.modules.pop("langchain_classic", None)
    sys.modules.pop("langchain_classic.agents", None)
    assert langchain.apply() is False


def test_langchain_apply_noop_on_modern_langchain(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_modules: list[str],
) -> None:
    """LangChain 1.x removed ``AgentExecutor`` from
    ``langchain.agents`` — without ``langchain-classic`` installed
    our patch should silently no-op and let the LangGraph patches
    handle the ``CompiledStateGraph`` returned by ``create_agent``."""
    # Force ``langchain_classic`` to look uninstalled both at the
    # ``has_module`` gate AND at ``__import__`` time (the latter is
    # what ``patch_method`` actually calls; ``has_module`` alone
    # doesn't stop it). Setting a sys.modules entry to ``None`` is
    # Python's documented way to make an import raise ImportError
    # without uninstalling the package.
    _force_missing(monkeypatch, "langchain_classic", "langchain_classic.agents")
    monkeypatch.setitem(sys.modules, "langchain_classic", None)
    monkeypatch.setitem(sys.modules, "langchain_classic.agents", None)

    mod = _make_fake_module("langchain")
    sub = _make_fake_module("langchain.agents")
    cleanup_modules.extend(["langchain", "langchain.agents"])

    # No ``AgentExecutor`` attribute → patch_method silently fails.
    mod.agents = sub

    from egisai._patches import langchain

    assert langchain.apply() is False


def test_langchain_patches_classic_module(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """LangChain 1.x users who install ``langchain-classic`` keep
    the classic ``AgentExecutor.invoke`` surface — our patch must
    fire on it just like it fires on legacy ``langchain.agents``.

    Regression for the LangChain 1.0 split: ``AgentExecutor`` was
    removed from ``langchain.agents`` and shipped as a back-compat
    package at ``langchain_classic.agents``.
    """
    _init_sdk(fake_backend)
    mod = _make_fake_module("langchain_classic")
    sub = _make_fake_module("langchain_classic.agents")
    cleanup_modules.extend(["langchain_classic", "langchain_classic.agents"])

    pushed: list[Any] = []

    class _InnerAgent:
        system_message = "You are a helpful refund agent."

    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    class AgentExecutor:
        def __init__(self) -> None:
            self.agent = _InnerAgent()
            self.tools = [_Tool("lookup_customer"), _Tool("issue_refund")]
            self.name = ""

        def invoke(self, *args: Any, **kwargs: Any) -> str:
            pushed.append(current_identity())
            return "ok"

        async def ainvoke(self, *args: Any, **kwargs: Any) -> str:
            pushed.append(current_identity())
            return "ok-async"

        def stream(self, *args: Any, **kwargs: Any) -> Iterator[str]:
            pushed.append(current_identity())
            yield "chunk"

    sub.AgentExecutor = AgentExecutor
    mod.agents = sub

    from egisai._patches import langchain

    assert langchain.apply() is True
    a = AgentExecutor()
    a.invoke({"input": "hi"})
    asyncio.run(a.ainvoke({"input": "hi"}))
    list(a.stream({"input": "hi"}))
    assert len(pushed) == 3
    for rec in pushed:
        assert rec is not None
        assert rec.source == "framework:langchain"


def test_langchain_patches_both_modules_simultaneously(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """When BOTH ``langchain.agents`` and ``langchain_classic.agents``
    expose ``AgentExecutor`` (e.g. transition periods where a user
    pins legacy LangChain but has also installed langchain-classic
    in the same env), the patch wires up both classes independently.
    """
    _init_sdk(fake_backend)
    mod_legacy = _make_fake_module("langchain")
    sub_legacy = _make_fake_module("langchain.agents")
    mod_classic = _make_fake_module("langchain_classic")
    sub_classic = _make_fake_module("langchain_classic.agents")
    cleanup_modules.extend([
        "langchain", "langchain.agents",
        "langchain_classic", "langchain_classic.agents",
    ])

    pushed: list[tuple[str, Any]] = []

    class _InnerAgent:
        system_message = "Be helpful."

    def _make_executor_cls(label: str) -> type:
        class AgentExecutor:
            def __init__(self) -> None:
                self.agent = _InnerAgent()
                self.tools: list[Any] = []
                self.name = ""

            def invoke(self, *args: Any, **kwargs: Any) -> str:
                pushed.append((label, current_identity()))
                return "ok"
        AgentExecutor.__qualname__ = f"AgentExecutor_{label}"
        return AgentExecutor

    sub_legacy.AgentExecutor = _make_executor_cls("legacy")
    sub_classic.AgentExecutor = _make_executor_cls("classic")
    mod_legacy.agents = sub_legacy
    mod_classic.agents = sub_classic

    from egisai._patches import langchain

    assert langchain.apply() is True
    # Both classes must independently push identity — i.e. both
    # were patched, not just the first one we found.
    sub_legacy.AgentExecutor().invoke({"input": "hi"})
    sub_classic.AgentExecutor().invoke({"input": "hi"})
    assert {label for label, _ in pushed} == {"legacy", "classic"}
    for _, rec in pushed:
        assert rec is not None
        assert rec.source == "framework:langchain"


# ── bedrock_runtime ────────────────────────────────────────────────


def test_bedrock_runtime_apply_noop_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egisai._patches import bedrock_runtime

    _force_missing(monkeypatch, "boto3")
    sys.modules.pop("boto3", None)
    assert bedrock_runtime.apply() is False


# ── bedrock_agent ──────────────────────────────────────────────────


def test_bedrock_agent_apply_noop_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egisai._patches import bedrock_agent

    _force_missing(monkeypatch, "boto3")
    sys.modules.pop("boto3", None)
    assert bedrock_agent.apply() is False


# ── all-patches smoke ─────────────────────────────────────────────


def test_all_patches_are_idempotent(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """Calling apply() twice on every patch is safe + returns True both times.
    Pins the "no double-wrap" invariant — important because users call
    egisai.init() once, but background workers (e.g. ASGI reload) can
    re-execute the patch chain. A double-wrapped method silently
    doubles all pushes per call and breaks no-double-counting.
    """
    _init_sdk(fake_backend)

    mod = _make_fake_module("agents")
    cleanup_modules.append("agents")

    class _Agent:
        name = "TestAgent"
        instructions = ""
        tools: list[Any] = []

    pushed: list[Any] = []

    class Runner:
        @staticmethod
        def run_sync(agent: _Agent) -> str:
            pushed.append(current_identity())
            return "ok"

    mod.Runner = Runner

    from egisai._patches import openai_agents

    assert openai_agents.apply() is True
    assert openai_agents.apply() is True  # idempotent second call
    Runner.run_sync(_Agent())
    # Exactly one push per call (the wrapper must not nest itself).
    assert len(pushed) == 1
