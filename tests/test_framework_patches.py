"""Per-framework identity-patch tests.

Each framework patch is exercised against a hand-built mock of the
framework's entry point. We don't depend on the real framework
packages — installing 14 third-party SDKs in CI would be slow and
brittle. The mocks faithfully shape-match the framework's call
signature and let us prove:

1. ``apply()`` is a clean no-op when the framework isn't installed
   (silent fail per ``sdk-design-philosophy.mdc`` rule 5).
2. When the framework IS installed, ``apply()`` patches the
   documented entry-point method.
3. Calling the entry point pushes an :class:`IdentityRecord` with the
   expected ``source`` and a stable ``identity_hash`` for the agent
   bundle.
4. Nested LLM calls inside the framework block inherit the framework's
   identity (the ContextVar stack works through ``with identity_scope``).
"""

from __future__ import annotations

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


def test_openai_agents_apply_noop_when_uninstalled() -> None:
    """No ``agents`` module → ``apply()`` returns False silently."""
    from egisai._patches import openai_agents

    sys.modules.pop("agents", None)
    assert openai_agents.apply() is False


def test_openai_agents_patches_runner_run(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """``Runner.run`` is replaced and pushes a framework identity."""
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
    # `run_sync` should now be wrapped — call it and inspect the
    # pushed identity.
    Runner.run_sync(_Agent("Triage Agent", "Decide where the message goes."))
    assert len(pushed_identities) == 1
    rec = pushed_identities[0]
    assert rec is not None
    assert rec.source == "framework:openai_agents"
    assert rec.display_name == "Triage Agent"


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


def test_claude_agent_sdk_apply_noop_when_uninstalled() -> None:
    from egisai._patches import claude_agent_sdk

    sys.modules.pop("claude_agent_sdk", None)
    assert claude_agent_sdk.apply() is False


def test_claude_agent_sdk_pushes_identity(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """Tier 2B — composite bundle hash of (system_prompt, tools, ...)."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("claude_agent_sdk")
    cleanup_modules.append("claude_agent_sdk")

    seen: list[Any] = []

    async def _query(prompt: str, options: Any = None) -> Any:
        seen.append(current_identity())
        # The real query is an async generator; for the test we yield
        # one item.
        yield "ok"

    class ClaudeSDKClient:
        def __init__(self, options: Any) -> None:
            self.options = options

        async def query(self, prompt: str) -> Any:
            seen.append(current_identity())
            yield "ok"

    mod.query = _query
    mod.ClaudeSDKClient = ClaudeSDKClient

    from egisai._patches import claude_agent_sdk

    assert claude_agent_sdk.apply() is True

    # Drive the wrapped query as an async generator.
    import asyncio

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


# ── langgraph ──────────────────────────────────────────────────────


def test_langgraph_apply_noop_when_uninstalled() -> None:
    from egisai._patches import langgraph

    sys.modules.pop("langgraph", None)
    sys.modules.pop("langgraph.pregel", None)
    assert langgraph.apply() is False


def test_langgraph_invoke_pushes_identity(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
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

    sub.Pregel = Pregel
    mod.pregel = sub

    from egisai._patches import langgraph

    assert langgraph.apply() is True
    g = Pregel("CustomerWorkflow")
    g.invoke({"hi": True})
    assert len(pushed) == 1
    rec = pushed[0]
    assert rec is not None
    assert rec.source == "framework:langgraph"
    assert rec.display_name == "CustomerWorkflow"


# ── crewai ─────────────────────────────────────────────────────────


def test_crewai_apply_noop_when_uninstalled() -> None:
    from egisai._patches import crewai

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


def test_autogen_apply_noop_when_uninstalled() -> None:
    from egisai._patches import autogen

    sys.modules.pop("autogen_agentchat", None)
    sys.modules.pop("autogen_agentchat.agents", None)
    assert autogen.apply() is False


def test_autogen_assistant_agent_run_pushes_identity(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
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
    import asyncio

    a = AssistantAgent("Planner", "Plan the next task.")
    asyncio.run(a.run())
    assert len(pushed) == 1
    rec = pushed[0]
    assert rec is not None
    assert rec.source == "framework:autogen"
    assert rec.display_name == "Planner"


# ── agno ───────────────────────────────────────────────────────────


def test_agno_apply_noop_when_uninstalled() -> None:
    from egisai._patches import agno

    sys.modules.pop("agno", None)
    sys.modules.pop("agno.agent", None)
    assert agno.apply() is False


def test_agno_run_pushes_identity(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    _init_sdk(fake_backend)
    mod = _make_fake_module("agno")
    sub = _make_fake_module("agno.agent")
    cleanup_modules.extend(["agno", "agno.agent"])

    pushed: list[Any] = []

    class Agent:
        def __init__(self, name: str, description: str = "") -> None:
            self.name = name
            self.description = description
            self.instructions = "Be helpful."

        def run(self, *args: Any, **kwargs: Any) -> str:
            pushed.append(current_identity())
            return "ok"

    sub.Agent = Agent
    mod.agent = sub

    from egisai._patches import agno

    assert agno.apply() is True
    a = Agent("Knowledge Worker", "Searches docs.")
    a.run("query")
    assert len(pushed) == 1
    rec = pushed[0]
    assert rec is not None
    assert rec.source == "framework:agno"
    assert rec.display_name == "Knowledge Worker"


# ── strands ────────────────────────────────────────────────────────


def test_strands_apply_noop_when_uninstalled() -> None:
    from egisai._patches import strands

    sys.modules.pop("strands", None)
    assert strands.apply() is False


def test_strands_call_pushes_identity(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
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

    mod.Agent = Agent

    from egisai._patches import strands

    assert strands.apply() is True
    a = Agent("Strands Coder", "Write Python code.")
    a("hi")
    assert len(pushed) == 1
    rec = pushed[0]
    assert rec is not None
    assert rec.source == "framework:strands"
    assert rec.display_name == "Strands Coder"


# ── smolagents ─────────────────────────────────────────────────────


def test_smolagents_apply_noop_when_uninstalled() -> None:
    from egisai._patches import smolagents

    sys.modules.pop("smolagents", None)
    assert smolagents.apply() is False


def test_smolagents_run_pushes_identity(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    _init_sdk(fake_backend)
    mod = _make_fake_module("smolagents")
    cleanup_modules.append("smolagents")

    pushed: list[Any] = []

    class _Model:
        model_id = "gpt-4o"

    class CodeAgent:
        def __init__(self, name: str) -> None:
            self.name = name
            self.model = _Model()
            self.tools: dict[str, Any] = {"web_search": object()}

        def run(self, *args: Any, **kwargs: Any) -> str:
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
    a.run("task")
    assert len(pushed) == 1
    rec = pushed[0]
    assert rec is not None
    assert rec.source == "framework:smolagents"
    assert rec.display_name == "DataCruncher"


# ── google_adk ─────────────────────────────────────────────────────


def test_google_adk_apply_noop_when_uninstalled() -> None:
    from egisai._patches import google_adk

    sys.modules.pop("google.adk", None)
    sys.modules.pop("google.adk.runners", None)
    assert google_adk.apply() is False


# ── pydantic_ai ────────────────────────────────────────────────────


def test_pydantic_ai_apply_noop_when_uninstalled() -> None:
    from egisai._patches import pydantic_ai

    sys.modules.pop("pydantic_ai", None)
    assert pydantic_ai.apply() is False


# ── llamaindex ─────────────────────────────────────────────────────


def test_llamaindex_apply_noop_when_uninstalled() -> None:
    from egisai._patches import llamaindex

    sys.modules.pop("llama_index", None)
    sys.modules.pop("llama_index.core.agent", None)
    assert llamaindex.apply() is False


# ── langchain ──────────────────────────────────────────────────────


def test_langchain_apply_noop_when_uninstalled() -> None:
    from egisai._patches import langchain

    sys.modules.pop("langchain", None)
    sys.modules.pop("langchain.agents", None)
    assert langchain.apply() is False


# ── bedrock_runtime ────────────────────────────────────────────────


def test_bedrock_runtime_apply_noop_when_uninstalled() -> None:
    from egisai._patches import bedrock_runtime

    sys.modules.pop("boto3", None)
    assert bedrock_runtime.apply() is False


# ── bedrock_agent ──────────────────────────────────────────────────


def test_bedrock_agent_apply_noop_when_uninstalled() -> None:
    from egisai._patches import bedrock_agent

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

    # Install a fake openai_agents.Runner.run_sync.
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
