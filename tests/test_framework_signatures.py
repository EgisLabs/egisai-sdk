"""Signature-parity gate for every framework patch.

The 0.17.2 ``TypeError: object async_generator can't be used in
'await' expression`` regression on ``ClaudeSDKClient.query`` shipped
because the in-repo test stub was an async-generator function while
the real upstream was a coroutine — so our wrong-kind wrap (it
turned a coroutine into an async generator) "passed" the unit test
but crashed in user code.

This file is the defence-in-depth gate that would have caught it:
for every (framework, method) we patch, we build a stub whose
signature shape **mirrors the real upstream library** (verified
empirically with ``inspect`` on each installed package — see
``test_framework_audit.py`` for the optional gate that re-checks
against the real libs when present), run ``apply()``, and assert
``inspect.iscoroutinefunction`` / ``isasyncgenfunction`` /
``isgeneratorfunction`` on the **wrapped** attribute matches the
original. Any wrong-kind regression now fails here loudly.

The shape table:

   framework               method                       kind         coro  agen  sgen
   claude_agent_sdk        ClaudeSDKClient.query        async         T     F     F
   claude_agent_sdk        query (module-level)         async_iter    F     T     F
   openai_agents (agents)  Runner.run                   async         T     F     F
   openai_agents (agents)  Runner.run_sync              sync          F     F     F
   openai_agents (agents)  Runner.run_streamed          sync          F     F     F
   langgraph               Pregel.invoke                sync          F     F     F
   langgraph               Pregel.ainvoke               async         T     F     F
   langgraph               Pregel.stream                sync_iter     F     F     T
   langgraph               Pregel.astream               async_iter    F     T     F
   autogen_agentchat       AssistantAgent.run           async         T     F     F
   autogen_agentchat       AssistantAgent.run_stream    async_iter    F     T     F
   crewai                  Agent.execute_task           sync          F     F     F
   agno                    Agent.run                    polymorphic   F     F     F
   agno                    Agent.arun                   polymorphic   F     F     F
   agno                    Agent.print_response         sync          F     F     F
   smolagents              {Multi,Tool,Code}Agent.run   polymorphic   F     F     F
   strands                 Agent.__call__               sync          F     F     F
   strands                 Agent.invoke_async           async         T     F     F
   pydantic_ai             Agent.run                    async         T     F     F
   pydantic_ai             Agent.run_sync               sync          F     F     F
   llama_index.core.agent  *Agent.run                   sync          F     F     F
   google.adk.runners      Runner.run                   sync_iter     F     F     T
   google.adk.runners      Runner.run_async             async_iter    F     T     F
"""

from __future__ import annotations

import inspect
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

# ── Helpers ─────────────────────────────────────────────────────────


def _make_fake_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


@pytest.fixture
def cleanup_modules() -> Iterator[list[str]]:
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


def _assert_shape(
    method: Any,
    *,
    coro: bool,
    agen: bool,
    sgen: bool,
    label: str,
) -> None:
    """Assert the wrapped method preserves the upstream's call shape."""
    got_coro = inspect.iscoroutinefunction(method)
    got_agen = inspect.isasyncgenfunction(method)
    got_sgen = inspect.isgeneratorfunction(method)
    assert got_coro == coro, (
        f"{label}: iscoroutinefunction expected {coro}, got {got_coro}"
    )
    assert got_agen == agen, (
        f"{label}: isasyncgenfunction expected {agen}, got {got_agen}"
    )
    assert got_sgen == sgen, (
        f"{label}: isgeneratorfunction expected {sgen}, got {got_sgen}"
    )


# ── claude_agent_sdk ────────────────────────────────────────────────


def test_signature_claude_agent_sdk(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
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
    # Module-level ``query`` is an async generator.
    _assert_shape(
        mod.query, coro=False, agen=True, sgen=False,
        label="claude_agent_sdk.query",
    )
    # ``ClaudeSDKClient.query`` is a coroutine — NOT an async generator.
    _assert_shape(
        ClaudeSDKClient.query, coro=True, agen=False, sgen=False,
        label="ClaudeSDKClient.query",
    )


# ── openai_agents ───────────────────────────────────────────────────


def test_signature_openai_agents(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    _init_sdk(fake_backend)
    mod = _make_fake_module("agents")
    cleanup_modules.append("agents")

    class _Agent:
        name = "n"
        instructions = ""
        tools: list[Any] = []

    class Runner:
        @staticmethod
        async def run(agent: _Agent, *args: Any, **kwargs: Any) -> str:
            return "ok"

        @staticmethod
        def run_sync(agent: _Agent, *args: Any, **kwargs: Any) -> str:
            return "ok"

        @staticmethod
        def run_streamed(agent: _Agent, *args: Any, **kwargs: Any) -> Any:
            return object()

    mod.Runner = Runner

    from egisai._patches import openai_agents

    openai_agents.apply()
    _assert_shape(
        Runner.run, coro=True, agen=False, sgen=False,
        label="openai_agents Runner.run",
    )
    _assert_shape(
        Runner.run_sync, coro=False, agen=False, sgen=False,
        label="openai_agents Runner.run_sync",
    )
    _assert_shape(
        Runner.run_streamed, coro=False, agen=False, sgen=False,
        label="openai_agents Runner.run_streamed",
    )


# ── langgraph ───────────────────────────────────────────────────────


def test_signature_langgraph(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    _init_sdk(fake_backend)
    mod = _make_fake_module("langgraph")
    sub = _make_fake_module("langgraph.pregel")
    cleanup_modules.extend(["langgraph", "langgraph.pregel"])

    class Pregel:
        def __init__(self) -> None:
            self.nodes: dict[str, Any] = {}

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
    _assert_shape(Pregel.invoke, coro=False, agen=False, sgen=False, label="Pregel.invoke")
    _assert_shape(Pregel.ainvoke, coro=True, agen=False, sgen=False, label="Pregel.ainvoke")
    _assert_shape(Pregel.stream, coro=False, agen=False, sgen=True, label="Pregel.stream")
    _assert_shape(Pregel.astream, coro=False, agen=True, sgen=False, label="Pregel.astream")


# ── autogen ────────────────────────────────────────────────────────


def test_signature_autogen(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
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
    _assert_shape(
        AssistantAgent.run, coro=True, agen=False, sgen=False,
        label="autogen AssistantAgent.run",
    )
    _assert_shape(
        AssistantAgent.run_stream, coro=False, agen=True, sgen=False,
        label="autogen AssistantAgent.run_stream",
    )


# ── crewai ─────────────────────────────────────────────────────────


def test_signature_crewai(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    _init_sdk(fake_backend)
    mod = _make_fake_module("crewai")
    cleanup_modules.append("crewai")

    class Agent:
        def __init__(self, role: str = "r") -> None:
            self.role = role
            self.tools: list[Any] = []

        def execute_task(self, *args: Any, **kwargs: Any) -> str:
            return "ok"

    mod.Agent = Agent

    from egisai._patches import crewai

    crewai.apply()
    _assert_shape(
        Agent.execute_task, coro=False, agen=False, sgen=False,
        label="crewai Agent.execute_task",
    )


# ── strands ────────────────────────────────────────────────────────


def test_signature_strands(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    _init_sdk(fake_backend)
    mod = _make_fake_module("strands")
    cleanup_modules.append("strands")

    class Agent:
        def __init__(self, name: str = "n") -> None:
            self.name = name
            self.tools: list[Any] = []

        def __call__(self, *args: Any, **kwargs: Any) -> str:
            return "ok"

        async def invoke_async(self, *args: Any, **kwargs: Any) -> str:
            return "ok"

    mod.Agent = Agent

    from egisai._patches import strands

    strands.apply()
    _assert_shape(
        Agent.__call__, coro=False, agen=False, sgen=False,
        label="strands Agent.__call__",
    )
    _assert_shape(
        Agent.invoke_async, coro=True, agen=False, sgen=False,
        label="strands Agent.invoke_async",
    )


# ── pydantic_ai ────────────────────────────────────────────────────


def test_signature_pydantic_ai(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    _init_sdk(fake_backend)
    mod = _make_fake_module("pydantic_ai")
    cleanup_modules.append("pydantic_ai")

    class Agent:
        def __init__(self) -> None:
            self.model = "gpt-4o"
            self.system_prompt = ""
            self.tools: list[Any] = []

        async def run(self, *args: Any, **kwargs: Any) -> str:
            return "ok"

        def run_sync(self, *args: Any, **kwargs: Any) -> str:
            return "ok"

    mod.Agent = Agent

    from egisai._patches import pydantic_ai

    pydantic_ai.apply()
    _assert_shape(
        Agent.run, coro=True, agen=False, sgen=False,
        label="pydantic_ai Agent.run",
    )
    _assert_shape(
        Agent.run_sync, coro=False, agen=False, sgen=False,
        label="pydantic_ai Agent.run_sync",
    )


# ── llamaindex ─────────────────────────────────────────────────────


def test_signature_llamaindex(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """LlamaIndex agent .run methods are all sync ``def`` returning a
    ``WorkflowHandler``. The wrapped attribute MUST stay sync —
    wrapping as ``async`` (the 0.17.0–0.17.4 bug) hides the handle
    inside a coroutine and breaks ``.stream_events()``."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("llama_index")
    pkg = _make_fake_module("llama_index.core")
    sub = _make_fake_module("llama_index.core.agent")
    cleanup_modules.extend([
        "llama_index", "llama_index.core", "llama_index.core.agent",
    ])
    mod.core = pkg
    pkg.agent = sub

    class FunctionAgent:
        def __init__(self) -> None:
            self.name = "n"
            self.system_prompt = ""
            self.tools: list[Any] = []

        def run(self, *args: Any, **kwargs: Any) -> object:
            return object()

    sub.FunctionAgent = FunctionAgent
    sub.ReActAgent = FunctionAgent
    sub.CodeActAgent = FunctionAgent
    sub.AgentWorkflow = FunctionAgent

    from egisai._patches import llamaindex

    llamaindex.apply()
    _assert_shape(
        FunctionAgent.run, coro=False, agen=False, sgen=False,
        label="llamaindex *Agent.run",
    )


# ── agno (polymorphic) ─────────────────────────────────────────────


def test_signature_agno(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    """Agno's ``run`` and ``arun`` are plain ``def``s that *return*
    polymorphic values (coro/async-iter/sync-iter/plain). The patched
    method itself must remain a plain function — the polymorphism is
    resolved at runtime, not at the function-attribute level."""
    _init_sdk(fake_backend)
    mod = _make_fake_module("agno")
    sub = _make_fake_module("agno.agent")
    cleanup_modules.extend(["agno", "agno.agent"])

    class Agent:
        def __init__(self) -> None:
            self.name = "n"
            self.description = ""
            self.instructions = ""

        def run(self, *args: Any, stream: bool = False, **kwargs: Any) -> Any:
            return "ok"

        def arun(self, *args: Any, stream: bool = False, **kwargs: Any) -> Any:
            async def _coro() -> str:
                return "ok"
            return _coro()

        def print_response(self, *args: Any, **kwargs: Any) -> None:
            return None

    sub.Agent = Agent
    mod.agent = sub

    from egisai._patches import agno

    agno.apply()
    _assert_shape(
        Agent.run, coro=False, agen=False, sgen=False, label="agno Agent.run",
    )
    _assert_shape(
        Agent.arun, coro=False, agen=False, sgen=False, label="agno Agent.arun",
    )
    _assert_shape(
        Agent.print_response, coro=False, agen=False, sgen=False,
        label="agno Agent.print_response",
    )


# ── smolagents (polymorphic) ───────────────────────────────────────


def test_signature_smolagents(
    fake_backend: Any, cleanup_modules: list[str]
) -> None:
    _init_sdk(fake_backend)
    mod = _make_fake_module("smolagents")
    cleanup_modules.append("smolagents")

    class _Model:
        model_id = "gpt-4o"

    class CodeAgent:
        def __init__(self) -> None:
            self.name = "n"
            self.model = _Model()
            self.tools: dict[str, Any] = {}

        def run(self, *args: Any, stream: bool = False, **kwargs: Any) -> Any:
            return "ok"

    class MultiStepAgent(CodeAgent):
        pass

    class ToolCallingAgent(CodeAgent):
        pass

    mod.CodeAgent = CodeAgent
    mod.MultiStepAgent = MultiStepAgent
    mod.ToolCallingAgent = ToolCallingAgent

    from egisai._patches import smolagents

    smolagents.apply()
    for cls in (CodeAgent, MultiStepAgent, ToolCallingAgent):
        _assert_shape(
            cls.run, coro=False, agen=False, sgen=False,
            label=f"smolagents {cls.__name__}.run",
        )
