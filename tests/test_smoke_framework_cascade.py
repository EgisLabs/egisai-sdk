"""End-to-end cascade tests for Tier-2 framework patches.

Tier-2 frameworks (``openai_agents``, ``langchain``, ``langgraph``,
``crewai``, ``autogen``, ``agno``, ``strands``, ``smolagents``,
``llamaindex``, ``pydantic_ai``, ``google_adk``) don't enforce policy
themselves — they're identity wrappers that open a Run, push an
:class:`IdentityRecord`, and delegate the actual LLM call governance
to the underlying provider patch (in practice almost always
``_patches/openai.py``, since these frameworks call OpenAI by default).

The cascade contract that needs to hold for every Tier-2 framework:

1. The framework's entry point is the **outer** wrap; it opens a Run
   and pushes its identity (``current_identity().source ==
   "framework:<name>"``).
2. **Inside** the framework's body, an OpenAI ``chat.completions.create``
   fires. That call goes through ``_patches/openai.py``'s gate, which
   reads ``current_identity()`` from the contextvar instead of
   re-deriving from the inner system prompt — and runs the
   ``pii_scan`` rule against the user's prompt.
3. When the rule says ``sanitize``, the SSN is masked **BEFORE** the
   underlying ``openai.chat.completions.create`` ships any bytes to
   the provider. The captured kwargs on the fake OpenAI client must
   NOT contain the raw SSN; the audit row stamped on the Run carries
   ``verdict=sanitize`` plus a ``sanitizations[].type=ssn`` record.
4. When the rule says ``block``, the gate raises ``PermissionError``
   inside the framework body. The framework wrap surfaces that
   PermissionError back to the user (the Run still closes); the
   captured kwargs on the fake OpenAI client are empty (the SDK was
   never actually called).
5. Privacy contract: the raw SSN never appears anywhere on the audit
   wire — not in payload previews, not in prompt previews, not on
   any framework's Run-step envelope.

These tests don't care about the framework's full SDK shape (those are
exhaustively covered by ``test_framework_patches.py``); they care
specifically about the cascade — the *invariant* that putting an
egisai-patched OpenAI call **inside** an egisai-patched framework
correctly composes the two patches so the policy still fires.

Why ``openai`` everywhere: it's the lingua franca for Tier-2
frameworks (LangChain → ``ChatOpenAI``, LangGraph → ``ChatOpenAI``,
CrewAI → ``OpenAI``, autogen → ``OpenAIChatCompletionClient``,
agno → ``OpenAIChat``, etc.). Pinning the cascade against the same
underlying provider keeps the matrix tractable AND mirrors the most
common production path.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

# ── Shared helpers ──────────────────────────────────────────────────


def _flush() -> None:
    from egisai import shutdown

    shutdown()


def _init_sdk(app: str = "cascade-smoke") -> None:
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app=app,
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )


def _pii_sanitize_rule(types_: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": "cascade-pii-san",
        "name": "cascade-sanitize-pii",
        "type": "pii_scan",
        "tenant": None,
        "config": {
            "action": "sanitize",
            "types": types_ or ["ssn"],
            "mask_char": "#",
        },
    }


def _pii_block_rule(types_: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": "cascade-pii-block",
        "name": "cascade-block-pii",
        "type": "pii_scan",
        "tenant": None,
        "config": {
            "action": "block",
            "types": types_ or ["ssn"],
            "message": "PII blocked",
        },
    }


def _set_rules(*rules: dict[str, Any]) -> None:
    from egisai._policy_cache import replace_rules

    replace_rules(f'"cr{len(rules)}"', list(rules))


def _all_audit_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in events:
        kind = e.get("kind")
        if kind in (None, "run.step"):
            out.append(e)
    return out


def _assert_raw_text_absent(
    events: list[dict[str, Any]], raw: str
) -> None:
    """Locked invariant: raw PII never appears on the wire.

    ``raw`` MUST be a real SSA-valid SSN — see the comment in
    ``test_smoke_provider_battery.py``'s anthropic sanitize test
    for why area numbers like 987 don't trip the detector.
    """
    for ev in events:
        assert raw not in repr(ev), (
            f"raw secret {raw!r} leaked into wire envelope "
            f"kind={ev.get('kind')} fields={list(ev)}"
        )


# ── Fake OpenAI SDK (the cascade's downstream provider) ────────────


def _install_fake_openai() -> type:
    """Plant a fake ``openai.resources.chat.completions.Completions``.

    Tier-2 frameworks call into this class transitively (LangChain
    via ``ChatOpenAI``, LangGraph the same, CrewAI directly, etc.).
    Putting one fake at the OpenAI seam means every Tier-2 cascade
    test goes through identical gate code paths — which is the whole
    point of the cascade invariant.
    """
    fake = types.ModuleType("openai")
    res = types.ModuleType("openai.resources")
    chat = types.ModuleType("openai.resources.chat")
    completions = types.ModuleType("openai.resources.chat.completions")
    responses = types.ModuleType("openai.resources.responses")

    class Completions:
        _captured_kwargs: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            return types.SimpleNamespace(
                id="cascade-out",
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="ok", tool_calls=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=types.SimpleNamespace(
                    prompt_tokens=3, completion_tokens=1,
                ),
            )

    class AsyncCompletions:
        _captured_kwargs: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            return types.SimpleNamespace(
                id="cascade-out-async",
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="ok", tool_calls=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=types.SimpleNamespace(
                    prompt_tokens=3, completion_tokens=1,
                ),
            )

    completions.Completions = Completions
    completions.AsyncCompletions = AsyncCompletions
    sys.modules.update(
        {
            "openai": fake,
            "openai.resources": res,
            "openai.resources.chat": chat,
            "openai.resources.chat.completions": completions,
            "openai.resources.responses": responses,
        }
    )
    Completions._captured_kwargs = []
    AsyncCompletions._captured_kwargs = []
    return Completions


@pytest.fixture
def cascade_env(fake_backend: Any) -> Iterator[tuple[Any, type]]:
    """SDK init + fake openai + openai patch applied.

    Every framework cascade test layers ITS own framework fixture on
    top of this base. The base proves the OpenAI seam is gated; the
    framework fixture proves the outer wrap composes correctly.
    """
    _init_sdk(app="cascade")
    Completions = _install_fake_openai()
    from egisai._patches import openai as openai_patch

    assert openai_patch.apply() is True
    yield fake_backend, Completions
    for mod in (
        "openai", "openai.resources", "openai.resources.chat",
        "openai.resources.chat.completions", "openai.resources.responses",
    ):
        sys.modules.pop(mod, None)


def _invoke_openai_inside(prompt: str) -> Any:
    """Helper: call our patched fake openai with a user prompt.

    Tier-2 framework bodies do this transitively (their own runtime
    wires ``ChatOpenAI`` → ``Completions.create``); in our test the
    fake framework's body invokes this directly to simulate that
    seam without depending on every framework's real package.
    """
    import openai.resources.chat.completions as oa_completions

    return oa_completions.Completions().create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )


# ── openai_agents cascade ──────────────────────────────────────────


@pytest.fixture
def openai_agents_cascade(cascade_env: Any) -> Iterator[tuple[Any, type, type]]:
    fake_backend, Completions = cascade_env
    fake = types.ModuleType("agents")

    class _Agent:
        def __init__(self, name: str, instructions: str = "") -> None:
            self.name = name
            self.instructions = instructions
            self.tools: list[Any] = []

    captured_prompts: list[str] = []

    class Runner:
        @staticmethod
        async def run(agent: _Agent, prompt: str, *a: Any, **kw: Any) -> Any:
            captured_prompts.append(prompt)
            return _invoke_openai_inside(prompt)

        @staticmethod
        def run_sync(agent: _Agent, prompt: str, *a: Any, **kw: Any) -> Any:
            captured_prompts.append(prompt)
            return _invoke_openai_inside(prompt)

        @staticmethod
        def run_streamed(agent: _Agent, prompt: str, *a: Any, **kw: Any) -> Any:
            captured_prompts.append(prompt)
            _invoke_openai_inside(prompt)
            return types.SimpleNamespace(stream_events=lambda: iter(()))

    fake.Runner = Runner
    fake._Agent = _Agent
    sys.modules["agents"] = fake

    from egisai._patches import openai_agents

    assert openai_agents.apply() is True
    yield fake_backend, Completions, _Agent
    sys.modules.pop("agents", None)


def test_openai_agents_cascade_sanitize(
    openai_agents_cascade: Any,
) -> None:
    fake_backend, Completions, _Agent = openai_agents_cascade
    _set_rules(_pii_sanitize_rule())
    raw = "111-22-3333"

    from agents import Runner  # type: ignore[import-not-found]

    Runner.run_sync(_Agent("TriageBot", "Route tickets."), f"SSN {raw}")
    _flush()

    sent_messages = Completions._captured_kwargs[-1]["messages"]
    assert raw not in sent_messages[-1]["content"], (
        "openai_agents → openai cascade let raw SSN through"
    )
    _assert_raw_text_absent(fake_backend.events_received, raw)
    san = [
        e for e in _all_audit_events(fake_backend.events_received)
        if e.get("verdict") == "sanitize"
    ]
    assert san, "no sanitize verdict on the cascade audit row"


def test_openai_agents_cascade_block(openai_agents_cascade: Any) -> None:
    fake_backend, Completions, _Agent = openai_agents_cascade
    _set_rules(_pii_block_rule())
    raw = "222-33-4444"

    from agents import Runner  # type: ignore[import-not-found]

    with pytest.raises(PermissionError):
        Runner.run_sync(_Agent("TriageBot", "Route tickets."), f"SSN {raw}")
    _flush()

    assert Completions._captured_kwargs == [], (
        "blocked cascade still let the openai SDK fire"
    )
    _assert_raw_text_absent(fake_backend.events_received, raw)


def test_openai_agents_cascade_async(openai_agents_cascade: Any) -> None:
    fake_backend, Completions, _Agent = openai_agents_cascade
    _set_rules(_pii_sanitize_rule())
    raw = "333-44-5555"

    from agents import Runner  # type: ignore[import-not-found]

    asyncio.run(Runner.run(_Agent("AsyncBot", "."), f"SSN {raw}"))
    _flush()

    assert raw not in Completions._captured_kwargs[-1]["messages"][-1]["content"]
    _assert_raw_text_absent(fake_backend.events_received, raw)


# ── LangGraph cascade ───────────────────────────────────────────────


@pytest.fixture
def langgraph_cascade(cascade_env: Any) -> Iterator[tuple[Any, type, type]]:
    """Patch ``langgraph.pregel.Pregel`` so its ``invoke`` runs a downstream openai call.

    LangChain 1.x's ``create_agent`` returns a
    ``CompiledStateGraph(Pregel)``; LangGraph's own ``add_node`` graphs
    also reach Pregel. Either way the gate the cascade needs to cover
    is ``Pregel.invoke`` → ``ChatOpenAI`` → ``openai.create``.
    """
    fake_backend, Completions = cascade_env
    lg = types.ModuleType("langgraph")
    pregel_mod = types.ModuleType("langgraph.pregel")
    cap: list[Any] = []

    class _Node:
        def __init__(self, name: str) -> None:
            self.name = name

    class Pregel:
        # Real Pregel.invoke takes ``input, config=None``.
        def __init__(self) -> None:
            self.nodes = {"agent": _Node("agent")}

        def invoke(self, input: Any, config: Any = None) -> Any:
            cap.append((input, config))
            text = (
                input.get("input")
                if isinstance(input, dict)
                else str(input)
            )
            _invoke_openai_inside(text)
            return {"output": "done"}

        async def ainvoke(self, input: Any, config: Any = None) -> Any:
            cap.append((input, config))
            text = (
                input.get("input")
                if isinstance(input, dict)
                else str(input)
            )
            _invoke_openai_inside(text)
            return {"output": "done"}

        def stream(self, input: Any, config: Any = None) -> Any:
            cap.append((input, config))
            text = (
                input.get("input")
                if isinstance(input, dict)
                else str(input)
            )
            _invoke_openai_inside(text)
            yield {"output": "done"}

    pregel_mod.Pregel = Pregel
    lg.pregel = pregel_mod
    sys.modules["langgraph"] = lg
    sys.modules["langgraph.pregel"] = pregel_mod

    from egisai._patches import langgraph as lg_patch

    assert lg_patch.apply() is True
    yield fake_backend, Completions, Pregel
    sys.modules.pop("langgraph", None)
    sys.modules.pop("langgraph.pregel", None)


def test_langgraph_cascade_sanitize(langgraph_cascade: Any) -> None:
    fake_backend, Completions, Pregel = langgraph_cascade
    _set_rules(_pii_sanitize_rule())
    raw = "444-55-6789"

    Pregel().invoke({"input": f"verify SSN {raw} please"})
    _flush()

    sent = Completions._captured_kwargs[-1]["messages"][-1]["content"]
    assert raw not in sent, "langgraph cascade leaked raw SSN to openai"
    _assert_raw_text_absent(fake_backend.events_received, raw)


def test_langgraph_cascade_block(langgraph_cascade: Any) -> None:
    fake_backend, Completions, Pregel = langgraph_cascade
    _set_rules(_pii_block_rule())
    raw = "555-66-7890"

    with pytest.raises(PermissionError):
        Pregel().invoke({"input": f"SSN {raw}"})
    _flush()

    assert Completions._captured_kwargs == []
    _assert_raw_text_absent(fake_backend.events_received, raw)


# ── LangChain cascade ───────────────────────────────────────────────


@pytest.fixture
def langchain_cascade(cascade_env: Any) -> Iterator[tuple[Any, type, type]]:
    fake_backend, Completions = cascade_env
    lc = types.ModuleType("langchain")
    agents_mod = types.ModuleType("langchain.agents")
    cap: list[Any] = []

    class _LLMChain:
        prompt = types.SimpleNamespace(
            template="You are a helpful research agent."
        )

    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Agent:
        llm_chain = _LLMChain()
        system_message = "You are a helpful research agent."

    class AgentExecutor:
        def __init__(self) -> None:
            self.name = "researcher"
            self.agent = _Agent()
            self.tools = [_Tool("search")]

        def invoke(self, inputs: dict[str, Any], *a: Any, **kw: Any) -> Any:
            cap.append(inputs)
            _invoke_openai_inside(str(inputs.get("input", "")))
            return {"output": "done"}

        async def ainvoke(
            self, inputs: dict[str, Any], *a: Any, **kw: Any
        ) -> Any:
            cap.append(inputs)
            _invoke_openai_inside(str(inputs.get("input", "")))
            return {"output": "done"}

        def stream(
            self, inputs: dict[str, Any], *a: Any, **kw: Any
        ) -> Any:
            cap.append(inputs)
            _invoke_openai_inside(str(inputs.get("input", "")))
            yield {"output": "done"}

    agents_mod.AgentExecutor = AgentExecutor
    lc.agents = agents_mod
    sys.modules["langchain"] = lc
    sys.modules["langchain.agents"] = agents_mod

    from egisai._patches import langchain as lc_patch

    assert lc_patch.apply() is True
    yield fake_backend, Completions, AgentExecutor
    sys.modules.pop("langchain", None)
    sys.modules.pop("langchain.agents", None)


def test_langchain_cascade_sanitize(langchain_cascade: Any) -> None:
    fake_backend, Completions, AgentExecutor = langchain_cascade
    _set_rules(_pii_sanitize_rule())
    # Area 666 is in the SSA's invalid set (along with 900-999) and
    # the detector won't trip on it — use a real assigned range so
    # the cascade test exercises a true positive end-to-end.
    raw = "234-56-7891"

    AgentExecutor().invoke({"input": f"trace SSN {raw} for me"})
    _flush()

    assert raw not in Completions._captured_kwargs[-1]["messages"][-1][
        "content"
    ]
    _assert_raw_text_absent(fake_backend.events_received, raw)


def test_langchain_cascade_block(langchain_cascade: Any) -> None:
    fake_backend, Completions, AgentExecutor = langchain_cascade
    _set_rules(_pii_block_rule())
    raw = "777-88-9012"

    with pytest.raises(PermissionError):
        AgentExecutor().invoke({"input": f"SSN {raw}"})
    _flush()

    assert Completions._captured_kwargs == []
    _assert_raw_text_absent(fake_backend.events_received, raw)


# ── CrewAI cascade ──────────────────────────────────────────────────


@pytest.fixture
def crewai_cascade(cascade_env: Any) -> Iterator[tuple[Any, type, type]]:
    fake_backend, Completions = cascade_env
    crewai = types.ModuleType("crewai")
    crewai_agent = types.ModuleType("crewai.agent")

    class _CrewAgent:
        def __init__(self, role: str) -> None:
            self.role = role
            self.goal = "verify customer data"
            self.backstory = ""
            self.tools: list[Any] = []

        def execute_task(self, task: Any, *a: Any, **kw: Any) -> Any:
            prompt = getattr(task, "description", str(task))
            _invoke_openai_inside(prompt)
            return "done"

    crewai_agent.Agent = _CrewAgent
    crewai.Agent = _CrewAgent
    sys.modules["crewai"] = crewai
    sys.modules["crewai.agent"] = crewai_agent

    from egisai._patches import crewai as crewai_patch

    assert crewai_patch.apply() is True
    yield fake_backend, Completions, _CrewAgent
    sys.modules.pop("crewai", None)
    sys.modules.pop("crewai.agent", None)


def test_crewai_cascade_sanitize(crewai_cascade: Any) -> None:
    fake_backend, Completions, _CrewAgent = crewai_cascade
    _set_rules(_pii_sanitize_rule())
    raw = "234-56-7890"

    agent = _CrewAgent("Data Auditor")
    task = types.SimpleNamespace(description=f"audit SSN {raw}")
    agent.execute_task(task)
    _flush()

    assert raw not in Completions._captured_kwargs[-1]["messages"][-1][
        "content"
    ]
    _assert_raw_text_absent(fake_backend.events_received, raw)


def test_crewai_cascade_block(crewai_cascade: Any) -> None:
    fake_backend, Completions, _CrewAgent = crewai_cascade
    _set_rules(_pii_block_rule())
    raw = "345-67-8901"

    agent = _CrewAgent("Data Auditor")
    task = types.SimpleNamespace(description=f"audit SSN {raw}")
    with pytest.raises(PermissionError):
        agent.execute_task(task)
    _flush()

    assert Completions._captured_kwargs == []
    _assert_raw_text_absent(fake_backend.events_received, raw)


# ── AutoGen cascade ─────────────────────────────────────────────────


@pytest.fixture
def autogen_cascade(cascade_env: Any) -> Iterator[tuple[Any, type, type]]:
    fake_backend, Completions = cascade_env
    autogen = types.ModuleType("autogen_agentchat")
    agents_mod = types.ModuleType("autogen_agentchat.agents")

    class AssistantAgent:
        def __init__(
            self,
            name: str,
            system_message: str = "",
            model_client: Any = None,
            tools: list[Any] | None = None,
        ) -> None:
            self.name = name
            self.system_message = system_message
            self.model_client = model_client
            self.tools = tools or []

        async def run(self, *, task: str, **kwargs: Any) -> Any:
            _invoke_openai_inside(task)
            return types.SimpleNamespace(messages=[], stop_reason="stop")

        def run_stream(self, *, task: str, **kwargs: Any) -> Any:
            _invoke_openai_inside(task)

            async def gen() -> Any:
                if False:
                    yield {"event": "msg"}

            return gen()

    agents_mod.AssistantAgent = AssistantAgent
    autogen.agents = agents_mod
    sys.modules["autogen_agentchat"] = autogen
    sys.modules["autogen_agentchat.agents"] = agents_mod

    from egisai._patches import autogen as autogen_patch

    assert autogen_patch.apply() is True
    yield fake_backend, Completions, AssistantAgent
    sys.modules.pop("autogen_agentchat", None)
    sys.modules.pop("autogen_agentchat.agents", None)


def test_autogen_cascade_sanitize(autogen_cascade: Any) -> None:
    fake_backend, Completions, AssistantAgent = autogen_cascade
    _set_rules(_pii_sanitize_rule())
    raw = "456-78-9012"

    agent = AssistantAgent("Helper", "Helpful assistant.")
    asyncio.run(agent.run(task=f"SSN {raw}"))
    _flush()

    assert raw not in Completions._captured_kwargs[-1]["messages"][-1][
        "content"
    ]
    _assert_raw_text_absent(fake_backend.events_received, raw)


def test_autogen_cascade_block(autogen_cascade: Any) -> None:
    fake_backend, Completions, AssistantAgent = autogen_cascade
    _set_rules(_pii_block_rule())
    raw = "567-89-0123"

    agent = AssistantAgent("Helper", "Helpful assistant.")
    with pytest.raises(PermissionError):
        asyncio.run(agent.run(task=f"SSN {raw}"))
    _flush()

    assert Completions._captured_kwargs == []
    _assert_raw_text_absent(fake_backend.events_received, raw)


# ── Agno cascade ────────────────────────────────────────────────────


@pytest.fixture
def agno_cascade(cascade_env: Any) -> Iterator[tuple[Any, type, type]]:
    fake_backend, Completions = cascade_env
    agno = types.ModuleType("agno")
    agno_agent_mod = types.ModuleType("agno.agent")

    class _AgnoAgent:
        def __init__(self, name: str, instructions: str = "") -> None:
            self.name = name
            self.instructions = instructions
            self.tools: list[Any] = []

        def run(self, prompt: str, *a: Any, **kw: Any) -> Any:
            _invoke_openai_inside(prompt)
            return types.SimpleNamespace(content="ok")

        async def arun(self, prompt: str, *a: Any, **kw: Any) -> Any:
            _invoke_openai_inside(prompt)
            return types.SimpleNamespace(content="ok")

    agno_agent_mod.Agent = _AgnoAgent
    agno.agent = agno_agent_mod
    sys.modules["agno"] = agno
    sys.modules["agno.agent"] = agno_agent_mod

    from egisai._patches import agno as agno_patch

    assert agno_patch.apply() is True
    yield fake_backend, Completions, _AgnoAgent
    sys.modules.pop("agno", None)
    sys.modules.pop("agno.agent", None)


def test_agno_cascade_sanitize(agno_cascade: Any) -> None:
    fake_backend, Completions, _AgnoAgent = agno_cascade
    _set_rules(_pii_sanitize_rule())
    raw = "678-90-1234"

    agent = _AgnoAgent("InfoBot", "Be helpful.")
    agent.run(f"please look up SSN {raw}")
    _flush()

    assert raw not in Completions._captured_kwargs[-1]["messages"][-1][
        "content"
    ]
    _assert_raw_text_absent(fake_backend.events_received, raw)


def test_agno_cascade_block(agno_cascade: Any) -> None:
    fake_backend, Completions, _AgnoAgent = agno_cascade
    _set_rules(_pii_block_rule())
    raw = "789-01-2345"

    agent = _AgnoAgent("InfoBot", "Be helpful.")
    with pytest.raises(PermissionError):
        agent.run(f"please look up SSN {raw}")
    _flush()

    assert Completions._captured_kwargs == []
    _assert_raw_text_absent(fake_backend.events_received, raw)


# ── Strands cascade ─────────────────────────────────────────────────


@pytest.fixture
def strands_cascade(cascade_env: Any) -> Iterator[tuple[Any, type, type]]:
    fake_backend, Completions = cascade_env
    strands = types.ModuleType("strands")

    class _StrandsAgent:
        def __init__(self, name: str = "StrandsBot") -> None:
            self.name = name
            self.system_prompt = "You are helpful."
            self.tools: list[Any] = []
            self.model = "gpt-4o"

        def __call__(self, prompt: str, *a: Any, **kw: Any) -> Any:
            _invoke_openai_inside(prompt)
            return types.SimpleNamespace(message=types.SimpleNamespace(
                content=[types.SimpleNamespace(text="ok")]
            ))

    strands.Agent = _StrandsAgent
    sys.modules["strands"] = strands

    from egisai._patches import strands as strands_patch

    assert strands_patch.apply() is True
    yield fake_backend, Completions, _StrandsAgent
    sys.modules.pop("strands", None)


def test_strands_cascade_sanitize(strands_cascade: Any) -> None:
    fake_backend, Completions, _StrandsAgent = strands_cascade
    _set_rules(_pii_sanitize_rule())
    raw = "890-12-3456"

    agent = _StrandsAgent()
    agent(f"lookup SSN {raw}")
    _flush()

    assert raw not in Completions._captured_kwargs[-1]["messages"][-1][
        "content"
    ]
    _assert_raw_text_absent(fake_backend.events_received, raw)


def test_strands_cascade_block(strands_cascade: Any) -> None:
    fake_backend, Completions, _StrandsAgent = strands_cascade
    _set_rules(_pii_block_rule())
    raw = "012-34-5678"

    agent = _StrandsAgent()
    with pytest.raises(PermissionError):
        agent(f"lookup SSN {raw}")
    _flush()

    assert Completions._captured_kwargs == []
    _assert_raw_text_absent(fake_backend.events_received, raw)


# ── Pydantic AI cascade ─────────────────────────────────────────────


@pytest.fixture
def pydantic_ai_cascade(cascade_env: Any) -> Iterator[tuple[Any, type, type]]:
    fake_backend, Completions = cascade_env
    pa = types.ModuleType("pydantic_ai")

    class _PydanticAgent:
        def __init__(self, model: str, *, system_prompt: str = "") -> None:
            self.model = model
            self.name = "pydantic-ai-agent"
            self._system_prompt = system_prompt
            self.tools: list[Any] = []

        async def run(self, prompt: str, *a: Any, **kw: Any) -> Any:
            _invoke_openai_inside(prompt)
            return types.SimpleNamespace(output="ok")

        def run_sync(self, prompt: str, *a: Any, **kw: Any) -> Any:
            _invoke_openai_inside(prompt)
            return types.SimpleNamespace(output="ok")

        def run_stream(self, prompt: str, *a: Any, **kw: Any) -> Any:
            _invoke_openai_inside(prompt)

            async def stream() -> Any:
                if False:
                    yield "chunk"

            return stream()

    pa.Agent = _PydanticAgent
    sys.modules["pydantic_ai"] = pa

    from egisai._patches import pydantic_ai as pa_patch

    assert pa_patch.apply() is True
    yield fake_backend, Completions, _PydanticAgent
    sys.modules.pop("pydantic_ai", None)


def test_pydantic_ai_cascade_sanitize(pydantic_ai_cascade: Any) -> None:
    fake_backend, Completions, _PydanticAgent = pydantic_ai_cascade
    _set_rules(_pii_sanitize_rule())
    raw = "123-45-6789"

    agent = _PydanticAgent("openai:gpt-4o", system_prompt="be helpful")
    agent.run_sync(f"SSN {raw}")
    _flush()

    assert raw not in Completions._captured_kwargs[-1]["messages"][-1][
        "content"
    ]
    _assert_raw_text_absent(fake_backend.events_received, raw)


def test_pydantic_ai_cascade_block(pydantic_ai_cascade: Any) -> None:
    fake_backend, Completions, _PydanticAgent = pydantic_ai_cascade
    _set_rules(_pii_block_rule())
    raw = "234-56-7890"

    agent = _PydanticAgent("openai:gpt-4o", system_prompt="be helpful")
    with pytest.raises(PermissionError):
        agent.run_sync(f"SSN {raw}")
    _flush()

    assert Completions._captured_kwargs == []
    _assert_raw_text_absent(fake_backend.events_received, raw)


# ── Privacy invariant across every cascade ─────────────────────────


def test_all_cascade_audit_rows_carry_no_response_preview(
    openai_agents_cascade: Any,
) -> None:
    """Every cascade Run-step audit row must NOT carry a
    ``response_preview`` field — the SDK never persists model output."""
    fake_backend, Completions, _Agent = openai_agents_cascade
    _set_rules(_pii_sanitize_rule())

    from agents import Runner  # type: ignore[import-not-found]

    Runner.run_sync(_Agent("X", "."), "SSN 345-67-8901")
    _flush()
    for ev in fake_backend.events_received:
        assert "response_preview" not in ev, (
            f"cascade audit row leaked response_preview: keys={list(ev)}"
        )
        if "step_payload" in ev:
            sp = ev.get("step_payload") or {}
            assert "response_preview" not in sp, (
                f"cascade step_payload leaked response_preview: {sp!r}"
            )
