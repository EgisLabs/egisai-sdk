"""The handoff between the SDK and an egress node.

One call must produce one audit row. When a customer runs both control
points — the SDK inside services their team wrote, a node in front of
everything else — the only thing preventing two is this header. So the
tests are mostly about when it is *absent*: on ungoverned calls, on
traffic to the customer's own services, and on anything the SDK merely
observed.
"""

from __future__ import annotations

import types

import pytest

from egisai._context import set_policy_checked
from egisai._patches import decision


class _Url:
    def __init__(self, host: str) -> None:
        self.host = host


def _request(host: str, headers: dict | None = None) -> types.SimpleNamespace:
    return types.SimpleNamespace(url=_Url(host), headers=dict(headers or {}))


@pytest.fixture(autouse=True)
def _clean():
    set_policy_checked(False)
    yield
    set_policy_checked(False)


class TestWhenItIsStamped:
    def test_a_governed_call_to_a_vendor_carries_the_marker(self) -> None:
        set_policy_checked(True)
        request = _request("api.openai.com")
        decision._stamp(request)
        assert request.headers[decision.HEADER] == decision.VALUE

    def test_every_vendor_the_node_watches_is_covered(self) -> None:
        # A host missing here costs a duplicate row. Not fatal, but it
        # is the kind of gap nobody notices until an auditor does.
        set_policy_checked(True)
        for host in decision._HOSTS:
            request = _request(host)
            decision._stamp(request)
            assert decision.HEADER in request.headers, host

    def test_azure_and_bedrock_endpoints_count(self) -> None:
        set_policy_checked(True)
        for host in (
            "my-org.openai.azure.com",
            "bedrock-runtime.us-east-1.amazonaws.com",
        ):
            request = _request(host)
            decision._stamp(request)
            assert decision.HEADER in request.headers, host


class TestWhenItIsNot:
    def test_an_ungoverned_call_is_left_alone(self) -> None:
        # This is the important one. If the SDK did not decide, the
        # node must — stamping here would create a hole where nothing
        # governs the call at all.
        set_policy_checked(False)
        request = _request("api.openai.com")
        decision._stamp(request)
        assert decision.HEADER not in request.headers

    def test_the_customers_own_api_never_sees_it(self) -> None:
        set_policy_checked(True)
        request = _request("internal.acme.example")
        decision._stamp(request)
        assert decision.HEADER not in request.headers

    def test_an_explicit_header_is_not_overwritten(self) -> None:
        set_policy_checked(True)
        request = _request("api.openai.com", {decision.HEADER: "mine"})
        decision._stamp(request)
        assert request.headers[decision.HEADER] == "mine"

    def test_a_request_with_no_url_does_not_raise(self) -> None:
        set_policy_checked(True)
        broken = types.SimpleNamespace(headers={})
        decision._stamp(broken)
        assert broken.headers == {}

    def test_immutable_headers_do_not_break_the_call(self) -> None:
        # Never worth failing a customer's model call over a header we
        # could not add.
        class _Frozen(dict):
            def __setitem__(self, *_a):
                raise TypeError("read-only")

        set_policy_checked(True)
        request = types.SimpleNamespace(
            url=_Url("api.openai.com"), headers=_Frozen()
        )
        decision._stamp(request)


class TestPatching:
    def test_applying_twice_does_not_stack_wrappers(self) -> None:
        httpx = pytest.importorskip("httpx")
        decision.apply()
        first = httpx.Client.send
        decision.apply()
        assert httpx.Client.send is first

    def test_the_value_carries_no_authority(self) -> None:
        # A verdict here would invite somebody to trust it, and the
        # node has no way to tell a real one from a forged one.
        assert decision.VALUE == "governed"
