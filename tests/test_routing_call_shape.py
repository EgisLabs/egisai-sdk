"""SDK-side call-shape detection for Smart Model Routing.

``inspect_payload`` reads three facts off the outgoing payload that
the prompt preview can't carry, and each one guards against a routed
call that would *fail* rather than merely cost more:

* images  — a text-only target rejects the request outright.
* max out — a target with a lower ceiling truncates or 400s.
* caching — a swap cold-starts the cache, so the "cheaper" model can
  bill more than the one the caller asked for.

The payloads below are the real shapes the OpenAI and Anthropic
patches build, plus the malformed ones a user's own code can produce.
Detection must never raise: this runs on the model-call hot path, and
the SDK's whole contract is that governance can't break the call.
"""

from __future__ import annotations

from egisai._routing import CallShape, inspect_payload


class TestDefensive:
    def test_non_dict_payload_is_the_default_shape(self) -> None:
        assert inspect_payload(None) == CallShape()
        assert inspect_payload("nope") == CallShape()
        assert inspect_payload(42) == CallShape()

    def test_empty_payload_imposes_no_restriction(self) -> None:
        assert inspect_payload({}) == CallShape()

    def test_malformed_messages_do_not_raise(self) -> None:
        shape = inspect_payload(
            {"messages": [None, "text", 7, {"content": object()}]}
        )
        assert shape.has_images is False
        assert shape.uses_prompt_caching is False

    def test_exotic_content_types_do_not_raise(self) -> None:
        shape = inspect_payload({"messages": [{"content": {"weird": True}}]})
        assert shape == CallShape()


class TestTools:
    def test_tools_detected(self) -> None:
        assert inspect_payload({"tools": [{"name": "search"}]}).has_tools

    def test_legacy_functions_detected(self) -> None:
        assert inspect_payload({"functions": [{"name": "f"}]}).has_tools

    def test_empty_tools_is_not_tool_use(self) -> None:
        assert not inspect_payload({"tools": []}).has_tools


class TestImages:
    def test_openai_vision_message(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what's here?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x/y.png"},
                        },
                    ],
                }
            ]
        }
        assert inspect_payload(payload).has_images

    def test_anthropic_image_block(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "data": "..."},
                        }
                    ],
                }
            ]
        }
        assert inspect_payload(payload).has_images

    def test_plain_string_content_is_text_only(self) -> None:
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        assert not inspect_payload(payload).has_images


class TestMaxOutput:
    def test_anthropic_max_tokens(self) -> None:
        assert inspect_payload({"max_tokens": 4096}).max_output_tokens == 4096

    def test_openai_max_completion_tokens(self) -> None:
        payload = {"max_completion_tokens": 16_384}
        assert inspect_payload(payload).max_output_tokens == 16_384

    def test_largest_wins_when_several_are_present(self) -> None:
        payload = {"max_tokens": 512, "max_output_tokens": 64_000}
        assert inspect_payload(payload).max_output_tokens == 64_000

    def test_absent_is_unspecified(self) -> None:
        assert inspect_payload({}).max_output_tokens == 0

    def test_non_int_values_are_ignored(self) -> None:
        assert inspect_payload({"max_tokens": "4096"}).max_output_tokens == 0
        assert inspect_payload({"max_tokens": None}).max_output_tokens == 0
        assert inspect_payload({"max_tokens": True}).max_output_tokens == 0

    def test_negative_is_clamped_to_unspecified(self) -> None:
        assert inspect_payload({"max_tokens": -10}).max_output_tokens == 0


class TestPromptCaching:
    def test_cache_control_on_a_content_block(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "a very long document",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        }
        assert inspect_payload(payload).uses_prompt_caching

    def test_cache_control_on_the_anthropic_system_list(self) -> None:
        payload = {
            "system": [
                {
                    "type": "text",
                    "text": "long system prompt",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
        assert inspect_payload(payload).uses_prompt_caching

    def test_plain_string_system_is_not_cached(self) -> None:
        payload = {
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hi"}],
        }
        assert not inspect_payload(payload).uses_prompt_caching

    def test_uncached_call_is_routable(self) -> None:
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        assert not inspect_payload(payload).uses_prompt_caching


def test_realistic_anthropic_agent_turn() -> None:
    """A cached, tool-carrying, multimodal turn — everything at once."""
    payload = {
        "model": "claude-opus-4-8",
        "max_tokens": 8192,
        "tools": [{"name": "read_file"}],
        "system": [
            {
                "type": "text",
                "text": "You are a code reviewer.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "review this"},
                    {"type": "image", "source": {"data": "..."}},
                ],
            }
        ],
    }
    shape = inspect_payload(payload)
    assert shape.has_tools
    assert shape.has_images
    assert shape.max_output_tokens == 8192
    assert shape.uses_prompt_caching
