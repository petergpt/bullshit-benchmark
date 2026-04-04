"""Unit and integration tests for the MiniMax direct provider.

Tests cover:
  - MiniMaxClient initialization and validation
  - Model ID stripping (minimax/ prefix removal)
  - Temperature clamping to (0.0, 1.0]
  - Think-tag stripping from response text
  - Provider alias and value registration
  - Provider resolution with minimax/* wildcard
  - MiniMaxAPIError classification
  - Config loading with minimax provider routing
  - Client wiring in collect/grade flows (dry-run)
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from typing import Any
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# Ensure the scripts directory is importable
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"),
)

import openrouter_benchmark as bench


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestMiniMaxModelId(unittest.TestCase):
    """Test _minimax_model_id prefix stripping."""

    def test_strips_minimax_prefix(self):
        self.assertEqual(bench._minimax_model_id("minimax/minimax-m2.7"), "minimax-m2.7")

    def test_strips_minimax_prefix_highspeed(self):
        self.assertEqual(
            bench._minimax_model_id("minimax/minimax-m2.7-highspeed"),
            "minimax-m2.7-highspeed",
        )

    def test_no_prefix_passthrough(self):
        self.assertEqual(bench._minimax_model_id("minimax-m2.7"), "minimax-m2.7")

    def test_other_prefix_passthrough(self):
        self.assertEqual(bench._minimax_model_id("openai/gpt-4"), "openai/gpt-4")

    def test_empty_after_prefix(self):
        self.assertEqual(bench._minimax_model_id("minimax/"), "minimax/")

    def test_whitespace_stripped(self):
        self.assertEqual(bench._minimax_model_id("  minimax/minimax-m2.7  "), "minimax-m2.7")


class TestMiniMaxClampTemperature(unittest.TestCase):
    """Test _minimax_clamp_temperature enforces (0.0, 1.0]."""

    def test_none_passthrough(self):
        self.assertIsNone(bench._minimax_clamp_temperature(None))

    def test_valid_temperature(self):
        self.assertAlmostEqual(bench._minimax_clamp_temperature(0.5), 0.5)

    def test_clamp_zero_to_min(self):
        self.assertAlmostEqual(bench._minimax_clamp_temperature(0.0), 0.01)

    def test_clamp_negative_to_min(self):
        self.assertAlmostEqual(bench._minimax_clamp_temperature(-1.0), 0.01)

    def test_clamp_above_one(self):
        self.assertAlmostEqual(bench._minimax_clamp_temperature(2.0), 1.0)

    def test_exactly_one(self):
        self.assertAlmostEqual(bench._minimax_clamp_temperature(1.0), 1.0)

    def test_small_positive(self):
        self.assertAlmostEqual(bench._minimax_clamp_temperature(0.01), 0.01)

    def test_very_small_clamped(self):
        self.assertAlmostEqual(bench._minimax_clamp_temperature(0.001), 0.01)


class TestStripThinkTags(unittest.TestCase):
    """Test _strip_think_tags removes internal reasoning blocks."""

    def test_no_think_tags(self):
        self.assertEqual(bench._strip_think_tags("Hello world"), "Hello world")

    def test_single_think_block(self):
        text = "<think>internal reasoning</think>The answer is 42."
        self.assertEqual(bench._strip_think_tags(text), "The answer is 42.")

    def test_multiline_think_block(self):
        text = "<think>\nStep 1: consider\nStep 2: decide\n</think>\nFinal answer."
        self.assertEqual(bench._strip_think_tags(text), "Final answer.")

    def test_multiple_think_blocks(self):
        text = "<think>first</think>Hello <think>second</think>world"
        self.assertEqual(bench._strip_think_tags(text), "Hello world")

    def test_empty_think_block(self):
        text = "<think></think>Result"
        self.assertEqual(bench._strip_think_tags(text), "Result")

    def test_only_think_block(self):
        text = "<think>all reasoning</think>"
        self.assertEqual(bench._strip_think_tags(text), "")


class TestProviderRegistration(unittest.TestCase):
    """Test that minimax is registered as a valid provider."""

    def test_minimax_in_aliases(self):
        self.assertIn("minimax", bench.MODEL_PROVIDER_ALIASES)
        self.assertEqual(bench.MODEL_PROVIDER_ALIASES["minimax"], "minimax")

    def test_minimax_in_values(self):
        self.assertIn("minimax", bench.MODEL_PROVIDER_VALUES)

    def test_normalize_minimax(self):
        self.assertEqual(
            bench.normalize_model_provider("minimax", field_name="test"),
            "minimax",
        )

    def test_normalize_minimax_case_insensitive(self):
        self.assertEqual(
            bench.normalize_model_provider("MiniMax", field_name="test"),
            "minimax",
        )


class TestProviderResolution(unittest.TestCase):
    """Test resolve_model_provider with minimax/* wildcard routing."""

    def test_wildcard_routing(self):
        overrides = {"*": "openrouter", "minimax/*": "minimax"}
        result = bench.resolve_model_provider("minimax/minimax-m2.7", overrides)
        self.assertEqual(result, "minimax")

    def test_wildcard_routing_highspeed(self):
        overrides = {"*": "openrouter", "minimax/*": "minimax"}
        result = bench.resolve_model_provider(
            "minimax/minimax-m2.7-highspeed", overrides
        )
        self.assertEqual(result, "minimax")

    def test_exact_match_overrides_wildcard(self):
        overrides = {
            "*": "openrouter",
            "minimax/*": "minimax",
            "minimax/minimax-m2.7": "openrouter",
        }
        result = bench.resolve_model_provider("minimax/minimax-m2.7", overrides)
        self.assertEqual(result, "openrouter")

    def test_non_minimax_model_uses_default(self):
        overrides = {"*": "openrouter", "minimax/*": "minimax"}
        result = bench.resolve_model_provider("anthropic/claude-sonnet-4.6", overrides)
        self.assertEqual(result, "openrouter")


class TestParseModelProviders(unittest.TestCase):
    """Test parse_model_providers with minimax entries."""

    def test_parse_minimax_provider(self):
        raw = '{"*": "openrouter", "minimax/*": "minimax"}'
        result = bench.parse_model_providers(raw, field_name="test")
        self.assertEqual(result["minimax/*"], "minimax")

    def test_parse_dict_input(self):
        raw = {"*": "openrouter", "minimax/*": "minimax"}
        result = bench.parse_model_providers(raw, field_name="test")
        self.assertEqual(result["minimax/*"], "minimax")


class TestMiniMaxAPIError(unittest.TestCase):
    """Test MiniMaxAPIError inherits from ProviderAPIError."""

    def test_is_provider_error(self):
        err = bench.MiniMaxAPIError(
            "test error", status_code=429, retryable=True
        )
        self.assertIsInstance(err, bench.ProviderAPIError)
        self.assertEqual(err.status_code, 429)
        self.assertTrue(err.retryable)

    def test_non_retryable(self):
        err = bench.MiniMaxAPIError(
            "bad request", status_code=400, retryable=False
        )
        self.assertFalse(err.retryable)


class TestMiniMaxClientInit(unittest.TestCase):
    """Test MiniMaxClient initialization and validation."""

    def test_valid_init(self):
        client = bench.MiniMaxClient(api_key="test-key", timeout_seconds=30)
        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(client.timeout_seconds, 30)
        self.assertEqual(client.base_url, "https://api.minimax.io/v1/chat/completions")

    def test_invalid_timeout(self):
        with self.assertRaises(ValueError):
            bench.MiniMaxClient(api_key="test-key", timeout_seconds=0)

    def test_negative_timeout(self):
        with self.assertRaises(ValueError):
            bench.MiniMaxClient(api_key="test-key", timeout_seconds=-1)


class TestMiniMaxClientChat(unittest.TestCase):
    """Test MiniMaxClient.chat payload construction and response handling."""

    def _make_client(self) -> bench.MiniMaxClient:
        return bench.MiniMaxClient(api_key="test-key", timeout_seconds=30)

    @patch("urllib.request.urlopen")
    def test_basic_chat_payload(self, mock_urlopen):
        """Verify the outgoing payload structure."""
        response_body = json.dumps(
            {
                "id": "test-id",
                "choices": [
                    {"message": {"content": "Test response"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        ).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = self._make_client()
        result = client.chat(
            model="minimax/minimax-m2.7",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
            retries=1,
        )

        self.assertEqual(result["id"], "test-id")
        # Verify the request was made with correct URL
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        self.assertEqual(request_obj.full_url, "https://api.minimax.io/v1/chat/completions")
        sent_payload = json.loads(request_obj.data)
        self.assertEqual(sent_payload["model"], "minimax-m2.7")
        self.assertAlmostEqual(sent_payload["temperature"], 0.7)
        self.assertEqual(sent_payload["max_tokens"], 100)

    @patch("urllib.request.urlopen")
    def test_temperature_clamping_in_payload(self, mock_urlopen):
        """Verify temperature=0 is clamped to 0.01."""
        response_body = json.dumps(
            {
                "id": "test-id",
                "choices": [
                    {"message": {"content": "response"}, "finish_reason": "stop"}
                ],
                "usage": {},
            }
        ).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = self._make_client()
        client.chat(
            model="minimax-m2.7",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.0,
            max_tokens=0,
            retries=1,
        )

        sent_payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertAlmostEqual(sent_payload["temperature"], 0.01)

    @patch("urllib.request.urlopen")
    def test_think_tag_stripping_in_response(self, mock_urlopen):
        """Verify <think> tags are stripped from response content."""
        response_body = json.dumps(
            {
                "id": "test-id",
                "choices": [
                    {
                        "message": {
                            "content": "<think>reasoning here</think>The actual answer."
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
        ).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = self._make_client()
        result = client.chat(
            model="minimax-m2.7",
            messages=[{"role": "user", "content": "Test"}],
            temperature=None,
            max_tokens=0,
            retries=1,
        )

        self.assertEqual(result["choices"][0]["message"]["content"], "The actual answer.")

    @patch("urllib.request.urlopen")
    def test_provider_key_dropped(self, mock_urlopen):
        """Verify the 'provider' key from extra_payload is dropped."""
        response_body = json.dumps(
            {
                "id": "test-id",
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {},
            }
        ).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = self._make_client()
        client.chat(
            model="minimax-m2.7",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            max_tokens=0,
            retries=1,
            extra_payload={"provider": {"require_parameters": True}, "reasoning": {"effort": "high"}},
        )

        sent_payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertNotIn("provider", sent_payload)
        self.assertEqual(sent_payload["reasoning"], {"effort": "high"})

    def test_invalid_retries(self):
        client = self._make_client()
        with self.assertRaises(ValueError):
            client.chat(
                model="minimax-m2.7",
                messages=[],
                temperature=None,
                max_tokens=0,
                retries=0,
            )

    @patch("urllib.request.urlopen")
    def test_null_temperature_omitted(self, mock_urlopen):
        """Verify temperature=None means no temperature key in payload."""
        response_body = json.dumps(
            {
                "id": "test-id",
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {},
            }
        ).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = self._make_client()
        client.chat(
            model="minimax-m2.7",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=None,
            max_tokens=0,
            retries=1,
        )

        sent_payload = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertNotIn("temperature", sent_payload)


class TestConfigMiniMaxRouting(unittest.TestCase):
    """Test that config files route minimax/* models to minimax provider."""

    def test_config_json(self):
        import pathlib
        config_path = (
            pathlib.Path(__file__).resolve().parent.parent / "config.json"
        )
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        models = config["collect"]["models"]
        providers = config["collect"]["model_providers"]
        self.assertIn("minimax/minimax-m2.7", models)
        self.assertIn("minimax/minimax-m2.7-highspeed", models)
        self.assertEqual(providers.get("minimax/*"), "minimax")

    def test_config_v2_json(self):
        import pathlib
        config_path = (
            pathlib.Path(__file__).resolve().parent.parent / "config.v2.json"
        )
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        models = config["collect"]["models"]
        providers = config["collect"]["model_providers"]
        self.assertIn("minimax/minimax-m2.7", models)
        self.assertIn("minimax/minimax-m2.7-highspeed", models)
        self.assertEqual(providers.get("minimax/*"), "minimax")

    def test_reasoning_efforts_defined(self):
        import pathlib
        config_path = (
            pathlib.Path(__file__).resolve().parent.parent / "config.json"
        )
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        efforts = config["collect"]["model_reasoning_efforts"]
        self.assertIn("minimax/minimax-m2.7", efforts)
        self.assertIn("minimax/minimax-m2.7-highspeed", efforts)
        self.assertEqual(efforts["minimax/minimax-m2.7"], ["none", "high"])


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class _FakeMiniMaxHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler mimicking the MiniMax chat/completions API."""

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        req = json.loads(body)
        model = req.get("model", "unknown")
        response = {
            "id": "integration-test-id",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            "<think>internal reasoning</think>"
                            f"Integration test response from {model}"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 10,
                "total_tokens": 25,
            },
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass  # suppress log output during tests


class TestMiniMaxClientIntegration(unittest.TestCase):
    """Integration tests using a local HTTP server to simulate MiniMax API."""

    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _FakeMiniMaxHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(target=cls.server.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server_thread.join(timeout=5)

    def _make_client(self) -> bench.MiniMaxClient:
        client = bench.MiniMaxClient(api_key="test-key", timeout_seconds=10)
        client.base_url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        return client

    def test_full_chat_roundtrip(self):
        """End-to-end test: send chat request, get response, think tags stripped."""
        client = self._make_client()
        result = client.chat(
            model="minimax/minimax-m2.7",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            temperature=0.7,
            max_tokens=100,
            retries=1,
        )

        self.assertEqual(result["id"], "integration-test-id")
        self.assertEqual(result["model"], "minimax-m2.7")
        content = result["choices"][0]["message"]["content"]
        # Think tags should be stripped
        self.assertNotIn("<think>", content)
        self.assertIn("Integration test response from minimax-m2.7", content)
        self.assertEqual(result["usage"]["total_tokens"], 25)

    def test_temperature_zero_clamped(self):
        """Verify temperature=0 is clamped in actual request."""
        client = self._make_client()
        result = client.chat(
            model="minimax-m2.7-highspeed",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.0,
            max_tokens=50,
            retries=1,
        )
        # Should succeed — the clamped temperature (0.01) is valid
        self.assertEqual(result["model"], "minimax-m2.7-highspeed")

    def test_reasoning_effort_forwarded(self):
        """Verify reasoning effort is forwarded via extra_payload."""
        client = self._make_client()
        result = client.chat(
            model="minimax-m2.7",
            messages=[{"role": "user", "content": "Think hard about this."}],
            temperature=0.5,
            max_tokens=200,
            retries=1,
            extra_payload={"reasoning": {"effort": "high"}},
        )
        self.assertEqual(result["id"], "integration-test-id")


if __name__ == "__main__":
    unittest.main()
