import importlib.util
import json
import pathlib
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "openrouter_benchmark_atlascloud",
    ROOT / "scripts" / "openrouter_benchmark.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return b'{"choices":[{"message":{"content":"ok"}}]}'


class AtlasCloudProviderTests(unittest.TestCase):
    def test_provider_aliases_normalize(self) -> None:
        for alias in ("atlas", "atlas-cloud", "atlas_cloud", "atlascloud"):
            with self.subTest(alias=alias):
                self.assertEqual(
                    MODULE.normalize_model_provider(alias, field_name="provider"),
                    "atlascloud",
                )

    @mock.patch.object(MODULE.urllib.request, "urlopen", return_value=_Response())
    def test_client_sends_openai_compatible_request(self, urlopen) -> None:
        with mock.patch.dict(
            MODULE.os.environ,
            {
                "ATLASCLOUD_BASE_URL": "https://api.atlascloud.ai/v1",
                "ATLASCLOUD_USER_AGENT": "BullshitBench/1.0",
            },
        ):
            client = MODULE.AtlasCloudClient("test-key", timeout_seconds=30)
            result = client.chat(
                model="qwen/qwen3.8-max",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.2,
                max_tokens=64,
                retries=1,
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://api.atlascloud.ai/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(request.get_header("User-agent"), "BullshitBench/1.0")
        self.assertEqual(payload["model"], "qwen/qwen3.8-max")
        self.assertEqual(payload["messages"][0]["content"], "hello")
        self.assertEqual(result["choices"][0]["message"]["content"], "ok")


if __name__ == "__main__":
    unittest.main()
