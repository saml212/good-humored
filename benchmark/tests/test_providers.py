"""Unit tests for provider dispatch and env-file parsing — pure logic
only. No subprocess/network calls happen here: the CLI/API factories
build a `complete` closure without invoking it, and the openai-compat
factory tests point at a throwaway fixture dir instead of real secrets.
Run: python3 -m unittest discover -s benchmark/tests -t .
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from benchmark import providers
from benchmark.providers import (CLAUDE_MODELS, CODEX_MODELS, get_provider,
                                 make_codex_cli, make_openai_compat,
                                 parse_env_file)


class TestGetProviderDispatch(unittest.TestCase):
    def test_bare_alias_defaults_to_claude(self):
        complete = get_provider("haiku")
        self.assertEqual(complete.__name__, "claude_cli_haiku")

    def test_explicit_claude_prefix(self):
        complete = get_provider("claude:sonnet")
        self.assertEqual(complete.__name__, "claude_cli_sonnet")

    def test_fable_alias(self):
        complete = get_provider("claude:fable")
        self.assertEqual(complete.__name__, "claude_cli_fable")

    def test_codex_prefix(self):
        complete = get_provider("codex:mini")
        self.assertEqual(complete.__name__, "codex_cli_mini")

    def test_unknown_prefix_raises(self):
        with self.assertRaises(ValueError):
            get_provider("nope:whatever")


class TestModelAliasTables(unittest.TestCase):
    def test_fable_alias_present(self):
        self.assertEqual(CLAUDE_MODELS["fable"], "claude-fable-5")

    def test_codex_known_alias(self):
        self.assertEqual(CODEX_MODELS["sol"], "gpt-5.6-sol")

    def test_codex_alias_passthrough(self):
        # An alias not in CODEX_MODELS is passed through unchanged — same
        # contract as CLAUDE_MODELS.get(model, model).
        complete = make_codex_cli("gpt-5.6-terra")
        self.assertEqual(complete.__name__, "codex_cli_gpt-5.6-terra")


class TestParseEnvFile(unittest.TestCase):
    def test_parses_keys_ignores_comments_and_blank_lines(self):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".env", delete=False) as f:
            f.write("# comment\n\nFOO_API_KEY=abc123\n"
                    "BASE_URL='https://example.com'\n")
            path = f.name
        try:
            env = parse_env_file(path)
        finally:
            os.remove(path)
        self.assertEqual(env, {"FOO_API_KEY": "abc123",
                               "BASE_URL": "https://example.com"})

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            parse_env_file("/nonexistent/path/does-not-exist.env")


class TestOpenAICompatFactory(unittest.TestCase):
    """make_openai_compat reads secrets at factory time, so these tests
    point _FLEET_SECRETS_DIR at a throwaway fixture dir — never the real
    ~/agents/.fleet-secrets — and never call complete(), so no network
    traffic happens."""

    def setUp(self):
        self._orig_dir = providers._FLEET_SECRETS_DIR
        self._tmp = tempfile.mkdtemp(prefix="gh-fake-secrets-")
        providers._FLEET_SECRETS_DIR = self._tmp

    def tearDown(self):
        providers._FLEET_SECRETS_DIR = self._orig_dir

    def _write(self, name, content):
        with open(os.path.join(self._tmp, name), "w") as f:
            f.write(content)

    def test_missing_env_file_is_actionable(self):
        with self.assertRaises(RuntimeError) as cm:
            make_openai_compat("deepseek")
        self.assertIn("deepseek.env", str(cm.exception))

    def test_empty_key_raises(self):
        self._write("deepseek.env", "DEEPSEEK_API_KEY=\n")
        with self.assertRaises(RuntimeError):
            make_openai_compat("deepseek")

    def test_base_url_from_var(self):
        self._write("qwen.env", "QWEN_API_KEY=fake-not-real\n"
                                "QWEN_BASE_URL=https://example.com/v1\n")
        complete = make_openai_compat("qwen")
        self.assertEqual(complete.__name__, "api_qwen")

    def test_unknown_provider_raises_value_error(self):
        with self.assertRaises(ValueError):
            make_openai_compat("not-a-provider")


class _FakeSecretsMixin:
    """Shared setUp/tearDown pointing _FLEET_SECRETS_DIR at a throwaway
    fixture dir with a fake (not-real) deepseek key, so tests can build
    real `complete` closures without ever touching real secrets."""

    def setUp(self):
        self._orig_dir = providers._FLEET_SECRETS_DIR
        self._tmp = tempfile.mkdtemp(prefix="gh-fake-secrets-")
        providers._FLEET_SECRETS_DIR = self._tmp
        with open(os.path.join(self._tmp, "deepseek.env"), "w") as f:
            f.write("DEEPSEEK_API_KEY=fake-not-real\n")

    def tearDown(self):
        providers._FLEET_SECRETS_DIR = self._orig_dir


class TestOpenAICompatRequestBody(_FakeSecretsMixin, unittest.TestCase):
    """Request-body construction for the new `temperature` kwarg.
    HTTP is mocked throughout (urlopen never touches the network) —
    only the JSON body handed to it is inspected."""

    def _fake_response(self, content="a joke"):
        payload = json.dumps(
            {"choices": [{"message": {"content": content}}]}).encode("utf-8")
        cm = mock.MagicMock()
        cm.__enter__.return_value.read.return_value = payload
        cm.__exit__.return_value = False
        return cm

    def _sent_body(self, complete_fn):
        with mock.patch.object(providers.urllib.request, "urlopen",
                               return_value=self._fake_response()) as m:
            complete_fn("prompt text")
            req = m.call_args[0][0]
        return json.loads(req.data.decode("utf-8"))

    def test_default_temperature_matches_historical_body(self):
        # No temperature kwarg passed at all — must reproduce the
        # pre-existing hardcoded body (1.0, same key order) exactly.
        complete = make_openai_compat("deepseek")
        body = self._sent_body(complete)
        self.assertEqual(body["temperature"], 1.0)
        self.assertEqual(list(body.keys()),
                         ["model", "messages", "temperature", "max_tokens"])

    def test_explicit_temperature_forwarded(self):
        complete = make_openai_compat("deepseek", temperature=1.2)
        body = self._sent_body(complete)
        self.assertEqual(body["temperature"], 1.2)

    def test_none_omits_temperature_field(self):
        complete = make_openai_compat("deepseek", temperature=None)
        body = self._sent_body(complete)
        self.assertNotIn("temperature", body)

    def test_zero_temperature_is_not_treated_as_omit(self):
        # 0.0 is falsy in Python but a legitimate (fully greedy) setting —
        # must NOT be dropped the way None is.
        complete = make_openai_compat("deepseek", temperature=0.0)
        body = self._sent_body(complete)
        self.assertIn("temperature", body)
        self.assertEqual(body["temperature"], 0.0)


class TestGetProviderTemperatureDispatch(_FakeSecretsMixin, unittest.TestCase):
    """get_provider's confound guard: temperature is a hard error for
    CLI-backed specs (claude:/codex:/bare), and passes through cleanly
    for api: specs. No network calls — factories are only constructed,
    never invoked, except where the base class needs a real key file."""

    def test_temperature_rejected_for_claude_prefix(self):
        with self.assertRaises(ValueError):
            get_provider("claude:sonnet", temperature=0.7)

    def test_temperature_rejected_for_codex_prefix(self):
        with self.assertRaises(ValueError):
            get_provider("codex:mini", temperature=0.7)

    def test_temperature_rejected_for_bare_alias(self):
        # bare alias defaults to "claude" — same CLI confound applies.
        with self.assertRaises(ValueError):
            get_provider("haiku", temperature=0.7)

    def test_temperature_rejected_even_at_zero(self):
        # 0.0 is falsy but still a real override request — must not slip
        # past an `if temperature:` truthiness check.
        with self.assertRaises(ValueError):
            get_provider("claude:sonnet", temperature=0.0)

    def test_temperature_accepted_for_api_prefix(self):
        complete = get_provider("api:deepseek", temperature=0.2)
        self.assertEqual(complete.__name__, "api_deepseek")

    def test_no_temperature_kwarg_is_byte_identical_call(self):
        # Exact previous call shape (no temperature at all) still works.
        complete = get_provider("api:deepseek")
        self.assertEqual(complete.__name__, "api_deepseek")


if __name__ == "__main__":
    unittest.main()
