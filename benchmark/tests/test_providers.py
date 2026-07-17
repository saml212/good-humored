"""Unit tests for provider dispatch and env-file parsing — pure logic
only. No subprocess/network calls happen here: the CLI/API factories
build a `complete` closure without invoking it, and the openai-compat
factory tests point at a throwaway fixture dir instead of real secrets.
Run: python3 -m unittest discover -s benchmark/tests -t .
"""

import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
