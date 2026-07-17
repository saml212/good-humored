"""End-to-end integration smoke tests for env/ against real training
frameworks (as opposed to env/tests/, which unit-tests the reward stack
and environments in isolation with no trainer involved).

Nothing in this subpackage is imported by env/__init__.py -- it pulls in
heavy, optional third-party dependencies (torch, transformers, trl) that
the rest of the package does not require. Run its scripts directly, e.g.:

    python3 -m env.integration.trl_grpo_smoke
"""
