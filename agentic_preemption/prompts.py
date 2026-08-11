#!/usr/bin/env python3
"""Prompt pieces for the preemption load.

Recovered from the former sibling `agentic_coding/agent_session.py` (which was
deleted) so `agentic_preemption/` is now self-contained. `deadlock_run.py` imports
`SYSTEM_PROMPT` + `build_first_user`; `prepare_task.py` uses `TASK_INSTRUCTION`.
"""

SYSTEM_PROMPT = (
    "You are a meticulous senior software engineer acting as an autonomous coding "
    "agent. Respond with the next concrete code edit and a one-line rationale."
)

# The coding task the agent is handed (written into generated/agent_task.json).
TASK_INSTRUCTION = (
    "You are a senior Python engineer working inside a repository. The current "
    "contents of `metrics_pipeline.py` are shown below. Refactor it for clarity "
    "and correctness: remove duplication, add type hints and docstrings, fix any "
    "bugs you spot, and add unit tests. Work iteratively -- propose the next "
    "concrete edit, explain why, and show the changed code. Keep going until the "
    "module is clean and well-tested."
)


def build_first_user(marker: str, instruction: str, seed: str) -> str:
    """Assemble one request's user message: a unique marker, the task, and the
    file to work on (fenced as python)."""
    return f"{marker}\n{instruction}\n\n```python\n{seed}\n```"
