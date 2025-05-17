"""
Multi-Agent System for SWE-bench Lite Bug Fixing
===============================================

Updated to import helper *tools* from the **tools/** package.
Directory structure assumed:

```
project/
├── agents.py            # <— this file
└── tools/
    ├── __init__.py      # can be empty
    ├── search_code_tool.py
    ├── run_tests_tool.py
    ├── apply_patch_tool.py
    └── git_reset_tool.py
```

Usage
-----
```bash
python agents.py task.json
```
where *task.json* is a single JSON row from SWE‑bench Lite.
"""

#from __future__ import annotations
import logging
from collections.abc import AsyncGenerator
import pathlib
import sys
from typing import Any, Dict
# ---------------------------------------------------------------------------
from .tools import (search_code, run_tests, apply_patch, git_reset,get_swe_lite_instance)  # type: ignore
# ---------------------------------------------------------------------------
# Google ADK imports
# ---------------------------------------------------------------------------
from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents.invocation_context import (     # InvocationContext lives here
    InvocationContext,
)
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext
from google.adk.events import Event, EventActions
from google.genai import types
# ---------------------------------------------------------------------------
# Custom deterministic agent that just runs pytest
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

model= LiteLlm(
    model            = "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    max_tokens       = 1024,
    temperature      = 1,
    api_base         = None,         
)

tester_agent = LlmAgent(
    name="tester",
    model=model,
    tools=[run_tests],
    instruction="""
As the Tester agent, you have these values in state:
  • python_exe   (path to the per‐repo venv python executable)
  • repo_path    (path to the checked‐out source)
  • test_paths   (space‐separated test file(s)/dir(s) or empty string)

Your job:
1) Retrieve `python_exe`, `repo_path` and `test_paths` from state.
2) Call the RunTests tool exactly once, passing in:
     python_exe=<python_exe>,
     repo_path=<repo_path>,
     paths=<test_paths>
3) Return the tool’s output, a dict containing:
     exit_code, passed, failed, and log.

Do not run pytest yourself or call any other tools.
""",
    output_key="test_result",
)

# ---------------------------------------------------------------------------
# Agents definitions
# ---------------------------------------------------------------------------


locator_agent = LlmAgent(
    name="locator",
    model=model,
    tools=[search_code],
    instruction="""You are the Locator agent. Your input is a JSON object with these fields:
  • instance_id        – the unique ID of the bug instance  
  • repo_path          – filesystem path to the checked-out repository  
  • test_patch         – the test diff text that failed  
  • problem_statement  – the human-readable bug description  

Do **not** call any tool except **search_code(pattern: str, path: List[str])**.  
Do **not** attempt to call any “get_state” or similar helper.  
Steps:
1. Read `problem_statement` from the input.  
2. Derive a pattern and a list of file paths (strings, relative to `repo_path`) where that pattern might occur.  
3. Call **search_code** exactly once
Output only searcg_code's output, a JSON array of {file, line, snippet}""",
    output_key="locator_output",
)

patcher_agent = LlmAgent(
    name="patcher",
    model=model,
    tools=[apply_patch],
    instruction="""Based on problem_statement(in the problem_statement field of instance in state) and the locator_output in state, generate a single line patch to fix the bug. 
    • Use the apply_patch tool to apply the patch to the file and line you found in locator_output. Apply patch takes a patch string, a file path string and a line number integer as input.
    • Only make calls to the apply_patch tool that is on your toolset already.
    If it fails, revise on next turn. Reply ONLY with apply_patch
JSON output.""",
)

def exit_loop(tool_context: ToolContext):
  """Call this function ONLY when the critic agent decides that bug fix is successful, signaling the iterative process should end."""
  print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
  tool_context.actions.escalate = True
  # Return empty dict as tools should typically return JSON-serializable output
  return {}


critic_agent = LlmAgent(
    name="critic",
    model=model,
    tools=[git_reset, exit_loop],
    instruction="""You are the Critic agent. You have been given:

  • `locator_output`, a JSON array of {file, line, snippet} dicts,
    containing all candidate locations in the code.

Do not call any `get_state` or similar tool.  
Simply examine the supplied `locator_output`, pick the best single location to patch,  
and then call the ApplyPatch tool with exactly these arguments:

  • file_path: the value of locator_output[N].file  
  • line_number: the value of locator_output[N].line  
  • patch_text: the new code text you want to insert  

Return the diff dict that ApplyPatch returns..""",
    output_key="critic_decision",
)



patch_loop = LoopAgent(
    name="patch_loop",
    sub_agents=[patcher_agent, tester_agent, critic_agent],          
    max_iterations=4,
)

root_agent = SequentialAgent(
    name="root",
    sub_agents=[locator_agent, patch_loop],
)

# ---------------------------------------------------------------------------
# Runner helper
# ---------------------------------------------------------------------------

def run_bug_fix(task: Dict[str, Any]):
    """Run the root agent on one SWE‑bench row dict."""

    runner = Runner(
        root_agent,
        session_service=InMemorySessionService(app_name="swe_bench_session"),
    )
    return runner.run(input=task)


if __name__ == "__main__":
    import json

    if len(sys.argv) != 2:
        print("Usage: python agents.py task.json", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as fh:
        task_row = json.load(fh)
    print(run_bug_fix(task_row))
