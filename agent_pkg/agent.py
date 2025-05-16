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
"""
class TesterAgent(BaseAgent):
    test_paths: str | list[str] = "tests"        # <-- here
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str = "tester", test_paths: str | list[str] = "tests"):
        super().__init__(name=name, test_paths=test_paths)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        paths = self.test_paths
        logger.info("[%s] Starting pytest on %s", self.name, paths)
        
        result = run_tests(paths)

        summary = (
            "✅ All tests passed."
            if result["exit_code"] == 0
            else f"❌ {result['failed']} failed, {result['passed']} passed."
        )
        logger.info("[%s] %s", self.name, summary)

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part.from_text(summary)]),
            actions=EventActions(state_delta={"test_result": result}),
        )
"""
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
    instruction="""As the Tester agent, retrieve `repo_path` and `test_paths` from the shared state.  
Then call the `RunTests` tool exactly once, passing those two arguments.  
Finally, return the tool’s output—which is a dict containing `exit_code`, `passed`, `failed`, and `log`.  
Do not attempt to run pytest yourself or call any other tools.""",
    output_key="test_result",
)

# ---------------------------------------------------------------------------
# Agents definitions
# ---------------------------------------------------------------------------

fetcher_agent = LlmAgent(
   name="fetcher",
   model=model,
   tools=[get_swe_lite_instance],
   instruction="""Fetch the SWE-bench-lite instance from the dataset using user input instance_id. Return the output of get_swe_lite_instance tool.""",
   output_key="instance",
)

locator_agent = LlmAgent(
    name="locator",
    model=model,
    tools=[search_code],
    instruction="""Search for the code snippet by locating the source file & approximate line to modify inspecting 'problem_statement' of the instance.
• Only make calls to the search_code tool that is already in your toolset to locate the source.
• Inspect stacktrace (should start with traceback) if present.
• Analyse stack trace to generate a pattern and a list of paths in the repository to search for that pattern.
• Call search_code tool with the pattern and list of paths you found.
Return the output of search_code: JSON {file, line, snippet}.""",
    output_key="locator_output",
)

patcher_agent = LlmAgent(
    name="patcher",
    model=model,
    tools=[apply_patch],
    instruction="""Based on problem_statement(in the problem_statement field of instance in state) and the locator_output in state, generate a single line patch to fix the bug. 
    • Use the apply_patch tool to apply the patch to the file and line you found in locator_output. Apply patch takes a patch string, a file path string and a line number integer as input.
    • Only make calls to the apply_patch tool that is on your toolset already.
    If it fails, revise on next turn. Reply ONLY with ApplyPatch
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
    instruction="""Evaluate `test_result`.
If all tests pass: respond `success`, call exit_loop tool.
Else: analyse traceback, update `locator_output` or `test_paths`, respond `retry` reset the git environment using 'git_reset' tool. """,
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
