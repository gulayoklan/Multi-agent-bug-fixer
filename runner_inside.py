#!/usr/bin/env python3
import sys, os, json
from datasets import load_dataset
from agent_pkg.tools import prepare_repo, get_swe_lite_instance
from agent_pkg.agent import root_agent
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from dotenv import load_dotenv
load_dotenv() 
# 1) Read the instance_id
iid = sys.argv[1]
print(f"▶ instance_id = {iid}", flush=True)

# 2) Load the row (test split)
print("▶ loading dataset…", flush=True)
row=get_swe_lite_instance(iid)

# 3) Clone & checkout (shallow)
print(f"▶ cloning {row['repo']}@{row['base_commit']} …", flush=True)
info= prepare_repo(row["repo"], row["base_commit"], workspace_root="/tmp")
repo_path=info["repo_path"]
python_exe=info["python_exe"]
print(f"▶ repo ready at {repo_path}", flush=True)

# 4) Build the ADK Runner
session_service = InMemorySessionService()
memory_service  = InMemoryMemoryService()

runner = Runner(
    app_name="bug-fixer",             # must match your agent.yaml `id:`
    agent=root_agent,
    session_service=session_service,
    memory_service=memory_service
)
print("▶ Runner created", flush=True)
# selected only instance_id, repo_path, test_patch, problem_statement to avoid contamination of agents by patch field
agent_input = {
    "instance_id": row["instance_id"],
    "repo_path":   repo_path,
    "test_patch":  row["test_patch"],
    "problem_statement": row["problem_statement"],
    "python_exe": python_exe,
}

# 5) Wrap the row as a user message
user_message = types.Content(
    role="user",
    parts=[types.Part(text=json.dumps(agent_input))]
)

# 6) Execute the agent
print("▶ running agent…", flush=True)
events = runner.run(
    user_id="user1",
    session_id=iid,
    new_message=user_message
)
session_service.create_session(
    app_name="bug-fixer",
    user_id="user1",
    session_id=iid,
)

for event in runner.run(
        user_id="user1",
        session_id=iid,
        new_message=user_message):
    if event.is_final_response():
        print(event.content.parts[0].text)
