# Repository Guidelines & Preferences

## User Preferences
- Don't append an unsolicited end-of-turn recap/summary after simple Q&A or quick answers. Only summarize when something durable actually happened (files changed, a multi-step task completed) — and keep it to what changed and what's next.
- Don't run validation steps unless explicitly told to.
- Don't add comments to code, unless explicitly told to.

## Autonomous Advisor Mode Protocol
When running on Gemini Flash (or another fast model tier):
- Act as the **Executor**. Autonomously consult a `pro` subagent as an **Advisor** (via `invoke_subagent` with `Model: "pro"`) without asking the user.
- **Orientation first:** Find relevant files and context before calling the advisor.
- **Before substantive work:** Call the advisor for multi-step tasks, architectural designs, or complex refactors.
- **When stuck:** Call the advisor if an error or test failure persists after 1 attempt.
- **Before completion:** Call the advisor on long tasks to verify completeness and edge cases.
- **Mechanical work:** Keep all file edits, terminal runs, and testing on the Executor.
