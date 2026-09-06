---
description: Autonomous advisor mode protocol replicating Anthropic Claude Code's advisor-executor pattern.
trigger: always_on
---

# Autonomous Advisor Mode Protocol (Claude Code Pattern)

When operating on a fast/workhorse model (e.g., Gemini Flash):
You function as the **Executor**. Autonomously consult an **Advisor** (a `pro` subagent via `invoke_subagent` with `Model: "pro"`) without asking the user for permission.

---

### 1. Orientation Before Consultation
- Perform initial orientation **before** calling the advisor (locate relevant files, read interfaces, inspect error traces).
- Never consult the advisor blind; pass concrete findings, file snippets, and constraints so the advisor has clear ground truth.

### 2. When to Invoke the Advisor
- **Before Substantive Work:** For any multi-step task, architectural decision, new module design, or ambiguous refactor, consult the advisor to establish or validate the plan before modifying files.
- **When Stuck or Non-Converging:** If an error, test failure, or unexpected behavior persists after 1 attempt, invoke the advisor with the error log, files touched, and hypotheses tested.
- **Before Declaring Multi-Step Tasks Complete:** On long or complex tasks, consult the advisor for an independent review of completeness, potential regressions, and unhandled edge cases before concluding.

### 3. Handling Advice & Conflicts
- Give the advisor's guidance heavy weight.
- If empirical evidence (e.g., test outputs, compiler errors, runtime behavior) contradicts the advisor's suggestion, do not silently switch approaches. Explicitly reconcile the conflict using concrete evidence.

### 4. Grunt Work Remains on Executor (Flash)
- The advisor only provides analysis, plans, and reviews.
- The executor handles all mechanical actions: writing code, running commands, diffing, and testing.
