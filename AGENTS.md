# AGENTS.md

## Project Skill Instructions

Use the following local skill for this project:

- `/Users/najongseong/git_repository/skills-registry/project_management/chatbot_project_manager/SKILL.md`
- `/Users/najongseong/git_repository/skills-registry/project_management/project_logger/SKILL.md`

When working in this repository, read and follow those skills before performing project work. In particular:

1. Maintain `project_progress.md` in the project root.
2. Log each user request with a timestamp before major work.
3. Log important actions, artifact updates, decisions, and outcomes.
4. When the user asks for a wrap-up or session summary, update the Daily Wrap-ups and Next Action Items sections.

## Project-Specific Notes

- The main Streamlit chatbot page is `pages/Telly.py`.
- Agent orchestration and chaining logic lives primarily in `core/chat_flow.py`.
- Agent construction is in `core/agent.py`.
- Prompt templates are in `core/prompt.py`.
- Regression scenarios are tracked in `test_scenario.py`.
- Historical bug notes are in `WORK_HISTORY.md`; ongoing request/action logs should go in `project_progress.md`.
