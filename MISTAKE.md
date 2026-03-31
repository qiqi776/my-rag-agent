# Mistake Notebook

This is the Mistake Notebook, use the mistake-notebook skill to retrive more details.

Below is a list of mistakes I previously made and solved:

## Mistake - Reintroducing Removed Spec Sections

- Creation Date: 2026-03-31
- Last Update Date: 2026-03-31
- Project: /home/zz/workspace/projects/ai-agent
- Branch: main
- Commit: cda8bc9

### Problem:

- I re-added sections to `specs/minimal-modular-rag-project.md` that the user had intentionally removed.
- The removed content included spec metadata / change-log style sections that the user does not want in this project spec.

### Insights:

- Spec content and formatting are part of project requirements, not cosmetic defaults.
- If the user deletes sections from `specs/`, treat that deletion as an explicit preference unless they ask to restore it.
- When updating specs, preserve current structure and only change content required by the task.

### Solution:

Respect current spec structure.

- Check current `specs/` content before editing.
- Do not restore removed metadata or change-log sections unless the user explicitly asks.
- When in doubt, leave deleted spec sections deleted.
