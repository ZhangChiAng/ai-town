# Repository instructions

Before planning or implementing work in this repository, read
[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) and treat it as the canonical product
baseline.

- Preserve its experimental hypothesis, information boundaries, and explicit
  non-goals.
- Treat the current Markdown files in `data/` as reference material until their
  project data format is defined; do not convert or modify them unless the user
  explicitly asks.
- Do not silently resolve undecided product behavior or add psychological
  mechanisms, numeric state, automatic recall, or automatic simulation.
- Clarify major product and behavioral decisions with the user before
  implementing them.
- Never infer, increment, or otherwise change the scene contract version on
  behalf of the user. Before any change that may require a different scene
  version, ask the user to specify the exact target version and use only that
  confirmed value. The current user-confirmed version is
  `ai-town.scene/1.0`.

# Coding Style

Follow Google Python Style for organization, naming, imports, docstrings, and
readability. Use Ruff as the formatting and linting tool.

Add short inline comments inside functions to explain non-obvious
implementation logic. Explain intent and control flow, not what an individual
line already says.
