# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `src/jaxued/`:
- `environments/` for environment implementations (`maze/`, `tmaze/`, `gymnax/`)
- `wrappers/` for auto-reset/replay wrappers
- `level_sampler.py` and `traits/` for UED sampling and trait logic

Runnable reference implementations are in `examples/` (for example, `maze_dr.py`, `maze_plr.py`, `maze_paired.py`, `tmaze_dr.py`). Test files are in `tests/`. Experiment presets are
in `configs/*.yaml`. Documentation source is in `docs/` with MkDocs config in `mkdocs.yml`. Runtime outputs typically go to `checkpoints/` and `results/`.

## Build, Test, and Development Commands
- `./.venv/bin/python -m pip install -e ".[examples]"`: editable install with example dependencies.
- `./.venv/bin/python -m pytest -q`: quick test run.
- `WANDB_MODE=disabled ./.venv/bin/python -m pytest -v -s`: run tests without Weights & Biases logging side effects.
- `WANDB_MODE=disabled ./.venv/bin/python -m pytest tests/test_dr_equivalence.py -v -s`: run DR v1/v2 equivalence test.
- In sandboxed runs, JAX may log CUDA plugin initialization warnings and then fall back to CPU; this is expected and not a test failure by itself.
- `tox`: run tests across supported Python versions defined in `tox.ini`.
- `./.venv/bin/python examples/maze_plr.py` (or other scripts in `examples/`): run training entrypoints.
- `./.venv/bin/python -m compileall <files...>`: syntax check specific files.
- `mkdocs serve`: preview docs locally from `docs/`.

## Python / Test Environment (Important)
- Reuse the existing virtual environment at `./.venv`; do not recreate it.
- Do not run `python`, `python3`, `pip`, or `pytest` from PATH.
- Use explicit venv executables:
- `./.venv/bin/python`
- `./.venv/bin/python -m pip`
- `./.venv/bin/python -m pytest` (or `./.venv/bin/pytest`)
- If `pytest` is missing from the venv, install it with `./.venv/bin/python -m pip install pytest`.

## Coding Style & Naming Conventions
Use Python 3.9+ with 4-space indentation and PEP 8-compatible formatting. Follow existing repository patterns: `snake_case` for functions/variables, `PascalCase` for classes,
descriptive names over abbreviations, and type hints for public APIs where practical. Keep implementations readable and local; avoid broad refactors unrelated to the change.

## Testing Guidelines
Testing uses `pytest` (with `pytest-cov` in `tox`). Add tests under `tests/` using `test_*.py` naming and `test_*` function names. For new behavior, include focused unit coverage and,
when relevant, script-level regression checks similar to `tests/test_examples_kinda.py`. No strict coverage gate is configured; maintain or improve coverage in touched modules.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects (for example: `Add ...`, `Fix ...`, `Make ...`). Keep subjects concise and scoped to one logical change. For PRs, include:
- a clear summary of behavior changes,
- linked issue/context,
- test commands run and results,
- config/CLI examples for reproducibility,
- screenshots or logs when changing docs/visual outputs.

## Python naming + comments

### Compatibility
- Follow the repo’s established naming conventions (abbreviations, casing, prefixes/suffixes, domain terms).
- Prefer small, local improvements; avoid broad renames or reformatting unless necessary.

### Defaults
- Prefer clear, literal names over cleverness.
- Make code read like a sentence.

### Avoid
- Single-letter names except tiny, conventional scopes (`i`, `j`, `_`).
- Abbreviations unless standard in the repo/domain (`id`, `url`, `db`).
- Vague buckets: `utils`, `helpers`, `misc`, `common`, `manager`, `handler` (unless consistently used in the repo).

### Prefer
- Specific nouns for data: `user`, `invoice`, `request_payload`, `model_params`.
- Specific verbs for functions: `parse_config`, `load_checkpoint`, `compute_loss`, `write_report`.
- Boolean names that read naturally: `is_ready`, `has_permission`, `should_retry`, `needs_update`.
- Units when ambiguity is likely: `timeout_seconds`, `interval_ms`, `size_bytes`.

### When naming is hard
- Treat it as a design smell: split responsibilities, extract a sharper function/class, name the general concept plainly and specialize the variants.

## Function design philosophy

- Prefer code that reads like a clear recipe at the right level of abstraction.
- Use function boundaries to express meaning, not just mechanics.
- Split functions when they perform distinct, friend-level actions.
- Keep orchestration/composition functions when they represent one coherent workflow step.
- Avoid splitting purely for ideological SRP if it harms readability or flow.
- Name functions by intent and outcome, not internal implementation details.
- If a name needs "AND" to describe what it does, treat that as a prompt to check whether it should be split.

## Docstring style

- Use one sentence per function docstring.
- Describe what the function does in plain language.
- Use `AND` only when the function truly performs multiple distinct actions.
- Do not use `AND` for incidental plumbing or internal setup details.
- Prefer recipe-style phrasing for orchestration functions.

### Comments
- Default: no inline comments.
- Refactor for readability instead (names, constants, small functions, types).
- Comment only for **why** (perf/constraints) or to cite an algorithm/math source.
- Use docstrings for public APIs (usage + expectations), not internal narration.
