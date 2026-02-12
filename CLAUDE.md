# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**egt-pop** is derived from [JaxUED](https://github.com/DramaCow/jaxued), an Unsupervised Environment Design (UED) library in JAX. The base library provides single-file, understandable implementations of UED algorithms (DR, PLR, ACCEL, PAIRED).

This repo extends JaxUED with:
- **New environments**: T-mazes and other environments (potentially based on [ReMiDi](https://github.com/Michael-Beukman/ReMiDi))
- **New algorithms**: Modifications to PLR and PAIRED

## Git Setup

- `origin` → this repo (egt-pop)
- `upstream` → original jaxued

To pull upstream updates: `git fetch upstream && git merge upstream/main`

## Common Commands

### Environment setup
Create the venv with system Python (not conda) so it works in all execution
contexts (terminal, Cursor, Codex, sandboxed agents):
```bash
/usr/bin/python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[examples]"
.venv/bin/pip install "jax[cuda12]"  # GPU support
```

### Running Training Examples
```bash
python examples/maze_dr.py                        # Domain Randomization
python examples/maze_plr.py                       # Robust PLR
python examples/maze_plr.py --exploratory_grad_updates  # PLR
python examples/maze_plr.py --use_accel           # ACCEL
python examples/maze_paired.py                    # PAIRED
```

### Evaluation
```bash
python examples/maze_plr.py --mode eval --checkpoint_directory=./checkpoints/<run_name>/<seed> --checkpoint_to_eval <update_step>
```

### Testing
```bash
./.venv/bin/python -m pytest -v -s   # Run tests
WANDB_MODE=disabled ./.venv/bin/python -m pytest   # Run without wandb
WANDB_MODE=disabled ./.venv/bin/python -m pytest tests/test_dr_equivalence.py -q  # Fast DR v1/v2 equivalence checks
RUN_SLOW_EQUIVALENCE_TESTS=1 WANDB_MODE=disabled ./.venv/bin/python -m pytest tests/test_dr_equivalence.py -q -k "runtime_is_within_tolerance or tensor_footprint"  # Optional slower compute-equivalence checks
```

## Architecture

### Core Components (`src/jaxued/`)

**UnderspecifiedEnv** (`environments/underspecified_env.py`): Base interface for UPOMDP environments. Subclass and implement:
- `step_env(rng, state, action, params)` - state transition
- `reset_env_to_level(rng, level, params)` - reset to specific level
- `action_space(params)` - return action space

**LevelSampler** (`level_sampler.py`): PLR/ACCEL level buffer management. Stateless design - pass a `sampler` dict to all operations. Handles:
- Level storage with scores and timestamps
- Prioritized sampling (rank or topk)
- Staleness weighting
- Duplicate checking

**Wrappers** (`wrappers/`):
- `AutoReplayWrapper` - auto-reset to same level on episode end
- `AutoResetWrapper` - auto-reset to new random level

### Maze Environment (`environments/maze/`)
Minigrid-style maze with partial observability. Key files:
- `env.py` - main Maze class
- `level.py` - Level dataclass and generators
- `renderer.py` - visualization

### Example Scripts Pattern
Examples in `examples/` are single-file, self-contained implementations including:
- PPO with RNN policy
- Environment setup
- Training loop with wandb logging
- Checkpointing with orbax

To add new environments: subclass `UnderspecifiedEnv` and replace the env initialization in example files.

## Key Design Patterns

- **Stateless JAX style**: Classes don't store state; pass state dicts to methods
- **vmapped parallelism**: Training vectorizes over many environments
- **Single-file implementations**: Each algorithm is fully contained for readability
- **Flax structs**: Use `@struct.dataclass` for JAX-compatible state objects

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

### Recipe readability (preferred style)
- Keep orchestration code shallow: avoid deep nesting and long inline setup blocks.
- At high levels, prefer English component names (`initialize_*`, `build_*`, `run_*`) over abbreviations.
- Isolate wiring/plumbing in small setup helpers so `main` and train/eval loops read as step-by-step recipes.
- In orchestration flow, prefer explicit unpacking over tuple indexing when it improves readability.
- Prefer naming intermediate values before function calls instead of embedding non-trivial calculations directly in call arguments, when this does not add computational cost.

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
