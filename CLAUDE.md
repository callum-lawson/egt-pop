# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JaxUED is an Unsupervised Environment Design (UED) library in JAX. It provides single-file, understandable implementations of UED algorithms (DR, PLR, ACCEL, PAIRED) with a focus on easy modification and rapid prototyping.

## Common Commands

### Installation
```bash
pip install jaxued                    # Core package
pip install "jaxued[examples]"        # With example dependencies
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
pytest -v -s                          # Run tests
WANDB_MODE=disabled pytest            # Run without wandb
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
