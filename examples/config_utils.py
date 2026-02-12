"""Config loading utilities for experiment management.

Provides YAML config loading with CLI override support, automatic run naming,
and checkpoint collision detection.
"""

import os
import sys
from pathlib import Path


def _parser_dest_names(parser):
    """Return all argparse destination names defined on a parser."""
    return {action.dest for action in parser._actions}


def _apply_legacy_key_map(yaml_config, legacy_key_map):
    """Translate legacy YAML keys to parser keys and fail on ambiguous duplicates."""
    translated_config = dict(yaml_config)
    for legacy_key, canonical_key in legacy_key_map.items():
        if legacy_key not in translated_config:
            continue
        legacy_value = translated_config.pop(legacy_key)
        if canonical_key in translated_config and translated_config[canonical_key] != legacy_value:
            print(
                "ERROR: Config sets both legacy key "
                f"'{legacy_key}' and canonical key '{canonical_key}' with different values."
            )
            sys.exit(1)
        translated_config.setdefault(canonical_key, legacy_value)
    return translated_config


def _reject_unknown_yaml_keys(parser, yaml_config):
    """Fail fast when YAML contains keys not defined by the argument parser."""
    valid_keys = _parser_dest_names(parser)
    unknown_keys = sorted(set(yaml_config) - valid_keys)
    if unknown_keys:
        print(
            "ERROR: Unknown config keys: "
            + ", ".join(unknown_keys)
            + ". Update the YAML keys or add a legacy key mapping."
        )
        sys.exit(1)


def struct_from_dict(cls, d):
    """Construct a Flax struct dataclass from a flat dict.

    Extracts only the keys that match field names in ``cls``,
    ignoring everything else.

    Args:
        cls: A ``@struct.dataclass`` class
        d: Flat dict (e.g. wandb config)

    Returns:
        An instance of ``cls``
    """
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: d[k] for k in field_names if k in d})


def load_config(parser, legacy_key_map=None, reject_unknown_yaml_keys=False):
    """Load config from YAML file with CLI overrides.

    Adds --config and --force args to parser, loads YAML if provided,
    applies YAML values as parser defaults (CLI wins on conflicts),
    derives group_name and run_name, and checks for checkpoint collisions.

    Args:
        parser: argparse.ArgumentParser with experiment arguments defined
        legacy_key_map: Optional mapping from old YAML keys to current parser keys
        reject_unknown_yaml_keys: Whether to fail when YAML includes unsupported keys

    Returns:
        dict: Final config with all values resolved
    """
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/tmaze_500steps.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing checkpoint directory if it exists",
    )

    # First parse to get config file path
    args, remaining = parser.parse_known_args()

    if args.config is not None:
        yaml_config = _load_yaml(args.config)
        yaml_config = _apply_legacy_key_map(yaml_config, legacy_key_map or {})
        if reject_unknown_yaml_keys:
            _reject_unknown_yaml_keys(parser, yaml_config)

        # Set YAML values as defaults (CLI args will override)
        parser.set_defaults(**yaml_config)

    # Parse again with YAML defaults applied
    config = vars(parser.parse_args())

    # Derive names
    config = _derive_names(config, args.config)

    # Check for checkpoint collision (only when config file provided and in train mode)
    # Skip check when running with defaults (testing/experimentation)
    if args.config is not None and config.get("mode", "train") == "train":
        _check_checkpoint_collision(config)

    return config


def _load_yaml(config_path):
    """Load YAML config file.

    Args:
        config_path: Path to YAML file

    Returns:
        dict: Parsed YAML content
    """
    try:
        import yaml
    except ImportError:
        print("ERROR: pyyaml not installed. Run: pip install pyyaml")
        sys.exit(1)

    path = Path(config_path)
    if not path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _derive_names(config, config_path):
    """Derive group_name and run_name from config.

    If a config file is provided, names are derived from its filename.
    Otherwise, uses run_name from CLI or generates a default.

    Args:
        config: Parsed config dict
        config_path: Path to config file (or None)

    Returns:
        dict: Config with group_name and run_name set
    """
    if config_path is not None:
        # Derive from config filename: configs/tmaze_500steps.yaml -> tmaze_500steps
        config_name = Path(config_path).stem
        config["group_name"] = config_name
        # run_name can be overridden by CLI, otherwise derive from config + seed
        if config.get("run_name") is None:
            config["run_name"] = f"{config_name}_{config['seed']}"
    else:
        # No config file - use run_name from CLI or generate default
        if config.get("run_name") is None:
            config["run_name"] = f"run_{config['seed']}"
        config["group_name"] = config["run_name"]

    return config


def _check_checkpoint_collision(config):
    """Check if checkpoint directory already exists.

    Errors unless --force is set.

    Args:
        config: Config dict with run_name, seed, and force
    """
    checkpoint_dir = os.path.join(
        os.getcwd(), "checkpoints", config["run_name"], str(config["seed"])
    )

    if os.path.exists(checkpoint_dir) and not config.get("force", False):
        print(f"ERROR: Checkpoint directory already exists: {checkpoint_dir}")
        print("Use --force to overwrite, or use a different --seed or --run_name")
        sys.exit(1)
