import argparse
import sys

import pytest

from examples.config_utils import load_config


LEGACY_KEY_MAP = {
    "eval_num_attempts": "n_eval_attempts",
    "num_updates": "n_updates",
    "num_steps": "n_steps",
}


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--n_updates", type=int, default=30000)
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--n_eval_attempts", type=int, default=10)
    return parser


def _write_yaml(tmp_path, data):
    yaml = pytest.importorskip("yaml")
    config_path = tmp_path / "legacy_config.yaml"
    config_path.write_text(yaml.safe_dump(data))
    return config_path


def test_load_config_translates_legacy_yaml_keys(tmp_path, monkeypatch):
    config_path = _write_yaml(
        tmp_path,
        {
            "num_updates": 42,
            "num_steps": 128,
            "eval_num_attempts": 7,
        },
    )
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path)])

    config = load_config(
        _build_parser(),
        legacy_key_map=LEGACY_KEY_MAP,
        reject_unknown_yaml_keys=True,
    )

    assert config["n_updates"] == 42
    assert config["n_steps"] == 128
    assert config["n_eval_attempts"] == 7


def test_load_config_rejects_conflicting_legacy_and_canonical_keys(tmp_path, monkeypatch):
    config_path = _write_yaml(
        tmp_path,
        {
            "num_updates": 42,
            "n_updates": 99,
        },
    )
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path)])

    with pytest.raises(SystemExit):
        load_config(
            _build_parser(),
            legacy_key_map=LEGACY_KEY_MAP,
            reject_unknown_yaml_keys=True,
        )


def test_load_config_rejects_unknown_yaml_keys_when_enabled(tmp_path, monkeypatch):
    config_path = _write_yaml(tmp_path, {"unknown_key": 1})
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path)])

    with pytest.raises(SystemExit):
        load_config(
            _build_parser(),
            legacy_key_map=LEGACY_KEY_MAP,
            reject_unknown_yaml_keys=True,
        )
