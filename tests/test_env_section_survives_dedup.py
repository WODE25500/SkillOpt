"""Regression: the ``env`` section must survive layer-format dedup.

``_resolve_layer_format_duplicates`` pops a flat key whenever the
equivalent structured key is present. For the mapping ``env.name -> env``
the flat key is the section name itself, so popping it deleted the entire
``env`` block from every structured config (base and child) before
inheritance — ``env.name``, ``env.split_dir``, ``env.skill_init``, etc.
were silently lost, which broke ``load_config`` for every environment.

The same section-name collision also applied to
``_drop_base_keys_overridden_by_layer``, which deleted a base's whole
``env`` section whenever a child overrode ``env.name``.
"""

from __future__ import annotations

from skillopt.config import _load_yaml, flatten_config, load_config


def test_env_section_survives_dedup_in_base(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "env:\n"
        "  name: searchqa\n"
        "  split_mode: split_dir\n"
        "  split_dir: data/searchqa_split\n"
        "  workers: 24\n",
        encoding="utf-8",
    )
    cfg = load_config(str(base))
    assert cfg["env"]["name"] == "searchqa"
    assert cfg["env"]["split_dir"] == "data/searchqa_split"


def test_env_section_survives_child_inheritance(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "env:\n"
        "  name: base\n"
        "  split_mode: ratio\n"
        "  workers: 4\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "_base_: base.yaml\n"
        "env:\n"
        "  name: pricewatch\n"
        "  split_mode: split_dir\n"
        "  split_dir: data/pricewatch_split\n",
        encoding="utf-8",
    )
    cfg = load_config(str(child))
    assert cfg["env"]["name"] == "pricewatch"
    assert cfg["env"]["split_mode"] == "split_dir"
    assert cfg["env"]["split_dir"] == "data/pricewatch_split"
    # Inherited env key is preserved alongside the child overrides.
    assert cfg["env"]["workers"] == 4


def test_flatten_config_keeps_env_keys(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        "env:\n"
        "  name: pricewatch\n"
        "  split_dir: data/pricewatch_split\n"
        "  max_completion_tokens: 2048\n",
        encoding="utf-8",
    )
    flat = flatten_config(load_config(str(config)))
    assert flat["env"] == "pricewatch"
    assert flat["split_dir"] == "data/pricewatch_split"
    assert flat["max_completion_tokens"] == 2048


def test_shipped_searchqa_config_still_loads_env():
    # Guard against regressions in the repo's own environment configs.
    cfg = _load_yaml("configs/searchqa/default.yaml")
    assert cfg["env"]["name"] == "searchqa"
    assert cfg["env"]["split_dir"] == "data/searchqa_split"
