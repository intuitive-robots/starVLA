"""
CoT (Chain-of-Thought) resolver for step-based annotation lookup.

Maps (trajectory_name, frame_index) → conversations using a pre-built JSONL mapping.
Trajectory names follow the lerobot convention: "{dataset_name}/{chunk_idx}/{file_idx}".
The mapping is built from annotations.jsonl via scripts/create_cot_mapping.py.

Return value of resolve() is a ShareGPT-style conversations list:
    [{"from": "human", "value": "<prompt with {instruction} placeholder>"},
     {"from": "gpt",   "value": "<assistant response with point/box>"}]

Backward compat: entries with legacy "cot_text" field are wrapped automatically.
"""

import json
from pathlib import Path
from typing import Optional


class CoTResolver:
    def resolve(self, trajectory_name: str, frame_index: int) -> Optional[list]:
        raise NotImplementedError


class NullCoTResolver(CoTResolver):
    """Always returns None — no CoT supervision (backward-compatible default)."""

    def resolve(self, trajectory_name: str, frame_index: int) -> Optional[list]:
        return None


class MappingCoTResolver(CoTResolver):
    """
    Step-based CoT lookup from a JSONL mapping file.

    Each line in the mapping file must have:
        trajectory_name  str        e.g. "cylinder_cube_full/0/1"
        start_frame      int        inclusive
        end_frame        int        inclusive
        conversations    list[dict] ShareGPT format: [{from, value}, {from, value}]
                         — OR legacy "cot_text" str (auto-wrapped for backward compat)

    For a given (trajectory_name, frame_index), returns the conversations list of the
    first interval [start_frame, end_frame] that contains frame_index, or None if no
    interval matches (holdout / unannotated steps → no CoT loss).
    """

    def __init__(self, mapping_path: str):
        mapping_path = Path(mapping_path)
        if not mapping_path.exists():
            raise FileNotFoundError(f"CoT mapping file not found: {mapping_path}")

        # index: trajectory_name → sorted list of (start_frame, end_frame, conversations)
        self._index: dict[str, list[tuple[int, int, list]]] = {}
        with open(mapping_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = entry["trajectory_name"]

                if "conversations" in entry:
                    conversations = entry["conversations"]
                else:
                    # Backward compat: wrap legacy cot_text as gpt-only conversation
                    conversations = [
                        {"from": "human", "value": "{instruction}"},
                        {"from": "gpt",   "value": entry["cot_text"]},
                    ]

                interval = (int(entry["start_frame"]), int(entry["end_frame"]), conversations)
                self._index.setdefault(key, []).append(interval)

        for key in self._index:
            self._index[key].sort(key=lambda x: x[0])

        total_intervals = sum(len(v) for v in self._index.values())
        print(
            f"[CoTResolver] Loaded {total_intervals} intervals across "
            f"{len(self._index)} trajectories from {mapping_path}"
        )

    @property
    def human_prompt_template(self) -> Optional[str]:
        """Return the human turn template from the first mapping entry, or None."""
        for intervals in self._index.values():
            if intervals:
                convs = intervals[0][2]
                if convs and convs[0].get("from") == "human":
                    return convs[0]["value"]
        return None

    def resolve(self, trajectory_name: str, frame_index: int) -> Optional[list]:
        intervals = self._index.get(trajectory_name)
        if not intervals:
            return None
        for start, end, conversations in intervals:
            if start <= frame_index <= end:
                return conversations
        return None

    @property
    def num_trajectories(self) -> int:
        return len(self._index)


def assert_cot_prompt_consistent(resolver: CoTResolver, cfg) -> None:
    """
    Assert that the human prompt template stored in the mapping matches the
    CoT_prompt configured in datasets.vla_data.  Mismatches mean the model
    would see a different user question at inference (two-pass CoT generation)
    vs. what was used during mapping creation.

    Only runs when resolver is a MappingCoTResolver with a recoverable template.
    """
    if not isinstance(resolver, MappingCoTResolver):
        return
    mapping_template = resolver.human_prompt_template
    if mapping_template is None:
        return

    vla_data = getattr(getattr(cfg, "datasets", None), "vla_data", None)
    if vla_data is None:
        return
    config_prompt = vla_data.get("CoT_prompt") if hasattr(vla_data, "get") else getattr(vla_data, "CoT_prompt", None)
    if config_prompt is None:
        return

    assert mapping_template == config_prompt, (
        f"\n[CoTResolver] Human prompt template mismatch!\n"
        f"  mapping file : {mapping_template!r}\n"
        f"  CoT_prompt   : {config_prompt!r}\n"
        f"Regenerate the CoT mapping with --human_prompt matching CoT_prompt in the YAML, "
        f"or update CoT_prompt in the YAML to match."
    )


def build_cot_resolver(cfg) -> CoTResolver:
    """
    Factory — reads datasets.vla_data.cot config block and returns
    the appropriate resolver. Falls back to NullCoTResolver when the
    block is absent or source is 'none'.

    Config schema (all optional):
        datasets:
          vla_data:
            cot:
              source: mapping      # 'mapping' | 'none'  (default: none)
              mapping_path: path/to/cot_mapping.jsonl
              loss_scale: 0.1      # consumed in train_starvla.py, not here
    """
    cot_cfg = getattr(getattr(cfg, "datasets", None), "vla_data", None)
    cot_cfg = getattr(cot_cfg, "cot", None) if cot_cfg is not None else None

    if cot_cfg is None:
        return NullCoTResolver()

    source = cot_cfg.get("source", "none") if hasattr(cot_cfg, "get") else getattr(cot_cfg, "source", "none")
    if source == "none" or source is None:
        return NullCoTResolver()

    if source == "mapping":
        mapping_path = cot_cfg.get("mapping_path") if hasattr(cot_cfg, "get") else getattr(cot_cfg, "mapping_path")
        if not mapping_path:
            raise ValueError("cot.source='mapping' requires cot.mapping_path to be set")
        return MappingCoTResolver(mapping_path)

    raise ValueError(f"Unknown cot.source: '{source}'. Expected 'mapping' or 'none'.")
